import json
import math
import os
import re
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

import certifi
from dotenv import load_dotenv

from langgraph.config import get_config, get_stream_writer

from app.logger import get_logger


load_dotenv()

log = get_logger("stay")


def announce(label: str) -> None:
    """Tell the page what is being done right now ("Measuring the drive...").
    Does nothing when it isn't running inside a chat turn."""
    try:
        get_stream_writer()({"label": label})
    except Exception:
        pass

PREFERENCES_FILE = Path(__file__).resolve().parents[2] / "data" / "preferences.json"

STAY_API = "https://api.stayingapi.com/v1"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
RATES_URL = "https://api.frankfurter.app/latest"
USER_AGENT = "MeetPilot/1.0"

# StayingAPI charges at least this many credits for any search.
MIN_SEARCH_CREDITS = 5
POLL_SECONDS = 4
RATE_CACHE_SECONDS = 6 * 3600

# Python's default trust store on Windows rejects StayingAPI's certificate
# chain; certifi's list verifies it properly.
TLS = ssl.create_default_context(cafile=certifi.where())


class StayError(Exception):
    """A problem the user should hear about, with a readable message."""


# ---------------------------------------------------------------------------
# Preferences (data/preferences.json). Nothing about the user is hard coded.
# ---------------------------------------------------------------------------

def load_default_preferences() -> dict:
    """data/preferences.json: the values everyone starts with."""
    try:
        return json.loads(PREFERENCES_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise StayError("data/preferences.json was not found.")
    except ValueError as exc:
        raise StayError(f"data/preferences.json is not valid JSON: {exc}")


def with_user_settings(prefs: dict, saved: dict) -> dict:
    """The defaults with what one user changed laid over them. The search settings
    are kept inside hotel_preferences (under "stay_search") in the database."""
    out = {name: dict(section) if isinstance(section, dict) else section for name, section in prefs.items()}

    for name in ("home_location", "travel_preferences", "hotel_preferences"):
        changed = dict(saved.get(name) or {})
        search = changed.pop("stay_search", None) if name == "hotel_preferences" else None

        out[name] = {**(out.get(name) or {}), **changed}

        if search:
            out["stay_search"] = {**(out.get("stay_search") or {}), **search}

    return out


def current_user() -> str | None:
    """The user whose chat this code is running for, if it is running inside one."""
    try:
        return get_config()["configurable"].get("user_id")
    except (KeyError, RuntimeError):
        return None


def load_preferences(user_id: str | None = None) -> dict:
    """The preferences for this user: the defaults plus what they saved in Settings.
    Without a user (and outside a chat) it is just the defaults."""
    from app import db      # imported here because db imports this file

    prefs = load_default_preferences()
    user_id = user_id or current_user()

    return with_user_settings(prefs, db.get_user_settings(user_id)) if user_id else prefs


def stay_setting(prefs: dict, name: str):
    try:
        return prefs["stay_search"][name]
    except KeyError:
        raise StayError(f"Add stay_search.{name} to data/preferences.json.")


def default_stay_dates(meeting_date: str) -> tuple[str, str]:
    """The night before the meeting, checking out on the meeting day."""
    day = date.fromisoformat(meeting_date)

    return (day - timedelta(days=1)).isoformat(), day.isoformat()


def parse_stay_answer(answer: dict) -> tuple[str, str, int]:
    """Validate the dates and guests the user sent from the stay form."""
    try:
        check_in = date.fromisoformat(str(answer.get("check_in", "")))
        check_out = date.fromisoformat(str(answer.get("check_out", "")))
        guests = int(answer.get("guests", 1))
    except (TypeError, ValueError):
        raise StayError("Those stay dates or guest numbers don't look right.")

    if check_out <= check_in:
        raise StayError("The check-out date must be after the check-in date.")

    if guests < 1:
        raise StayError("At least one guest is needed.")

    return check_in.isoformat(), check_out.isoformat(), guests


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def get_json(url: str, headers: dict | None = None, timeout: int = 30):
    """GET a URL. Returns (status, parsed body); an error status from the
    server comes back as a normal result, connection problems raise."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json", **(headers or {})},
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout, context=TLS) as response:
            return response.status, json.loads(response.read().decode("utf-8"))

    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")

        try:
            return exc.code, json.loads(raw)
        except ValueError:
            return exc.code, raw


# ---------------------------------------------------------------------------
# Venue -> coordinates (Nominatim). Asked politely: one request per second.
# ---------------------------------------------------------------------------

_places = {}
_places_lock = threading.Lock()
_last_lookup = 0.0


# Parts of an address a map can't use: directions, and places inside a building.
DIRECTION = re.compile(r"^(near|opposite|opp\.?|behind|beside|next to|adjacent to|in front of|close to|off)\b", re.I)
INSIDE = re.compile(r"^(hall|gate|room|floor|level|block|wing|suite|booth|stall|ground floor|\d+(st|nd|rd|th)? floor)\b", re.I)

CITY_RADIUS_KM = 100    # a match farther than this from the meeting's city is some other place


def place_parts(query: str) -> tuple[list[str], list[str]]:
    """The parts of a place text, in order: as written (without repeats), and
    cleaned (no bracketed notes, no directions, no rooms inside a building)."""
    parts = []

    for part in (p.strip() for p in query.split(",")):
        if part and part.casefold() not in {x.casefold() for x in parts}:
            parts.append(part)

    texts = [re.sub(r"\s*\([^)]*\)", "", p).strip() or p for p in parts]

    # Directions ("near the bus stand") and rooms ("Hall 3") are no place of their own;
    # keep them only if nothing else is left.
    texts = [t for t in texts if not (DIRECTION.match(t) or INSIDE.match(t))] or texts

    plain = []

    for text in texts:
        if text.casefold() not in {x.casefold() for x in plain}:
            plain.append(text)

    return parts, plain


def place_variants(query: str) -> list[str]:
    """Other ways to write a place text, tried when the whole text finds
    nothing. Emails repeat the venue and the city, add notes in brackets such
    as "(BIEC)", and give directions or rooms that a map can't match. The
    cleaned text comes first, then the venue's bracketed name, then the
    original wording; each as the whole text, the venue with the part after
    it, and the venue alone."""
    parts, plain = place_parts(query)

    def shorter(texts: list[str]) -> list[str]:
        out = [", ".join(texts)]

        if len(texts) > 2:
            out.append(f"{texts[0]}, {texts[1]}")

        if len(texts) > 1:
            out.append(texts[0])

        return out

    # "WTC Mumbai (World Trade Centre)": the bracket may hold the name a map knows.
    named = []

    for note in re.findall(r"\(([^)]{4,})\)", parts[0] if parts else ""):
        named.append(f"{note.strip()}, {plain[1]}" if len(plain) > 1 else note.strip())

    seen = {query.strip().casefold()}
    variants = []

    for variant in shorter(plain) + named + shorter(parts):
        if variant and variant.casefold() not in seen:
            seen.add(variant.casefold())
            variants.append(variant)

    return variants


def area_variants(query: str) -> list[str]:
    """When the venue itself can't be found: its neighbourhood, then the town
    ("Electronics City, Bengaluru", then "Bengaluru"), with the venue dropped."""
    _, plain = place_parts(query)

    return [", ".join(plain[i:]) for i in range(1, len(plain) - 1)]


def resolve_place(query: str, city: str = "") -> dict | None:
    """The place on the map: {"lat", "lng", "name", "area_query"}, or None.

    The whole text is tried, then simpler versions of it. With a `city`, a
    match must be in that city (named in the result, or within CITY_RADIUS_KM
    of its centre), because a name alone can match a place of the same name
    anywhere in the world. If the venue can't be found, the neighbourhood and
    then the centre of the city are returned, marked "approximate": True, so a
    distance can still be judged and the user is told it is only roughly right.
    Without a `city` only exact matches are returned."""
    city = city.strip()
    centre = lookup_place(city) if city else None

    def acceptable(place: dict) -> bool:
        if not centre or city.casefold() in place["name"].casefold():
            return True

        return distance_km(place["lat"], place["lng"], centre["lat"], centre["lng"]) <= CITY_RADIUS_KM

    for variant in [query, *place_variants(query)]:
        place = lookup_place(variant)

        if place and acceptable(place):
            _places[query.strip().casefold()] = place
            return place

        # Not cached means the lookup itself failed (no network): don't retry.
        if variant.strip().casefold() not in _places:
            return None

    if not city:
        return None

    for variant in area_variants(query):
        place = lookup_place(variant)

        if place and acceptable(place):
            log.info("Using %r for %r (the exact place was not found)", variant, query)
            return {**place, "approximate": True}

    if centre:
        log.info("Using the centre of %r for %r (the exact place was not found)", city, query)
        return {**centre, "approximate": True}

    return None


def lookup_place(query: str) -> dict | None:
    """One Nominatim lookup. `area_query` is the town, state and country, for a wider search."""
    global _last_lookup

    key = query.strip().casefold()

    if key in _places:
        return _places[key]

    announce(f"Finding \u201c{query.split(',')[0].strip()}\u201d on the map\u2026")

    with _places_lock:
        wait = 1.0 - (time.monotonic() - _last_lookup)

        if wait > 0:
            time.sleep(wait)

        _last_lookup = time.monotonic()

        try:
            status, body = get_json(NOMINATIM_URL + "?" + urllib.parse.urlencode(
                {"q": query, "format": "jsonv2", "limit": 1, "addressdetails": 1, "accept-language": "en"}
            ))
        except (urllib.error.URLError, TimeoutError) as exc:
            log.warning("Nominatim lookup failed for %r: %s", query, exc)
            return None

    if status != 200 or not isinstance(body, list):
        log.warning("Nominatim answered %s for %r", status, query)
        return None

    place = None

    if body:
        address = body[0].get("address") or {}
        area = next((address[k] for k in ("city", "town", "village", "municipality", "county", "state_district") if address.get(k)), "")
        wider = ", ".join(part for part in (area, address.get("state"), address.get("country")) if part)

        place = {
            "lat": float(body[0]["lat"]),
            "lng": float(body[0]["lon"]),
            "name": body[0]["display_name"],
            "area_query": wider,
        }

    _places[key] = place
    log.info("Resolved %r -> %s", query, place["name"] if place else "not found")

    return place


# ---------------------------------------------------------------------------
# Currency conversion
# ---------------------------------------------------------------------------

_rates = {}


def exchange_rate(source: str, target: str) -> tuple[float, str] | None:
    """(rate, date of the rate) from a live rate service, cached for a few
    hours. None if the rate can't be fetched."""
    cached = _rates.get((source, target))

    if cached and cached[0] > time.time():
        return cached[1], cached[2]

    try:
        status, body = get_json(RATES_URL + "?" + urllib.parse.urlencode({"from": source, "to": target}))
        rate = body["rates"][target]
        day = body.get("date", "")
    except (urllib.error.URLError, TimeoutError, KeyError, TypeError) as exc:
        log.warning("Could not get the %s->%s rate: %s", source, target, exc)
        return None

    _rates[(source, target)] = (time.time() + RATE_CACHE_SECONDS, rate, day)
    log.info("Rate %s->%s = %s (%s)", source, target, rate, day)

    return rate, day


def convert(amount: float, source: str, target: str) -> float | None:
    """Convert with a live rate, or None if the rate can't be fetched."""
    if source == target:
        return amount

    found = exchange_rate(source, target)

    return None if found is None else amount * found[0]


# ---------------------------------------------------------------------------
# StayingAPI
# ---------------------------------------------------------------------------

def stay_get(path: str, params: dict | None = None):
    key = os.getenv("STAYINGAPI_KEY")

    if not key:
        raise StayError("STAYINGAPI_KEY is not set in .env.")

    url = STAY_API + path + ("?" + urllib.parse.urlencode(params) if params else "")

    try:
        return get_json(url, {"Authorization": "Bearer " + key}, timeout=60)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise StayError(f"Could not reach StayingAPI: {exc}") from exc


def api_error(status: int, body) -> str:
    if status == 401:
        return "StayingAPI rejected the key (401). Check STAYINGAPI_KEY in .env."

    if status == 429:
        return "StayingAPI says we're searching too fast (429). Try again in a minute."

    detail = body if isinstance(body, str) else json.dumps(body)

    return f"StayingAPI answered {status}: {detail[:200]}"


def credits_available() -> int:
    status, body = stay_get("/account")

    if status != 200:
        raise StayError(api_error(status, body))

    return body["data"]["credits"]["available"]


def search_stays(location, check_in, check_out, guests, currency, settings, progress=None) -> list[dict]:
    """Search Booking.com hotels, always live: prices and availability change,
    so results are never reused. Searches are asynchronous: the API hands back
    a job that is polled (for free) until it finishes."""
    limit = int(settings["results_per_search"])
    reserve = int(settings["credit_reserve"])

    # Never let a search take the balance below the reserve.
    available = credits_available()
    cost = max(MIN_SEARCH_CREDITS, limit)

    if available - cost < reserve:
        raise StayError(
            f"Only {available} StayingAPI credits are left, and a search costs about {cost}. "
            f"I keep {reserve} in reserve, so I won't search."
        )

    params = {
        "location": location, "checkIn": check_in, "checkOut": check_out,
        "adults": guests, "limit": limit, "platforms": "booking",
    }

    if currency:
        params["currency"] = currency

    status, body = stay_get("/search", params)

    if status == 200:
        results = body["data"]
    elif status == 202:
        results = wait_for_job(body["data"]["jobId"], int(settings["wait_seconds"]), progress)
    else:
        raise StayError(api_error(status, body))

    log.info("Hotel search for %r: %d result(s), credits before=%s", location, len(results), available)

    return results


def wait_for_job(job_id: str, wait_seconds: int, progress=None) -> list[dict]:
    started = time.monotonic()

    while True:
        elapsed = int(time.monotonic() - started)

        if elapsed >= wait_seconds:
            raise StayError(
                f"The hotel search is taking longer than {wait_seconds} seconds. Please try again in a moment."
            )

        time.sleep(POLL_SECONDS)

        if progress:
            progress(f"Searching hotels... {int(time.monotonic() - started)}s")

        status, body = stay_get(f"/jobs/{job_id}")

        if status != 200:
            raise StayError(api_error(status, body))

        job = body["data"]

        if job["status"] == "completed":
            result = job["result"]
            return result if isinstance(result, list) else (result.get("data") or [])

        if job["status"] == "failed":
            error = job.get("error") or {}
            raise StayError(f"The hotel search failed: {error.get('message', 'unknown error')}")


# ---------------------------------------------------------------------------
# Ranking against the preferences in the JSON file
# ---------------------------------------------------------------------------

def distance_km(lat1, lng1, lat2, lng2) -> float:
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lng2 - lng1) * p / 2) ** 2)

    return 12742 * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Is a stay worth suggesting? Home -> venue distance and driving time (OSRM,
# a free public routing service), then a rule from preferences.json.
# ---------------------------------------------------------------------------

OSRM_URL = "https://router.project-osrm.org/route/v1/driving"

_routes = {}


def travel_setting(prefs: dict, name: str):
    try:
        return prefs["travel_preferences"][name]
    except KeyError:
        raise StayError(f"Add travel_preferences.{name} to data/preferences.json.")


def drive_route(origin: dict, destination: dict, fallback_speed_kmh: float, time_factor: float = 1.0) -> dict:
    """{"distance_km", "minutes", "road"}: by road when OSRM answers, otherwise
    the straight-line distance at an assumed speed (road is then False).
    OSRM times assume free-flowing roads, so a real drive takes longer;
    `time_factor` (travel_preferences.driving_time_factor) scales them up."""
    key = tuple(round(v, 4) for v in (origin["lat"], origin["lng"], destination["lat"], destination["lng"]))

    if key in _routes:
        return {**_routes[key], "minutes": round(_routes[key]["minutes"] * time_factor)}

    announce("Measuring the drive\u2026")

    try:
        status, body = get_json(
            f"{OSRM_URL}/{origin['lng']},{origin['lat']};{destination['lng']},{destination['lat']}?overview=false",
            timeout=10,
        )
        road = body["routes"][0] if status == 200 and isinstance(body, dict) and body.get("routes") else None

        if road:
            _routes[key] = {"distance_km": round(road["distance"] / 1000, 1), "minutes": round(road["duration"] / 60), "road": True}
            return {**_routes[key], "minutes": round(_routes[key]["minutes"] * time_factor)}

        log.warning("OSRM answered %s: using the straight-line distance", status)

    except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as exc:
        log.warning("OSRM lookup failed: %s. Using the straight-line distance.", exc)

    # Not cached, so the next request tries the routing service again.
    straight = distance_km(origin["lat"], origin["lng"], destination["lat"], destination["lng"])

    return {"distance_km": round(straight, 1), "minutes": round(straight / fallback_speed_kmh * 60), "road": False}


def hours_text(minutes: int) -> str:
    hours, rest = divmod(int(minutes), 60)

    return f"{hours} h {rest} min" if hours else f"{rest} min"


def assess_trip(prefs: dict, venue: dict | None, meeting_date: str, meeting_time: str, venue_name: str) -> dict | None:
    """Should a stay be suggested? A stay is recommended when the venue is
    farther than the limit, or when arriving on time would mean leaving
    before the earliest reasonable hour. Returns None when home or the venue
    can't be placed on the map, and the trip can't be judged."""
    home = prefs.get("home_location") or {}
    home_place = resolve_place(
        ", ".join(p for p in (home.get("area"), home.get("city"), home.get("state"), home.get("country")) if p),
        home.get("city", ""),
    )

    if not (home_place and venue):
        return None

    threshold = float(travel_setting(prefs, "stay_distance_threshold_km"))
    earliest = datetime.strptime(travel_setting(prefs, "earliest_reasonable_departure"), "%H:%M").time()
    buffer = int(travel_setting(prefs, "arrival_buffer_minutes"))

    route = drive_route(
        home_place, venue, float(travel_setting(prefs, "fallback_speed_kmh")),
        float(travel_setting(prefs, "driving_time_factor")),
    )
    where = f"{venue_name} is about {route['distance_km']:g} km from {home.get('city', 'home')}"

    if venue.get("approximate"):
        where = (f"I couldn't find {venue_name} on the map, but the centre of {venue['name'].split(',')[0]} "
                 f"is about {route['distance_km']:g} km from {home.get('city', 'home')}")

    if route["road"]:
        summary = f"{where} by road, roughly {hours_text(route['minutes'])} of driving."
    else:
        summary = f"{where} in a straight line, roughly {hours_text(route['minutes'])} of driving (a rough estimate: the route service didn't answer)."
    reasons = []
    leave_at, day_before = "", False

    if route["distance_km"] > threshold:
        reasons.append(f"it is over your {threshold:g} km limit")

    if meeting_time:
        start = datetime.combine(date.fromisoformat(meeting_date), datetime.strptime(meeting_time, "%H:%M").time())
        leave = start - timedelta(minutes=route["minutes"] + buffer)
        day_before = leave.date() < start.date()
        leave_at = f"{leave:%H:%M}"

        summary += f" To arrive {buffer} minutes before the {meeting_time} start, you would leave around {leave:%H:%M}{' the day before' if day_before else ''}."

        if day_before or leave.time() < earliest:
            reasons.append(f"you would have to leave before {earliest:%H:%M}")

    log.info(
        "Trip: %s (%.4f, %.4f) -> %s (%.4f, %.4f): %s km, %d min, %s",
        home_place["name"].split(",")[0], home_place["lat"], home_place["lng"],
        venue["name"].split(",")[0], venue["lat"], venue["lng"],
        route["distance_km"], route["minutes"], "OSRM road route" if route["road"] else "straight-line estimate",
    )

    summary += (f" An overnight stay may be convenient: {' and '.join(reasons)}." if reasons
                else " That looks like a same-day trip.")

    return {"summary": summary, "recommended": bool(reasons), "leave_at": leave_at, "day_before": day_before, "buffer_minutes": buffer, "reasons": reasons,
            "home": home_place, "venue": venue, **route}


def nice_date(text) -> str:
    """"2026-10-15" as "Thu 15 Oct"; anything else is returned unchanged."""
    try:
        return date.fromisoformat(str(text)).strftime("%a %d %b")
    except ValueError:
        return str(text or "")


def itinerary(meeting: dict, place: str, home_city: str, trip: dict | None, stay: dict) -> list[tuple[str, str]]:
    """The plan for an in-person meeting as (label, text) rows. `stay` is
    {"kind": "hotel" | "own" | "same_day"} and, for a hotel, the hotel and the
    search it came from."""
    when = " · ".join(part for part in (nice_date(meeting.get("date")), meeting.get("time")) if part)
    rows = [("Meeting", f"{meeting['title']} · {when}"), ("Where", place)]

    route = ""

    if trip:
        route = f"about {trip['distance_km']:g} km from {home_city or 'home'}, roughly {hours_text(trip['minutes'])} of driving"

        if not trip["road"]:
            route += " (an estimate)"

    if stay["kind"] == "hotel":
        hotel, search = stay["hotel"], stay.get("search", {})
        price = f"{hotel['price_per_night']:g} {hotel['currency']}/night" if hotel.get("price_per_night") is not None else "price unknown"
        rows.append(("Stay", f"{hotel['name']} · {price} · {nice_date(search.get('check_in'))} to {nice_date(search.get('check_out'))} · {search.get('guests')} guest(s)"))
        rows.append(("Travel", f"Arrive on {nice_date(search.get('check_in'))}: {route}." if route else "Travel time couldn't be worked out."))

    elif stay["kind"] == "own":
        rows.append(("Stay", "You'll arrange your own stay."))
        rows.append(("Travel", f"{route[0].upper() + route[1:]}." if route else "Travel time couldn't be worked out."))

    else:
        rows.append(("Stay", "None: a same-day trip."))
        leave = f" Leave around {trip['leave_at']}{' the day before' if trip['day_before'] else ''}." if trip and trip.get("leave_at") else ""
        rows.append(("Travel", f"{route[0].upper() + route[1:]}.{leave}" if route else "Travel time couldn't be worked out."))

    return rows


def link_with_dates(url: str | None, check_in: str, check_out: str, guests: int) -> str | None:
    """The hotel's Booking.com page for the dates and guests that were
    searched. A bare link opens with whatever dates the visitor used last, so
    the page could say "unavailable" for dates other than the ones we searched."""
    if not url or not url.startswith("https://www.booking.com/"):
        return url

    parts = urllib.parse.urlsplit(url)
    query = dict(urllib.parse.parse_qsl(parts.query))
    query.update({"checkin": check_in, "checkout": check_out, "group_adults": guests, "no_rooms": 1, "group_children": 0})

    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(query)))


def rank_stays(stays: list[dict], prefs: dict, venue: dict | None, nights: int) -> dict:
    """Apply the budget as a hard limit, and the other preferences as flags.
    Hotels over the budget are dropped; the rest are ordered by how few
    preferences they miss, then by price."""
    hp = prefs.get("hotel_preferences") or {}
    currency = hp.get("currency")
    max_price = hp.get("max_price_per_night")
    min_rating = hp.get("minimum_rating")
    rating_scale = hp.get("rating_scale")
    max_distance = hp.get("preferred_distance_from_venue_km")

    hotels = []
    excluded = 0
    cheapest_excluded = None

    for stay in stays:
        price = stay.get("price") or {}
        nightly = price.get("nightlyPrice")
        shown = price.get("currency")
        flags = []
        converted_from = None

        # The search gives no "available" flag, only a price: no price for these dates usually means none free.
        if nightly is None:
            flags.append("No price for these dates, it may be unavailable")

        # Convert to the preference currency before checking the budget.
        if nightly is not None and currency and shown and shown != currency:
            converted = convert(nightly, shown, currency)

            if converted is None:
                flags.append(f"Price is in {shown}; couldn't convert it to {currency}")
            else:
                converted_from = f"{nightly:g} {shown}"
                nightly, shown = round(converted), currency

        if max_price is not None and nightly is not None and shown == currency and nightly > max_price:
            excluded += 1
            cheapest_excluded = nightly if cheapest_excluded is None else min(cheapest_excluded, nightly)
            continue

        rating = None
        guest, source_scale = stay.get("guestRating"), stay.get("ratingScale")

        if guest is not None and source_scale and rating_scale:
            rating = round(guest * rating_scale / source_scale, 2)

            if min_rating is not None and rating < min_rating:
                flags.append(f"Rating {rating:g}/{rating_scale} is below your {min_rating:g}")

        location = stay.get("location") or {}
        distance = None

        if venue and location.get("lat") is not None and location.get("lng") is not None:
            distance = round(distance_km(venue["lat"], venue["lng"], location["lat"], location["lng"]), 1)

            if max_distance is not None and distance > max_distance:
                flags.append(f"{distance:g} km from the venue, over your {max_distance:g} km")

        breakfast = "breakfast" in (stay.get("amenities") or [])

        if hp.get("breakfast_required") and not breakfast:
            flags.append("No breakfast")

        hotels.append({
            "name": stay.get("name"),
            "stars": stay.get("starRating"),
            "rating": rating,
            "rating_scale": rating_scale,
            "reviews": stay.get("reviewCount"),
            "price_per_night": nightly,
            "total_price": nightly * nights if nightly is not None and shown == currency else price.get("totalPrice"),
            "currency": shown,
            "converted_from": converted_from,
            "distance_km": distance,
            "breakfast": breakfast,
            "address": location.get("address"),
            "url": stay.get("url"),
            "image": (stay.get("images") or [None])[0],
            "flags": flags,
        })

    hotels.sort(key=lambda h: (
        len(h["flags"]),
        h["price_per_night"] if h["price_per_night"] is not None else float("inf"),
        h["distance_km"] if h["distance_km"] is not None else float("inf"),
    ))

    return {"hotels": hotels, "excluded": excluded, "cheapest_excluded": cheapest_excluded}


# ---------------------------------------------------------------------------
# The whole search, as the Booking agent's stay branch calls it
# ---------------------------------------------------------------------------

def search_hotels_for(place: str, stay: dict, fallback_area: str = "", progress=None) -> dict:
    """Resolve the place, search, and rank. If nothing is listed right at the
    place, retry once with the wider town. Returns what the chat shows."""
    prefs = load_preferences()

    # What the user asked for in the chat ("show up to 5k") wins over their saved preference, for this search.
    overrides = {k: v for k, v in (stay.get("overrides") or {}).items() if k in ("max_price_per_night", "minimum_rating", "breakfast_required")}
    prefs = {**prefs, "hotel_preferences": {**(prefs.get("hotel_preferences") or {}), **overrides}}
    hp = prefs["hotel_preferences"]

    check_in, check_out, guests = parse_stay_answer(stay)
    nights = (date.fromisoformat(check_out) - date.fromisoformat(check_in)).days

    settings = {
        name: stay_setting(prefs, name)
        for name in ("results_per_search", "credit_reserve", "wait_seconds", "more_results")
    }

    # "Search more hotels" asks for a longer list than the first search did.
    if stay.get("limit"):
        settings["results_per_search"] = int(stay["limit"])

    venue = resolve_place(place)
    currency = hp.get("currency")

    stays = search_stays(place, check_in, check_out, guests, currency, settings, progress)
    widened_to = None

    if not stays:
        wider = (venue or {}).get("area_query") or fallback_area

        if wider and wider.strip().casefold() != place.strip().casefold():
            if progress:
                progress("Nothing listed right there, trying the wider area...")

            stays = search_stays(wider, check_in, check_out, guests, currency, settings, progress)
            widened_to = wider if stays else None

    ranked = rank_stays(stays, prefs, venue, nights)

    for hotel in ranked["hotels"]:
        hotel["url"] = link_with_dates(hotel["url"], check_in, check_out, guests)

    return {
        "hotels": ranked["hotels"],
        "found": len(stays),
        "excluded": ranked["excluded"],
        "cheapest_excluded": ranked["cheapest_excluded"],
        "place": place,
        "venue": venue["name"] if venue else None,
        "widened_to": widened_to,
        "check_in": check_in,
        "check_out": check_out,
        "nights": nights,
        "guests": guests,
        "max_price": hp.get("max_price_per_night"),
        "currency": currency,
        "next_cost": max(MIN_SEARCH_CREDITS, int(settings["results_per_search"]) + int(settings["more_results"])),
    }
