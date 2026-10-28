"""Each person's preferences: what the Settings popup reads and saves.

The defaults come from data/preferences.json. A person's own values are stored in
user_settings, and only the ones that differ from the defaults, so a default changed in
the file still reaches everyone who never touched that field."""
import re

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from app import auth, db
from app.tools.stay_tool import StayError, load_default_preferences, with_user_settings


router = APIRouter()

# kind, then the limits: text (max length), number / int (min, max), time, currency, bool
FIELDS = {
    "home_location": {
        "city": ("text", 80), "area": ("text", 80), "state": ("text", 80), "country": ("text", 80),
    },
    "travel_preferences": {
        "stay_distance_threshold_km": ("number", 1, 5000),
        "earliest_reasonable_departure": ("time",),
        "arrival_buffer_minutes": ("int", 0, 600),
        "fallback_speed_kmh": ("number", 10, 200),
        "driving_time_factor": ("number", 1, 3),
    },
    "hotel_preferences": {
        "max_price_per_night": ("number", 1, 10_000_000),
        "currency": ("currency",),
        "minimum_rating": ("number", 0, 5),
        "preferred_distance_from_venue_km": ("number", 0, 200),
        "breakfast_required": ("bool",),
        "preferred_room_type": ("text", 40),
    },
    "stay_search": {
        "results_per_search": ("int", 1, 20),
        "more_results": ("int", 1, 20),
    },
}

REQUIRED = {("home_location", "city")}


class Invalid(ValueError):
    pass


def clean(kind: tuple, value):
    """The value in its proper type, or Invalid with a sentence the user can read."""
    what = kind[0]

    if what == "bool":
        if not isinstance(value, bool):
            raise Invalid("must be on or off")
        return value

    if what == "text":
        if not isinstance(value, str):
            raise Invalid("must be text")
        value = value.strip()
        if len(value) > kind[1]:
            raise Invalid(f"must be at most {kind[1]} characters")
        return value

    if what == "time":
        if not isinstance(value, str) or not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", value.strip()):
            raise Invalid("must be a time like 06:00")
        return value.strip()

    if what == "currency":
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z]{3}", value.strip()):
            raise Invalid("must be a 3-letter code like INR")
        return value.strip().upper()

    # number / int
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Invalid("must be a number")
    if what == "int":
        if value != int(value):
            raise Invalid("must be a whole number")
        value = int(value)
    if not kind[1] <= value <= kind[2]:
        raise Invalid(f"must be between {kind[1]:g} and {kind[2]:g}")
    return value


def validate(body) -> dict:
    """The whole form, checked. Returns {section: {field: value}}; raises Invalid with
    {"section.field": message}."""
    problems = {}
    out = {}

    if not isinstance(body, dict):
        raise HTTPException(400, "Send the settings as an object.")

    for section, fields in FIELDS.items():
        given = body.get(section)
        if not isinstance(given, dict):
            problems[section] = "is missing"
            continue

        for extra in set(given) - set(fields):
            problems[f"{section}.{extra}"] = "is not a setting"

        out[section] = {}
        for name, kind in fields.items():
            if name not in given:
                problems[f"{section}.{name}"] = "is missing"
                continue
            try:
                value = clean(kind, given[name])
                if (section, name) in REQUIRED and value == "":
                    raise Invalid("is required")
                out[section][name] = value
            except Invalid as exc:
                problems[f"{section}.{name}"] = str(exc)

    if problems:
        raise Invalid(problems)

    return out


def current_values(user_id: str) -> tuple[dict, dict]:
    """(this user's values, the defaults), both shaped like FIELDS."""
    defaults = load_default_preferences()
    merged = with_user_settings(defaults, db.get_user_settings(user_id))

    def pick(prefs):
        return {s: {f: prefs.get(s, {}).get(f) for f in fields} for s, fields in FIELDS.items()}

    return pick(merged), pick(defaults)


def _need_header(request: Request) -> None:
    if request.headers.get("X-MeetPilot") != "1":
        raise HTTPException(403, "Missing header.")


def _answer(values: dict, defaults: dict) -> JSONResponse:
    return JSONResponse({"values": values, "defaults": defaults}, headers={"Cache-Control": "no-store"})


@router.get("/api/settings")
def read_settings(user: dict = Depends(auth.require_user)):
    try:
        values, defaults = current_values(user["id"])
    except StayError as exc:
        raise HTTPException(500, str(exc))
    return _answer(values, defaults)


@router.put("/api/settings")
async def save_settings(request: Request, user: dict = Depends(auth.require_user)):
    _need_header(request)

    try:
        wanted = validate(await request.json())
    except ValueError as exc:
        problems = exc.args[0] if exc.args and isinstance(exc.args[0], dict) else {"": "Send valid JSON."}
        return JSONResponse({"detail": "Some settings are not valid.", "fields": problems}, status_code=422)

    defaults = current_values(user["id"])[1]

    # Keep only what differs from the defaults. The search settings ride along in hotel_preferences.
    changed = {s: {f: v for f, v in fields.items() if v != defaults[s][f]} for s, fields in wanted.items()}
    hotel = {**changed["hotel_preferences"]}
    if changed["stay_search"]:
        hotel["stay_search"] = changed["stay_search"]

    try:
        db.save_user_settings(user["id"], changed["home_location"], changed["travel_preferences"], hotel)
    except db.AccountsUnavailable as exc:
        raise HTTPException(503, str(exc))

    return _answer(*current_values(user["id"]))


@router.delete("/api/settings")
def reset_settings(request: Request, user: dict = Depends(auth.require_user)):
    """Back to the defaults."""
    _need_header(request)

    try:
        db.reset_user_settings(user["id"])
    except db.AccountsUnavailable as exc:
        raise HTTPException(503, str(exc))

    return _answer(*current_values(user["id"]))
