import urllib.parse
from typing import Annotated, Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.config import get_config, get_stream_writer
from langgraph.prebuilt import InjectedState
from langgraph.types import Command, interrupt

from app import db
from app.agents.booking_agent import search_hotels
from app.providers import get_provider
from app.inbox import remember_analysis
from app.agents.llm import combine_stats, empty_stats
from app.graph.booking_graph import (
    booking_graph,
    initial_state as booking_initial_state,
    stay_initial_state,
)
from app.graph.graph import graph as gmail_graph, initial_state as gmail_initial_state
from app.logger import get_logger
from app.tools.calendar_tool import (
    REMINDERS,
    create_event,
    create_stay_event,
    event_id_for,
    event_location,
    find_conflicts,
    get_calendar_service,
    mark_scheduled,
    stay_event_id,
)
from app.tools.gmail_tool import OWN_MAIL_PREFIX, send_mail
from app.tools.stay_tool import (
    StayError,
    announce,
    assess_trip,
    current_user,
    default_stay_dates,
    itinerary,
    load_preferences,
    nice_date,
    parse_stay_answer,
    resolve_place,
    stay_setting,
)


log = get_logger("meetpilot_tools")


# ---------------------------------------------------------------------------
# Tools of the main agent. Two of them run a sub-agent (a LangGraph graph of
# its own); one asks the user a question. Booking and questions pause the
# graph with interrupt() until the user answers.
# ---------------------------------------------------------------------------

def run_subagent(name: str, subgraph, start_state: dict, recursion_limit: int) -> dict:
    """Run a sub-agent graph to the end. Each of its updates is forwarded to
    the stream (so the UI can show what it is doing), and the final values
    that matter are returned."""

    writer = get_stream_writer()
    final = {}

    for mode, chunk in subgraph.stream(
        start_state,
        stream_mode=["updates", "custom"],
        config={"recursion_limit": recursion_limit, "configurable": {"user_id": current_user()}},
    ):
        # Progress messages a step sends while it works, such as "Searching...".
        if mode == "custom":
            writer({"agent": name, "node": "progress", "values": chunk})
            continue

        for node, values in chunk.items():
            values = values or {}
            writer({"agent": name, "node": node, "values": values})

            for key in ("meetings", "emails", "listed_ids", "emails_checked", "llm_stats", "result", "hotels"):
                if key in values:
                    final[key] = values[key]

    return final


def current_session() -> str | None:
    """The id of the chat this tool is running for, if it is a valid UUID."""
    try:
        return db.as_uuid(get_config()["configurable"]["thread_id"])
    except (KeyError, RuntimeError):
        return None


def describe(number: int, meeting: dict) -> str:
    """One line per meeting for the model: no email text, just the facts."""
    when = " ".join(part for part in (meeting["date"], meeting["time"]) if part) or "no date/time"
    place = event_location(meeting)
    flag = " [already on the calendar]" if meeting.get("scheduled") else ""

    if meeting["mode"] == "offline" and not (meeting.get("venue") or meeting.get("address")):
        flag += " [venue missing]"

    return (
        f"{number}. {meeting['title']} — {meeting['mode']}, {when}"
        f"{', ' + place if place else ''}{flag}"
    )


def reply(tool_call_id: str, name: str, text: str, **update) -> Command:
    """A tool result (the message the model sees) plus any state changes."""
    return Command(update={
        "messages": [ToolMessage(text, name=name, tool_call_id=tool_call_id)],
        **update,
    })


@tool
def find_meetings(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    mode: Literal["all", "online", "offline"] = "all",
) -> Command:
    """Check the user's 5 latest emails for meetings. Returns a numbered list
    of the meetings found, with date, time, online or offline, and whether
    each is already on the calendar. If the user asks for only one kind, set
    `mode`: "offline" for in-person meetings only, "online" for online
    meetings only. The default "all" shows everything. Use it when the user
    asks to check their inbox, emails or meetings."""

    final = run_subagent("Gmail agent", gmail_graph, gmail_initial_state(), 25)

    analyzed = final.get("emails", [])
    checked = final.get("emails_checked", 0)

    # Save what Gemini analyzed and add the meetings found earlier.
    _, found = remember_analysis(final, current_session())

    # Only the kind asked for is kept, so the numbers on screen (1, 2, ...)
    # are the numbers book_meeting and plan_trip use.
    meetings = [m for m in found if mode == "all" or m["mode"] == mode]
    mark_scheduled(meetings)

    kind = {"all": "", "online": " online", "offline": " in-person"}[mode]

    if meetings:
        head = f"Checked {checked} email(s) ({len(analyzed)} new) and found {len(found)} meeting(s)"

        if mode != "all":
            head += f"; showing the {len(meetings)}{kind} one(s)"

        text = head + ":\n" + "\n".join(describe(i, m) for i, m in enumerate(meetings, 1))
    elif found:
        text = f"Checked {checked} email(s) and found {len(found)} meeting(s), but none are{kind}."
    else:
        text = f"Checked {checked} email(s). None are about a meeting."

    log.info("find_meetings: %s", text.splitlines()[0])

    return reply(
        tool_call_id, "find_meetings", text,
        meetings=meetings,
        llm_stats=combine_stats(
            state.get("llm_stats") or empty_stats(), final.get("llm_stats", empty_stats())
        ),
    )


@tool
def book_meeting(
    meeting_number: int,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Add one ONLINE meeting from the latest find_meetings list to the
    calendar, by its number. The user is asked to approve before anything is
    booked, so do not ask for confirmation yourself. In-person meetings are
    planned with plan_trip, not this tool. Book one meeting at a time."""

    meetings = state.get("meetings", [])

    if not 1 <= meeting_number <= len(meetings):
        return reply(
            tool_call_id, "book_meeting",
            f"There is no meeting number {meeting_number}. "
            f"The list has {len(meetings)} meeting(s).",
        )

    meeting = meetings[meeting_number - 1]

    if meeting["mode"] == "offline":
        return reply(
            tool_call_id, "book_meeting",
            "That is an in-person meeting. It is planned with plan_trip, which also adds it to "
            f"the calendar once the user confirms the itinerary. Call plan_trip({meeting_number}).",
        )

    if meeting["mode"] != "online":
        return reply(
            tool_call_id, "book_meeting",
            "The email doesn't say if that meeting is online or in person, so it can't be booked.",
        )

    if meeting.get("scheduled"):
        return reply(tool_call_id, "book_meeting", "That meeting is already on the calendar.")

    # The graph pauses here until the user approves or declines. Nothing
    # below runs without an explicit approval, whatever the model decides.
    decision = interrupt({
        "type": "approval",
        "meeting_number": meeting_number,
        "meeting": meeting,
    })

    if not (isinstance(decision, dict) and decision.get("approved")):
        log.info("book_meeting: user declined meeting %d", meeting_number)
        db.update_meeting(meeting, status="declined")
        return reply(tool_call_id, "book_meeting", "The user declined. Nothing was booked.")

    final = run_subagent(
        "Booking agent", booking_graph,
        booking_initial_state(meeting, state.get("timezone") or "UTC"), 12,
    )

    result = final.get("result", {"status": "error", "message": "No result.", "link": ""})
    log.info("book_meeting: meeting %d -> %s", meeting_number, result["status"])

    update = {
        "llm_stats": combine_stats(
            state.get("llm_stats") or empty_stats(), final.get("llm_stats", empty_stats())
        ),
    }

    if result["status"] in ("scheduled", "exists"):
        db.update_meeting(
            meeting, status="booked",
            calendar_event_id=event_id_for(meeting), event_link=result["link"],
        )
        booked = {**meeting, "scheduled": True, "event_link": result["link"]}
        update["meetings"] = meetings[:meeting_number - 1] + [booked] + meetings[meeting_number:]

    text = f"{result['status']}: {result['message']}"
    if result["link"]:
        text += f" Event link: {result['link']}"

    return reply(tool_call_id, "book_meeting", text, **update)


def describe_stay(result: dict, hotels: list) -> str:
    """The hotel search as the model sees it: facts only, no links."""
    if result["status"] == "error":
        return f"The hotel search failed: {result['message']}"

    search = result.get("search", {})
    lines = [
        f"{result['message']} Stay {search.get('check_in')} to {search.get('check_out')}, "
        f"{search.get('guests')} guest(s). The hotels are shown to the user as cards."
    ]

    for i, hotel in enumerate(hotels, 1):
        price = f"{hotel['price_per_night']:g} {hotel['currency']}/night" if hotel["price_per_night"] is not None else "price unknown"
        distance = f", {hotel['distance_km']:g} km from the venue" if hotel["distance_km"] is not None else ""
        flags = f" ({'; '.join(hotel['flags'])})" if hotel["flags"] else ""
        lines.append(f"{i}. {hotel['name']}: {price}{distance}{flags}")

    if not hotels and search.get("excluded"):
        lines.append(
            "Every hotel found was over the budget. If the user asks for a higher budget, call more_hotels "
            "with max_price_per_night set to the amount they say."
        )

    return "\n".join(lines)


def finish_trip(
    name: str, tool_call_id: str, state: dict, meeting_number: int,
    meetings: list, trip: dict | None, stay: dict, extra: dict,
) -> Command:
    """The last step for an in-person meeting: show the itinerary and, once
    the user confirms, add the meeting to the calendar with the travel and
    stay in its description. No Gemini call. `extra` is state to keep, such as
    a venue the user gave."""
    meeting = meetings[meeting_number - 1]

    if not (meeting.get("date") and meeting.get("time")):
        return reply(
            tool_call_id, name,
            "The email doesn't give both a date and a time, so the meeting can't be added to the calendar.",
            **extra,
        )

    try:
        home_city = (load_preferences().get("home_location") or {}).get("city", "")
    except StayError:
        home_city = ""

    rows = itinerary(meeting, event_location(meeting), home_city, trip, stay)

    # The graph pauses here. Nothing below runs without an explicit confirmation.
    decision = interrupt({
        "type": "itinerary",
        "meeting_number": meeting_number,
        "meeting": meeting,
        "rows": rows,
    })

    if not (isinstance(decision, dict) and decision.get("approved")):
        return reply(tool_call_id, name, "The user did not confirm the itinerary. Nothing was added to the calendar.", **extra)

    timezone = state.get("timezone") or "UTC"
    announce("Adding it to your calendar\u2026")

    try:
        service = get_calendar_service()
        clashes = find_conflicts(service, meeting, timezone)
        outcome = create_event(service, meeting, timezone, [f"{label}: {text}" for label, text in rows[2:]])
    except Exception as exc:
        log.exception("finish_trip: the calendar event failed")
        return reply(tool_call_id, name, f"The calendar event could not be created: {exc}", **extra)

    db.update_meeting(meeting, status="booked", calendar_event_id=event_id_for(meeting), event_link=outcome["link"])
    booked = {**meeting, "scheduled": True, "event_link": outcome["link"]}

    text = ("Added to the calendar" if outcome["created"] else "It was already on the calendar") + (
        ", with the stay and travel in the description. Reminders: "
        + ", ".join(f"{r['minutes']} min before ({r['method']})" for r in REMINDERS) + "."
    )

    if clashes:
        text += f" Heads up: it overlaps {', '.join(clashes)}."

    if outcome["link"]:
        text += f" Event link: {outcome['link']}"

    return reply(
        tool_call_id, name, text,
        **{**extra, "meetings": meetings[:meeting_number - 1] + [booked] + meetings[meeting_number:]},
    )


@tool
def plan_trip(
    meeting_number: int,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Plan an in-person meeting from the latest find_meetings list, by its
    number. If the venue is missing the user is asked for it, then told how
    far it is and asked whether they want a stay, then asked for the stay
    dates, and can decline at each step, so do not ask for any of these
    yourself. With a stay the hotels are shown as cards to choose from (then
    book_hotel, or more_hotels for others); with their own stay or a same-day
    trip the itinerary follows at once. Nothing reaches the calendar until the
    user confirms. Use it when the user wants to schedule, plan,
    arrange or find a hotel for an in-person meeting."""

    meetings = state.get("meetings", [])

    if not 1 <= meeting_number <= len(meetings):
        return reply(
            tool_call_id, "plan_trip",
            f"There is no meeting number {meeting_number}. "
            f"The list has {len(meetings)} meeting(s).",
        )

    meeting = meetings[meeting_number - 1]

    if meeting["mode"] != "offline":
        return reply(
            tool_call_id, "plan_trip",
            "Hotels can only be searched for an in-person meeting. "
            "Online meetings are added to the calendar with book_meeting.",
        )

    if not meeting.get("date"):
        return reply(
            tool_call_id, "plan_trip",
            "The email doesn't give a date for that meeting, so there are no stay dates to suggest.",
        )

    try:
        prefs = load_preferences()
        check_in, check_out = default_stay_dates(meeting["date"])
    except (StayError, ValueError) as exc:
        return reply(tool_call_id, "plan_trip", f"The hotel search isn't available: {exc}")

    # A venue is needed to search near it. This is asked here, in code, so it
    # never depends on the model remembering to ask.
    venue_given = False

    if not (meeting.get("venue") or meeting.get("address")):
        city = f" in {meeting['city']}" if meeting.get("city") else ""
        title = meeting["title"]

        answer = str(interrupt({
            "type": "question",
            "question": f"'{title}' is in person{city}, but the email doesn't "
                        "say where. What's the venue or address?",
        })).strip()

        if not answer:
            return reply(tool_call_id, "plan_trip", "No venue was given. Nothing was searched.")

        meeting = {**meeting, "venue": answer}
        meetings = meetings[:meeting_number - 1] + [meeting] + meetings[meeting_number:]
        venue_given = True
        db.update_meeting(meeting, venue=answer)

    # Keep a venue the user gave, whatever happens next.
    saved = {"meetings": meetings} if venue_given else {}

    # The clean text Gemini wrote for the map, or the venue and address as the email gave them.
    city = meeting.get("city", "")
    place = meeting.get("map_query") or event_location(meeting)
    matched = resolve_place(place, city)      # cached; the form shows what it found

    # How far is it, and is a stay worth suggesting? Plain code decides; the
    # user chooses. (None when the trip can't be judged: the stay form follows.)
    # The user can correct the place if the map matched the wrong one; the
    # corrected place is then used for the hotel search too.
    while True:
        try:
            trip = assess_trip(prefs, matched, meeting["date"], meeting.get("time", ""), place)
        except (StayError, ValueError) as exc:
            return reply(tool_call_id, "plan_trip", f"The travel check isn't available: {exc}", **saved)

        if not trip:
            break

        answer = interrupt({
            "type": "travel",
            "meeting_number": meeting_number,
            "meeting": meeting,
            "place": place,
            "matched": matched["name"],
            "home": {k: trip["home"][k] for k in ("name", "lat", "lng")},
            "venue": {**{k: trip["venue"][k] for k in ("name", "lat", "lng")}, "approximate": bool(trip["venue"].get("approximate"))},
            "road": trip["road"],
            "trip": {k: trip[k] for k in ("distance_km", "minutes", "leave_at", "day_before", "buffer_minutes", "reasons", "road")},
            "summary": trip["summary"],
            "recommended": trip["recommended"],
        })
        answer = answer if isinstance(answer, dict) else {}

        if str(answer.get("place") or "").strip():
            place = str(answer["place"]).strip()
            matched = resolve_place(place, city)
            continue

        choice = answer.get("choice")

        # No hotel search: the plan goes straight to the itinerary.
        if choice != "stay":
            return finish_trip(
                "plan_trip", tool_call_id, state, meeting_number, meetings, trip,
                {"kind": "own" if choice == "own" else "same_day"}, saved,
            )

        break

    # The graph pauses here for the stay form. Dates, guests and the place to
    # search near come from the user, never from the model.
    answer = interrupt({
        "type": "stay",
        "meeting_number": meeting_number,
        "meeting": meeting,
        "home_city": (prefs.get("home_location") or {}).get("city", ""),
        "place": place,
        "matched": matched["name"] if matched else None,
        "approximate": bool(matched and matched.get("approximate")),
        "check_in": check_in,
        "check_out": check_out,
        "guests": 1,
    })

    if not (isinstance(answer, dict) and answer.get("search")):
        return reply(tool_call_id, "plan_trip", "The user did not want a hotel search.", **saved)

    try:
        check_in, check_out, guests = parse_stay_answer(answer)
    except StayError as exc:
        return reply(tool_call_id, "plan_trip", f"Nothing was searched: {exc}", **saved)

    search_place = str(answer.get("place") or "").strip() or place

    final = run_subagent(
        "Booking agent", booking_graph,
        stay_initial_state(
            meeting, state.get("timezone") or "UTC",
            {"check_in": check_in, "check_out": check_out, "guests": guests, "place": search_place},
        ),
        5,
    )

    result = final.get("result", {"status": "error", "message": "No result.", "link": ""})
    log.info("plan_trip: meeting %d -> %s", meeting_number, result["status"])

    if result["status"] in ("found", "none"):
        db.update_meeting(meeting, status="hotels_searched")

    # Remember what was shown, so the user's "Choose" can be checked against it. A search that found
    # nothing within the budget is remembered too, so the user can ask again with a higher one.
    if result["status"] in ("found", "none"):
        saved["stay_options"] = {
            "meeting_number": meeting_number,
            "email_id": meeting.get("email_id"),
            "title": meeting["title"],
            "trip": trip,
            "limit": int(stay_setting(prefs, "results_per_search")),
            "search": result.get("search", {}),
            "hotels": final.get("hotels", []),
            "chosen": None,
        }

    return reply(
        tool_call_id, "plan_trip", describe_stay(result, final.get("hotels", [])),
        llm_stats=combine_stats(
            state.get("llm_stats") or empty_stats(), final.get("llm_stats", empty_stats())
        ),
        **saved,
    )


def stay_context(state: dict, meeting_number: int):
    """The meeting whose hotels are on screen: (number, meeting, options, None),
    or (None, None, None, why not). `meeting_number` 0 means "that meeting"."""
    meetings = state.get("meetings", [])
    options = state.get("stay_options") or {}

    if not options.get("search"):
        return None, None, None, "No hotel search has been made yet. Use plan_trip first."

    key = (options.get("email_id"), options.get("title"))

    for number, meeting in enumerate(meetings, 1):
        if (meeting.get("email_id"), meeting["title"]) == key:
            if meeting_number and meeting_number != number:
                return None, None, None, "That is not the meeting whose hotels were shown."

            return number, meeting, options, None

    return None, None, None, "The meeting whose hotels were shown is no longer in the list. Use plan_trip again."


def hotel_key(hotel: dict) -> str:
    """What identifies a hotel across searches: its page (without the dates in the link), else its name."""
    return urllib.parse.urlsplit(hotel.get("url") or "").path or (hotel.get("name") or "")


@tool
def more_hotels(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    meeting_number: int = 0,
    max_price_per_night: float = 0,
    minimum_rating: float = 0,
    breakfast_required: bool = False,
) -> Command:
    """Search again for more hotels for the in-person meeting whose hotels were
    just shown (or whose search found none within the budget): more results,
    added as extra cards (hotels already shown are not repeated). It uses
    StayingAPI credits, so use it only when the user asks for more or other
    hotels, or changes what they want. Leave `meeting_number` at 0.

    If the user says what they want this time, pass it, so the search uses it
    instead of their saved preference:
    - max_price_per_night: a new budget per night, in the user's currency
      ("show up to 5k" -> 5000). Use it when nothing fit the old budget.
    - minimum_rating: the lowest rating they accept, 0 to 5 ("4.5 and above").
    - breakfast_required: true if they want breakfast included.
    Leave them at 0 / false to keep the saved preference."""

    number, meeting, options, problem = stay_context(state, meeting_number)

    if problem:
        return reply(tool_call_id, "more_hotels", problem)

    # What the user asked for now, kept for the next "more hotels" too.
    overrides = dict(options.get("overrides") or {})

    if max_price_per_night:
        if not 0 < max_price_per_night <= 1_000_000:
            return reply(tool_call_id, "more_hotels", "The budget must be a positive amount per night.")
        overrides["max_price_per_night"] = float(max_price_per_night)

    if minimum_rating:
        if not 0 < minimum_rating <= 5:
            return reply(tool_call_id, "more_hotels", "The rating must be between 0 and 5.")
        overrides["minimum_rating"] = float(minimum_rating)

    if breakfast_required:
        overrides["breakfast_required"] = True

    try:
        prefs = load_preferences()
        limit = int(options.get("limit") or stay_setting(prefs, "results_per_search")) + int(stay_setting(prefs, "more_results"))
    except (StayError, ValueError) as exc:
        return reply(tool_call_id, "more_hotels", f"The hotel search isn't available: {exc}")

    search = options["search"]
    stay = {"check_in": search["check_in"], "check_out": search["check_out"], "guests": search["guests"],
            "place": search["place"], "limit": limit, "overrides": overrides}

    # The Booking agent's hotel step: plain code, no Gemini call.
    found = search_hotels({"meeting": meeting, "stay": stay})
    result = found["result"]

    if result["status"] == "error":
        return reply(tool_call_id, "more_hotels", f"The hotel search failed: {result['message']}")

    shown = {hotel_key(h) for h in options["hotels"]}
    new = [h for h in found["hotels"] if hotel_key(h) not in shown]
    combined = options["hotels"] + new
    searched = result["search"]

    if new:
        # "append" adds cards to the list on screen; with nothing on screen yet, the page builds the list.
        get_stream_writer()({"event": "hotels", "data": {
            "status": "found", "message": f"{len(new)} more hotel(s).", "search": searched,
            "hotels": new, "append": bool(options["hotels"]), "offset": len(options["hotels"]),
        }})

    update = {"stay_options": {**options, "hotels": combined, "limit": limit, "overrides": overrides, "search": {**options["search"], **searched}}}
    budget = f"{searched['max_price']:g} {searched['currency']}" if searched.get("max_price") else "the user's budget"

    if not new:
        over = (
            f" {searched['excluded']} hotel(s) were over {budget} per night (the cheapest is {searched['cheapest_excluded']:g}). "
            "If the user wants to see them, they can ask for a higher budget."
        ) if searched.get("excluded") else ""

        return reply(
            tool_call_id, "more_hotels",
            f"The search went up to {limit} results and found no new hotel within {budget} per night." + over +
            " They can also search again for a longer list.", **update,
        )

    lines = [f"Found {len(new)} more hotel(s) within {budget} per night; they are shown as extra cards. {len(combined)} in total."]

    for i, hotel in enumerate(new, len(options["hotels"]) + 1):
        price = f"{hotel['price_per_night']:g} {hotel['currency']}/night" if hotel["price_per_night"] is not None else "price unknown"
        lines.append(f"{i}. {hotel['name']}: {price}")

    return reply(tool_call_id, "more_hotels", "\n".join(lines), **update)


def booking_email(meeting: dict, hotel: dict, search: dict, reference: str, total, currency, rows: list) -> tuple[str, str]:
    """The confirmation mail (subject, body). It says plainly that it is a demo."""
    lines = [
        "This is a DEMO booking from MeetPilot. Nothing was charged and no room was reserved.",
        "",
        f"Reference: {reference}",
        f"Hotel: {hotel['name']}",
        f"Dates: {nice_date(search.get('check_in'))} to {nice_date(search.get('check_out'))} ({search.get('nights')} night(s), {search.get('guests')} guest(s))",
        f"Total: {total:g} {currency}" if total is not None else "Total: not available",
        "",
    ]
    lines += [f"{label}: {text}" for label, text in rows]
    lines += ["", "To book this hotel for real, open it on Booking.com: " + (hotel.get("url") or "(no link)")]

    return f"{OWN_MAIL_PREFIX}: {hotel['name']} ({reference})", "\n".join(lines)


@tool
def book_hotel(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    meeting_number: int = 0,
) -> Command:
    """Book the hotel the user chose for an in-person meeting (a demo for now:
    no real charge or reservation). The user sees the booking and confirms or
    cancels, so do not ask for confirmation yourself. After they confirm, the
    payment, a confirmation email and the calendar event follow on their own.
    Use it when the user asks to review, book or confirm the hotel. Leave
    `meeting_number` at 0."""

    number, meeting, options, problem = stay_context(state, meeting_number)

    if problem:
        return reply(tool_call_id, "book_hotel", problem)

    hotel = options.get("chosen")

    if not hotel:
        return reply(
            tool_call_id, "book_hotel",
            "No hotel has been chosen yet. The user chooses one with the Choose button on a hotel card.",
        )

    search = options.get("search", {})

    try:
        provider = get_provider()
    except StayError as exc:
        return reply(tool_call_id, "book_hotel", f"Booking isn't available: {exc}")

    reference = provider.reference(meeting, hotel, search)
    existing = db.get_booking(reference)

    if existing and existing.get("status") == "confirmed":
        return reply(tool_call_id, "book_hotel", f"This hotel is already booked for these dates. Reference {reference}. Nothing more to do.")

    quote = provider.quote(hotel, search)

    if not quote.get("available"):
        return reply(tool_call_id, "book_hotel", f"The hotel can't be booked for those dates: {quote.get('note') or 'no price'}")

    try:
        home_city = (load_preferences().get("home_location") or {}).get("city", "")
    except StayError:
        home_city = ""

    stay = {"kind": "hotel", "hotel": hotel, "search": search}
    rows = itinerary(meeting, event_location(meeting), home_city, options.get("trip"), stay)

    # The graph pauses here. Nothing below runs without an explicit confirmation.
    decision = interrupt({
        "type": "booking",
        "meeting_number": number,
        "meeting": meeting,
        "rows": rows,
        "hotel": hotel["name"],
        "total": quote["total"],
        "currency": quote["currency"],
        "nights": search.get("nights"),
        "guests": search.get("guests"),
        "reference": reference,
        "demo": provider.demo,
    })

    if not (isinstance(decision, dict) and decision.get("approved")):
        return reply(tool_call_id, "book_hotel", "The user cancelled. Nothing was paid or booked. The hotel cards are still on screen.")

    try:
        payment = provider.pay(quote, reference, announce)
        provider.book(hotel, search, reference, payment, announce)
    except Exception as exc:
        log.exception("book_hotel: the payment or booking failed")
        return reply(tool_call_id, "book_hotel", f"The payment or booking failed: {exc}. Nothing was booked or added to the calendar.")

    db.save_booking(reference, meeting, hotel, search, provider.name, payment.get("payment_id"), quote["total"], quote["currency"])

    # The confirmation mail, to the user's own address.
    announce("Sending your confirmation email\u2026")
    emailed = False

    try:
        subject, body = booking_email(meeting, hotel, search, reference, quote["total"], quote["currency"], rows)
        sent_id = send_mail(subject, body)
        db.mark_processed([(sent_id, subject, False)], None)     # so the inbox watcher doesn't analyze it
        emailed = True
    except Exception as exc:
        log.exception("book_hotel: the confirmation mail could not be sent")
        mail_problem = str(exc)

    # Then the calendar. What was booked is the STAY, so that is what goes on it: its own event from check-in
    # to check-out, with the meeting and the travel in the description. (The meeting itself is added to the
    # calendar only for online meetings, and for a trip with no hotel to book.)
    announce("Saving it to your calendar\u2026")
    details = [f"{label}: {text}" for label, text in rows] + [f"Booking: {reference}" + (" (demo: no real reservation)" if provider.demo else "")]
    link, saved = "", False

    try:
        service = get_calendar_service()
        outcome = create_stay_event(service, hotel, search, reference, details)
        link, saved = outcome["link"], True
        db.update_meeting(meeting, status="booked", calendar_event_id=stay_event_id(reference), event_link=link)
    except Exception as exc:
        log.exception("book_hotel: the calendar event failed")
        calendar_problem = str(exc)

    booked = {**meeting, "booking_ref": reference, "booking_email": emailed, "scheduled": saved, "event_link": link}
    meetings = state.get("meetings", [])

    text = (f"Booking confirmed{' (DEMO: no real charge or reservation)' if provider.demo else ''}. Reference {reference}. "
            f"Total {quote['total']:g} {quote['currency']}. "
            + ("A confirmation email was sent to the user. " if emailed else f"The confirmation email could not be sent ({mail_problem}). ")
            + (f"The stay ({nice_date(search.get('check_in'))} to {nice_date(search.get('check_out'))}) was added to the calendar. Event link: {link}"
               if saved else f"It could not be added to the calendar ({calendar_problem})."))

    return reply(tool_call_id, "book_hotel", text, meetings=meetings[:number - 1] + [booked] + meetings[number:])


@tool
def ask_user(question: str) -> str:
    """Ask the user a question and wait for the answer. Use it only when you
    need information that only the user has, such as which meeting they
    mean. Do not use it for the venue of an in-person meeting: book_meeting
    asks for that itself."""

    return str(interrupt({"type": "question", "question": question}))


MAIN_TOOLS = [find_meetings, book_meeting, plan_trip, more_hotels, book_hotel, ask_user]
