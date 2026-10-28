import hashlib
import json
from datetime import date, datetime, timedelta
from typing import Annotated
from zoneinfo import ZoneInfo

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from app.logger import get_logger
from app.tools.gmail_tool import RETRIES, per_thread, timed_http


log = get_logger("calendar")

EVENT_DURATION = timedelta(hours=1)

# Reminders added to every event the agent creates.
REMINDERS = [
    {"method": "popup", "minutes": 30},
    {"method": "email", "minutes": 60},
]


def _build_calendar_service(credentials):
    return build("calendar", "v3", http=timed_http(credentials))


get_calendar_service = per_thread(_build_calendar_service)


def event_id_for(meeting: dict) -> str:
    """Stable ID so scheduling the same meeting twice can't duplicate it.
    Calendar IDs allow a-v and 0-9; a SHA-1 hex digest qualifies."""
    key = "|".join([
        meeting["sender"], meeting["subject"], meeting["date"], meeting["time"],
    ])

    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def event_location(meeting: dict) -> str:
    """The place of a meeting as text: the link for an online meeting, or
    "venue, address, city" for an in-person one."""
    if meeting["mode"] == "online":
        return meeting.get("location", "")

    parts = [meeting.get("venue", ""), meeting.get("address", ""), meeting.get("city", "")]

    return ", ".join(part for part in parts if part) or meeting.get("location", "")


def create_event(service, meeting: dict, timezone: str, details: list[str] | None = None) -> dict:
    """Create a calendar event with reminders. `details` are extra lines for
    the description (an in-person meeting's travel and stay). Returns
    {"created": bool, "link": str}; created is False if it already existed."""

    start = datetime.strptime(
        f"{meeting['date']} {meeting['time']}", "%Y-%m-%d %H:%M"
    )
    end = start + EVENT_DURATION

    body = {
        "id": event_id_for(meeting),
        "summary": meeting["title"],
        "description": (
            f"Scheduled by MeetPilot.\n"
            f"From: {meeting['sender']}\n"
            f"Subject: {meeting['subject']}"
            + "".join("\n" + line for line in details or [])
        ),
        "location": event_location(meeting),
        "start": {"dateTime": start.isoformat(), "timeZone": timezone},
        "end": {"dateTime": end.isoformat(), "timeZone": timezone},
        "reminders": {"useDefault": False, "overrides": REMINDERS},
    }

    try:
        event = service.events().insert(
            calendarId="primary",
            body=body,
        ).execute(num_retries=RETRIES)
    except HttpError as exc:
        if exc.resp.status == 409:
            log.info("Event already exists: %s", body["id"])
            existing = service.events().get(
                calendarId="primary", eventId=body["id"]
            ).execute(num_retries=RETRIES)

            if existing.get("status") == "cancelled":
                restored = service.events().update(
                    calendarId="primary",
                    eventId=body["id"],
                    body={**body, "status": "confirmed"},
                ).execute(num_retries=RETRIES)
                log.info("Restored cancelled event %s", body["id"])
                return {"created": True, "link": restored.get("htmlLink", "")}

            return {"created": False, "link": existing.get("htmlLink", "")}
        raise

    log.info("Created event %s (%s)", event["id"], meeting["title"])

    return {"created": True, "link": event.get("htmlLink", "")}


def stay_event_id(reference: str) -> str:
    """Stable ID for a booked stay, so confirming the same booking twice can't add it twice."""
    return hashlib.sha1(("stay|" + reference).encode("utf-8")).hexdigest()


def create_stay_event(service, hotel: dict, search: dict, reference: str, details: list[str] | None = None) -> dict:
    """Put a booked hotel stay on the calendar as its own all-day event, from the
    check-in day through the check-out day. `details` are the lines of the
    description (the meeting, the travel, the booking reference). Returns
    {"created": bool, "link": str}; created is False if it was already there."""

    check_in = date.fromisoformat(search["check_in"])
    check_out = date.fromisoformat(search["check_out"])

    body = {
        "id": stay_event_id(reference),
        "summary": f"Stay: {hotel['name']}",
        "description": "Booked by MeetPilot." + "".join("\n" + line for line in details or []) + (
            "\nHotel page: " + hotel["url"] if hotel.get("url") else ""),
        "location": hotel.get("address") or hotel["name"],
        # all-day events end on the day AFTER their last day: +1 so the check-out day is covered too
        "start": {"date": check_in.isoformat()},
        "end": {"date": (check_out + timedelta(days=1)).isoformat()},
        "transparency": "transparent",      # a stay should not mark the whole day as busy
        "reminders": {"useDefault": False, "overrides": [{"method": "popup", "minutes": 15 * 60}]},   # 9 am the day before
    }

    try:
        event = service.events().insert(calendarId="primary", body=body).execute(num_retries=RETRIES)
    except HttpError as exc:
        if exc.resp.status == 409:
            log.info("Stay event already exists: %s", body["id"])
            existing = service.events().get(calendarId="primary", eventId=body["id"]).execute(num_retries=RETRIES)

            if existing.get("status") == "cancelled":
                restored = service.events().update(
                    calendarId="primary", eventId=body["id"], body={**body, "status": "confirmed"},
                ).execute(num_retries=RETRIES)
                return {"created": True, "link": restored.get("htmlLink", "")}

            return {"created": False, "link": existing.get("htmlLink", "")}
        raise

    log.info("Created stay event %s (%s)", event["id"], hotel["name"])

    return {"created": True, "link": event.get("htmlLink", "")}


def find_scheduled_link(service, meeting: dict) -> str | None:
    """Link to the meeting's calendar event if it's already there, else None."""
    try:
        event = service.events().get(
            calendarId="primary", eventId=event_id_for(meeting)
        ).execute(num_retries=RETRIES)
    except HttpError as exc:
        if exc.resp.status == 404:      # no such event
            return None
        raise

    if event.get("status") == "cancelled":      # deleted events linger as "cancelled"
        return None

    return event.get("htmlLink", "")


def mark_scheduled(meetings: list[dict]) -> list[tuple[dict, Exception]]:
    """Add `scheduled` and `event_link` to each meeting, so a card can show
    "already on your calendar" up front. Only meetings with a known mode, a
    date and a time can be on the calendar. A failed lookup leaves that meeting
    unmarked; failures are returned as (meeting, error) for the caller."""

    failures = []

    for meeting in meetings:
        meeting["scheduled"] = False
        meeting["event_link"] = ""

        if meeting["mode"] not in ("online", "offline") or not (meeting["date"] and meeting["time"]):
            continue

        try:
            link = find_scheduled_link(get_calendar_service(), meeting)
        except Exception as exc:
            log.warning("Calendar lookup failed for %r: %s", meeting["title"], exc)
            failures.append((meeting, exc))
            continue

        if link is not None:
            meeting["scheduled"] = True
            meeting["event_link"] = link

    return failures


def find_conflicts(service, meeting: dict, timezone: str) -> list[str]:
    """Titles of existing events that overlap the meeting's time slot."""

    tz = ZoneInfo(timezone)
    start = datetime.strptime(
        f"{meeting['date']} {meeting['time']}", "%Y-%m-%d %H:%M"
    ).replace(tzinfo=tz)
    end = start + EVENT_DURATION

    events = service.events().list(
        calendarId="primary",
        timeMin=start.isoformat(),
        timeMax=end.isoformat(),
        singleEvents=True,
        orderBy="startTime",
    ).execute(num_retries=RETRIES).get("items", [])

    own_id = event_id_for(meeting)

    return [
        event.get("summary", "(no title)")
        for event in events
        if event["id"] != own_id and event.get("status") != "cancelled"
    ]


# ---------------------------------------------------------------------------
# Tools the booking agent can call. The meeting and time zone come from the
# graph state (InjectedState), so Gemini never has to copy them.
# ---------------------------------------------------------------------------

def _result(status: str, message: str, link: str = "") -> str:
    return json.dumps({"status": status, "message": message, "link": link})


@tool
def check_calendar_conflicts(state: Annotated[dict, InjectedState]) -> str:
    """Check the user's Google Calendar for existing events that overlap this
    meeting's time slot. Returns a JSON list of the overlapping event titles
    (an empty list means the slot is free). Only use for meetings that have
    a date and time."""

    meeting = state["meeting"]

    if not (meeting["date"] and meeting["time"]):
        return json.dumps([])

    return json.dumps(
        find_conflicts(get_calendar_service(), meeting, state["timezone"])
    )


@tool
def create_calendar_event(
    state: Annotated[dict, InjectedState], note: str = ""
) -> str:
    """Create the Google Calendar event, with reminders, for this meeting.
    Only works for online meetings that have a date and time. In-person
    meetings are never added to the calendar. Pass `note` to warn the user
    about something, for example the title of an overlapping event. This
    ends the booking."""

    meeting = state["meeting"]

    # Hard rules in code, whatever the model decides.
    if meeting["mode"] != "online":
        return _result("held", "Only online meetings are added to the calendar.")

    if not (meeting["date"] and meeting["time"]):
        return _result("held", "The email doesn't give a date and time yet.")

    outcome = create_event(get_calendar_service(), meeting, state["timezone"])

    if not outcome["created"]:
        return _result(
            "exists", "This meeting is already on your calendar.", outcome["link"]
        )

    reminders = ", ".join(
        f"{r['minutes']} min before ({r['method']})" for r in REMINDERS
    )
    message = f"Added to your calendar. Reminders: {reminders}."

    if note:
        message += f" Heads up: {note}"

    return _result("scheduled", message, outcome["link"])


@tool
def hold_meeting(reason: str) -> str:
    """Put the meeting on hold without touching the calendar. Use for
    in-person or unknown-mode meetings, and for meetings missing a date or
    time. Give the user a short reason. This ends the booking."""

    return _result("held", reason)


BOOKING_TOOLS = [check_calendar_conflicts, create_calendar_event, hold_meeting]
