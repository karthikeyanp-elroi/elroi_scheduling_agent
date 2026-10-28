import json

from langchain_core.messages import ToolMessage
from langgraph.config import get_stream_writer

from app.agents.llm import last_ai_message, llm, run_agent_turn
from app.graph.state import BookingState
from app.logger import get_logger
from app.tools.calendar_tool import BOOKING_TOOLS, event_location
from app.tools.stay_tool import StayError, search_hotels_for


log = get_logger("booking_agent")


# The calls that end a booking: whichever one runs produces the result.
FINAL_TOOLS = {"create_calendar_event", "hold_meeting"}


SYSTEM_PROMPT = """
You are a booking agent. The user clicked Schedule on one meeting that was
found in their email. Decide what to do with it.

Rules:
- Online meeting with a date and a time: first call check_calendar_conflicts.
  Then call create_calendar_event. If the slot is not free, put the
  overlapping event titles in its `note` so the user is warned.
- Anything else (an in-person or unknown-mode meeting, or a meeting with no
  date or time): call hold_meeting with a short reason. In-person meetings
  are never added to the calendar.
- Finish by calling exactly one of create_calendar_event or hold_meeting.
"""

# bind_tools sends each tool's name, description and argument schema to
# Gemini with every request. State-injected arguments are hidden from it.
agent_llm = llm.bind_tools(BOOKING_TOOLS)


def describe_meeting(meeting: dict) -> str:
    """The request the agent receives."""
    return (
        "Book this meeting:\n"
        f"- title: {meeting['title']}\n"
        f"- mode: {meeting['mode']}\n"
        f"- date: {meeting['date'] or '(not stated)'}\n"
        f"- time: {meeting['time'] or '(not stated)'}\n"
        f"- place: {event_location(meeting) or '(not stated)'}"
    )


def agent_step(state: BookingState):
    return run_agent_turn(agent_llm, SYSTEM_PROMPT, state, log)


def route_after_agent(state: BookingState):
    last = last_ai_message(state["messages"])

    if last and last.tool_calls:
        return "tools"

    log.warning("Booking agent stopped without a final tool call")
    return "finish"


def route_after_tools(state: BookingState):
    """A final tool has run -> done. Otherwise (e.g. the conflict check)
    give the result back to Gemini."""

    last = last_ai_message(state["messages"])

    if last and any(call["name"] in FINAL_TOOLS for call in last.tool_calls):
        return "finish"

    return "agent"


def finish(state: BookingState):
    """The result is whatever the final tool returned."""

    for message in reversed(state["messages"]):
        if isinstance(message, ToolMessage) and message.name in FINAL_TOOLS:
            result = json.loads(message.content)
            log.info("finish: %s via %s", result["status"], message.name)
            return {"result": result}

    return {
        "result": {
            "status": "error",
            "message": "The booking agent did not finish. Please try again.",
            "link": "",
        }
    }


def route_request(state: BookingState):
    """A hotel request goes to the plain-code search. A calendar booking goes
    to the Gemini loop."""

    return "search_hotels" if state.get("request") == "stay" else "agent"


def stay_message(found: dict, hotels: list) -> str:
    wider = ""

    if found.get("widened_to"):
        wider = f"Nothing was listed right at '{found['place']}', so these are around {found['widened_to']}. "

    if hotels:
        return wider + f"{len(hotels)} hotel(s) within your budget."

    if not found["found"]:
        return "No hotels came back for that place or the wider area, for those dates."

    return (
        f"{found['found']} hotels came back, but all are over your budget of "
        f"{found['max_price']:g} {found['currency']} per night "
        f"(the cheapest is {found['cheapest_excluded']:g})."
    )


def search_hotels(state: BookingState):
    """Search hotels near the meeting. This is plain code and makes no Gemini
    call: the venue, dates and your preferences already say what to search."""

    meeting = state["meeting"]
    stay = state["stay"]
    place = stay.get("place") or event_location(meeting)
    writer = get_stream_writer()

    log.info("search_hotels: %r, stay=%s", place, stay)

    try:
        found = search_hotels_for(
            place,
            stay,
            fallback_area=meeting.get("city", ""),
            progress=lambda label: writer({"label": label}),
        )
    except StayError as exc:
        log.warning("search_hotels failed: %s", exc)
        return {"hotels": [], "result": {"status": "error", "message": str(exc), "link": ""}}

    hotels = found.pop("hotels")

    log.info("search_hotels: %d found, %d dropped over budget, %d shown",
             found["found"], found["excluded"], len(hotels))

    return {
        "hotels": hotels,
        "result": {
            "status": "found" if hotels else "none",
            "message": stay_message(found, hotels),
            "link": "",
            "search": found,
        },
    }
