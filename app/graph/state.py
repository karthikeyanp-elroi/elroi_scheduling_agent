import operator
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class MeetingState(TypedDict):
    # The agent loop's conversation: human request, Gemini replies
    # (including tool calls) and tool results. add_messages appends.
    messages: Annotated[list, add_messages]

    # Emails handed to the agent (by an email event) instead of listing the inbox.
    supplied: list[dict]

    # The latest emails, loaded by our own code before Gemini is called.
    emails: list[dict]      # the NEW ones, for Gemini: id, sender, subject, date, snippet
    listed_ids: list[str]   # every email listed, new or already analyzed

    # Final output of the run.
    meetings: list[dict]    # sender, subject, title, date, time, mode, location
    emails_checked: int

    # Gemini usage for the run: calls, input/output/total tokens, latency_s.
    llm_stats: dict
    llm_calls: Annotated[list, operator.add]   # per-call records for the debug log


class BookingState(TypedDict):
    messages: Annotated[list, add_messages]     # the booking agent's loop

    request: str        # "calendar" to book the meeting, "stay" to search hotels for it
    meeting: dict       # one entry from MeetingState["meetings"]
    timezone: str       # IANA name from the browser, e.g. "Asia/Kolkata"
    stay: dict          # for "stay": check_in, check_out, guests
    hotels: list        # for "stay": the ranked hotels

    result: dict        # status ("scheduled", "exists", "held", "error"), message, link
    llm_stats: dict     # Gemini usage for this booking
    llm_calls: Annotated[list, operator.add]    # per-call records for the debug log


class MainState(TypedDict):
    messages: Annotated[list, add_messages]     # the whole chat with the user

    meetings: list[dict]    # set by find_meetings; book_meeting reads from here
    stay_options: dict      # the latest hotel search and the hotel the user chose from it
    timezone: str           # IANA name from the browser, e.g. "Asia/Kolkata"

    llm_stats: dict         # Gemini usage for the chat, sub-agents included
    llm_calls: Annotated[list, operator.add]    # per-call records for the debug log
