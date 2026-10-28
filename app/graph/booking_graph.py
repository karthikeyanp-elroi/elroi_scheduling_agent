from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from app.agents.llm import empty_stats
from app.graph.state import BookingState
from app.agents.booking_agent import (
    agent_step,
    describe_meeting,
    route_after_agent,
    route_after_tools,
    route_request,
    search_hotels,
    finish,
)
from app.tools.calendar_tool import BOOKING_TOOLS


builder = StateGraph(BookingState)


builder.add_node(
    "agent",
    agent_step
)

# handle_tool_errors=False lets a Google API error surface to the endpoint
# (which reports it) instead of being fed back to Gemini to retry.
builder.add_node(
    "tools",
    ToolNode(BOOKING_TOOLS, handle_tool_errors=False)
)

builder.add_node(
    "finish",
    finish
)

# Plain code, no Gemini: searches hotels for a meeting.
builder.add_node(
    "search_hotels",
    search_hotels
)


builder.add_conditional_edges(
    START,
    route_request,
    ["agent", "search_hotels"]
)

builder.add_conditional_edges(
    "agent",
    route_after_agent,
    ["tools", "finish"]
)

builder.add_conditional_edges(
    "tools",
    route_after_tools,
    ["agent", "finish"]
)

builder.add_edge(
    "finish",
    END
)

builder.add_edge(
    "search_hotels",
    END
)


booking_graph = builder.compile()


def initial_state(meeting: dict, timezone: str) -> dict:
    """The state a booking run for one meeting starts from."""
    return {
        "messages": [HumanMessage(describe_meeting(meeting))],
        "request": "calendar",
        "meeting": meeting,
        "timezone": timezone,
        "stay": {},
        "hotels": [],
        "result": {},
        "llm_stats": empty_stats(),
        "llm_calls": [],
    }


def stay_initial_state(meeting: dict, timezone: str, stay: dict) -> dict:
    """The state a hotel search for one meeting starts from."""
    return {**initial_state(meeting, timezone), "request": "stay", "stay": stay, "messages": []}
