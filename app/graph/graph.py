from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from app.agents.llm import empty_stats
from app.graph.state import MeetingState
from app.agents.gmail_agent import (
    load_emails,
    route_after_load,
    agent_step,
    route_after_agent,
    route_after_tools,
    finish,
)
from app.tools.gmail_tool import TOOLS


builder = StateGraph(MeetingState)


# Lists the latest emails in plain code, with no Gemini call.
builder.add_node(
    "load_emails",
    load_emails
)

builder.add_node(
    "agent",
    agent_step
)

# Runs whichever tools Gemini asked for and appends the results as messages.
builder.add_node(
    "tools",
    ToolNode(TOOLS)
)

builder.add_node(
    "finish",
    finish
)


builder.add_edge(
    START,
    "load_emails"
)

builder.add_conditional_edges(
    "load_emails",
    route_after_load,
    ["agent", "finish"]
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


graph = builder.compile()


def initial_state(supplied: list[dict] | None = None) -> dict:
    """The state a Gmail agent run starts from. `supplied` hands it given
    emails (an email event); without it, it lists the newest in Gmail."""
    return {
        "supplied": supplied or [],
        "messages": [],     # load_emails writes the first message
        "emails": [],
        "listed_ids": [],
        "meetings": [],
        "emails_checked": 0,
        "llm_stats": empty_stats(),
        "llm_calls": [],
    }
