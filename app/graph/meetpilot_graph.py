from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from app.agents.meetpilot import agent_step, route_after_agent
from app.graph.state import MainState
from app.tools.meetpilot_tools import MAIN_TOOLS


builder = StateGraph(MainState)


builder.add_node(
    "agent",
    agent_step
)

builder.add_node(
    "tools",
    ToolNode(MAIN_TOOLS)
)


builder.add_edge(
    START,
    "agent"
)

builder.add_conditional_edges(
    "agent",
    route_after_agent,
    ["tools", END]
)

builder.add_edge(
    "tools",
    "agent"
)


# The checkpointer is the chat's memory: one thread_id per conversation.
# In memory only, so a server restart starts the chat fresh.
meetpilot = builder.compile(checkpointer=MemorySaver())
