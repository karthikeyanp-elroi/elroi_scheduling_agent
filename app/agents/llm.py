import os
import time

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


def empty_stats() -> dict:
    return {
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "latency_s": 0,
    }


def merge_stats(previous: dict, usage: dict, elapsed: float) -> dict:
    """Add one Gemini call's tokens and latency to the running totals."""
    return {
        "calls": previous["calls"] + 1,
        "input_tokens": previous["input_tokens"] + usage.get("input_tokens", 0),
        "output_tokens": previous["output_tokens"] + usage.get("output_tokens", 0),
        "total_tokens": previous["total_tokens"] + usage.get("total_tokens", 0),
        "latency_s": round(previous["latency_s"] + elapsed, 2),
    }


def combine_stats(a: dict, b: dict) -> dict:
    """Add one run's totals (e.g. a sub-agent's) to another's."""
    return {key: round(a[key] + b[key], 2) for key in a}


def last_ai_message(messages: list):
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message

    return None


def call_record(number: int, messages: list, response, usage: dict, elapsed: float) -> dict:
    """Everything about one Gemini call, for the UI debug log: what it was
    sent this turn (the messages added since its last reply), what it
    answered, and its cost."""

    new = []

    for message in reversed(messages):
        if isinstance(message, AIMessage):
            break
        new.append(message)

    request = "\n\n".join(
        f"[{m.type}{' ' + m.name if getattr(m, 'name', None) else ''}] {m.text}"
        for m in reversed(new)
    )

    return {
        "number": number,
        "latency_s": round(elapsed, 2),
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "messages_in_context": len(messages),
        "request": request[:3000],
        "response": response.text,
        "tool_calls": [
            {"name": c["name"], "args": c["args"]} for c in response.tool_calls
        ],
    }


def run_agent_turn(agent_llm, system_prompt: str, state: dict, log) -> dict:
    """One turn of any tool-calling agent: show Gemini the conversation so
    far and let it either call tools or answer. Returns the node update."""

    started = time.perf_counter()

    response = agent_llm.invoke(
        [SystemMessage(system_prompt)] + state["messages"]
    )

    elapsed = time.perf_counter() - started
    usage = response.usage_metadata or {}

    stats = merge_stats(state.get("llm_stats") or empty_stats(), usage, elapsed)

    log.info(
        "Gemini call #%d: latency=%.2fs input_tokens=%d output_tokens=%d "
        "total_tokens=%d tool_calls=%d",
        stats["calls"], elapsed, usage.get("input_tokens", 0),
        usage.get("output_tokens", 0), usage.get("total_tokens", 0),
        len(response.tool_calls),
    )

    for call in response.tool_calls:
        log.info("Tool requested: %s args=%s", call["name"], call["args"])

    record = call_record(
        stats["calls"], state["messages"], response, usage, elapsed
    )

    return {"messages": [response], "llm_stats": stats, "llm_calls": [record]}
