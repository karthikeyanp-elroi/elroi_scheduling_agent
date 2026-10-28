import time
import uuid

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from app import db
from app.agents.llm import empty_stats, llm
from app.graph.meetpilot_graph import meetpilot
from app.logger import get_logger
from app.streaming import debug_entries, describe_call, error_entry, sse


log = get_logger("chat")


def llm_stats_of(config: dict) -> dict:
    """The chat's running Gemini totals so far (sub-agents included)."""
    return meetpilot.get_state(config).values.get("llm_stats") or empty_stats()


def chat_events(thread_id: str, message: str | None, resume, timezone: str, debug: bool, user_id: str):
    """Run one turn of the chat and stream it. Events:

    token      a piece of MeetPilot's own reply
    activity   a step by MeetPilot or a sub-agent (label only, no email text)
    meetings   the current meeting list
    approval   the chat paused: the user must approve a booking
    question   the chat paused: MeetPilot asked the user something
    debug      Gemini / tool / error entries, only when `debug` is on
    done       the turn ended, with this turn's Gemini usage
    error      something failed
    """
    turn_id = uuid.uuid4().hex[:8]
    started = time.perf_counter()
    log.info("[chat %s] turn started thread=%s resume=%s debug=%s",
             turn_id, thread_id, resume is not None, debug)

    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}, "recursion_limit": 25}

    if resume is not None:
        graph_input = Command(resume=resume)
    else:
        graph_input = {"messages": [HumanMessage(message)], "timezone": timezone}

    # Saved to the database (when there is one): the chat, what was said, the cards.
    session = db.as_uuid(thread_id)
    said_text = None

    if session:
        db.ensure_session(session, timezone, (message or "")[:80] or None, user_id)

        # What the user said: their message, or their answer to a question.
        said = message if message is not None else (resume if isinstance(resume, str) else None)
        said_text = said

        if said:
            db.add_message(session, "user", "text", said, user_id=user_id)

    reply = []

    def remember(kind: str, payload: dict) -> None:
        if session:
            db.add_message(session, "assistant", kind, payload=payload, user_id=user_id)

    # Per agent, tool calls by id, so each result can show its arguments.
    pending = {"MeetPilot": {}, "Gmail agent": {}, "Booking agent": {}}
    paused = False

    # The steps taken, in order: sent to the page as "activity" and saved for the Agent activity page.
    steps = []

    def step(agent: str, label: str) -> str:
        steps.append({"agent": agent, "label": label})
        return sse("activity", {"agent": agent, "label": label})

    try:
        before = llm_stats_of(config)

        for mode, chunk in meetpilot.stream(
            graph_input, config=config, stream_mode=["messages", "updates", "custom"]
        ):
            # MeetPilot's own words. The "tools" node also emits messages
            # (tool results), which are not for the user.
            if mode == "messages":
                token, meta = chunk

                if meta.get("langgraph_node") == "agent" and token.text:
                    reply.append(token.text)
                    yield sse("token", {"text": token.text})

            # What a sub-agent did, forwarded by run_subagent.
            elif mode == "custom":
                # An event ready to send, from one of MeetPilot's own tools (more hotel cards).
                if "event" in chunk:
                    yield sse(chunk["event"], chunk["data"])
                    remember(chunk["event"], chunk["data"])
                    continue

                # A progress note from one of MeetPilot's own tools.
                if "agent" not in chunk:
                    yield step("MeetPilot", chunk["label"])
                    continue

                name, node, values = chunk["agent"], chunk["node"], chunk["values"]

                if node == "agent":
                    for call in values["messages"][-1].tool_calls:
                        yield step(name, describe_call(call))

                elif node == "load_emails":
                    yield step(name, f"load_emails() · {len(values['emails'])} emails, no Gemini")

                elif node == "progress":
                    yield step(name, values["label"])

                elif node == "search_hotels":
                    result = values["result"]
                    yield step(name, f"search_hotels() · {len(values['hotels'])} hotels, no Gemini")
                    hotels = {
                        "status": result["status"],
                        "message": result["message"],
                        "search": result.get("search", {}),
                        "hotels": values["hotels"],
                    }
                    yield sse("hotels", hotels)
                    remember("hotels", hotels)

                elif node == "finish":
                    yield step(name, "finished")

                if debug:
                    for entry in debug_entries(node, values, pending[name]):
                        yield sse("debug", {**entry, "agent": name})

            # MeetPilot's own steps and the pauses.
            else:
                for node, values in chunk.items():
                    if node == "__interrupt__":
                        for interrupt in values:
                            paused = True
                            payload = dict(interrupt.value)
                            kind = payload.pop("type")
                            yield sse(kind, payload)
                            remember(kind, payload)
                        continue


                    if node == "agent":
                        for call in values["messages"][-1].tool_calls:
                            yield step("MeetPilot", describe_call(call))

                    if "meetings" in values:
                        yield sse("meetings", {"meetings": values["meetings"]})
                        remember("meetings", {"meetings": values["meetings"]})

                    if debug:
                        for entry in debug_entries(node, values, pending["MeetPilot"]):
                            yield sse("debug", {**entry, "agent": "MeetPilot"})

        after = llm_stats_of(config)
        used = {key: round(after[key] - before[key], 2) for key in after}

        log.info(
            "[chat %s] turn finished in %.2fs paused=%s | gemini_calls=%d input_tokens=%d "
            "output_tokens=%d total_tokens=%d gemini_latency=%.2fs",
            turn_id, time.perf_counter() - started, paused, used["calls"],
            used["input_tokens"], used["output_tokens"], used["total_tokens"], used["latency_s"],
        )

        if session and reply:
            db.add_message(session, "assistant", "text", "".join(reply), user_id=user_id)

        # A request that made no Gemini call costs nothing, so it is not recorded.
        if used["calls"] > 0:
            db.record_usage(
                session, turn_id, "resume" if resume is not None else "message", llm.model,
                int(used["calls"]), int(used["input_tokens"]), int(used["output_tokens"]),
                int(used["total_tokens"]), used["latency_s"], user_id=user_id,
            )

        # The steps, for the Agent activity page. A turn with no steps (a plain reply) is saved too.
        db.record_run(
            session, turn_id, "resume" if resume is not None else "message",
            said_text or (message if message is not None else "(answered a question)"), steps, user_id=user_id,
        )

        yield sse("done", {**used, "paused": paused})

    except GeneratorExit:
        log.warning("[chat %s] browser disconnected before the turn finished", turn_id)
        raise

    except Exception as exc:
        log.exception("[chat %s] turn failed", turn_id)

        if debug:
            yield sse("debug", error_entry(f"Chat turn failed: {type(exc).__name__}"))

        yield sse("error", {"message": str(exc)})
