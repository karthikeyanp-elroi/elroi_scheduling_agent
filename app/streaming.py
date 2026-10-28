import json
import traceback


CLIP_CHARS = 3000


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def describe_call(call: dict) -> str:
    """Short label for a tool call. Never includes email content."""
    if call["name"] == "read_email_body":
        return f"read_email_body({call['args'].get('email_id', '')})"

    if call["name"] == "submit_meetings":
        return f"submit_meetings({len(call['args'].get('meetings', []))} meeting(s))"

    if call["name"] in ("book_meeting", "plan_trip"):
        return f"{call['name']}({call['args'].get('meeting_number', '')})"

    return f"{call['name']}()"


def clip(text: str) -> str:
    if len(text) <= CLIP_CHARS:
        return text

    return text[:CLIP_CHARS] + f"\n… ({len(text) - CLIP_CHARS} more chars)"


def debug_entries(node: str, values: dict, pending: dict) -> list[dict]:
    """Debug-log entries for one graph update, in three columns: `gemini`
    (each call's request and response), `tool` (each tool's arguments and
    result) and `error`. Contains raw email content, so it is only sent when
    the browser asks for it."""

    entries = []

    if node == "agent":
        for call in values["messages"][-1].tool_calls:
            pending[call["id"]] = call

        for record in values.get("llm_calls", []):
            entries.append({"column": "gemini", **record})

    elif node == "load_emails":
        entries.append({
            "column": "tool",
            "title": "load_emails (Python, no Gemini)",
            "args": {},
            "result": clip(values["messages"][-1].content) if values.get("messages")
            else f"{len(values.get('listed_ids', []))} email(s) listed, all analyzed before: nothing sent to Gemini.",
        })

    elif node == "search_hotels":
        shown = [
            {"name": h["name"], "price": h["price_per_night"], "currency": h["currency"],
             "distance_km": h["distance_km"], "flags": h["flags"]}
            for h in values["hotels"]
        ]
        entries.append({
            "column": "error" if values["result"]["status"] == "error" else "tool",
            "title": "search_hotels (Python, no Gemini)",
            "args": {},
            "result": clip(json.dumps({"result": values["result"], "hotels": shown}, indent=1)),
        })

    elif node == "tools":
        for message in values["messages"]:
            call = pending.get(message.tool_call_id, {})
            failed = message.status == "error"

            entries.append({
                "column": "error" if failed else "tool",
                "title": message.name,
                "args": call.get("args", {}),
                "result": clip(message.content),
            })

    return entries


def error_entry(title: str, exc: Exception | None = None) -> dict:
    detail = "".join(traceback.format_exception(exc)) if exc else traceback.format_exc()

    return {"column": "error", "title": title, "detail": detail}
