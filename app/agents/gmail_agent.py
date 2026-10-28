from langchain_core.messages import AIMessage, HumanMessage

from app import db
from app.agents.llm import last_ai_message, llm, run_agent_turn
from app.graph.state import MeetingState
from app.logger import get_logger
from app.tools.gmail_tool import TOOLS, fetch_recent_emails


log = get_logger("gmail_agent")


SYSTEM_PROMPT = """
You are a meeting-finder agent for the user's Gmail inbox. The user's
message lists their latest emails.

Work like this:
1. Decide which ones could be about a meeting, webinar, appointment,
   interview, conference or other scheduled event, judging by sender,
   subject and snippet. A trip also counts: an invitation or request to go
   on a trip, tour, outing or visit, or the plan for one, on a stated date
   (for example "Invitation to join our Kerala trip on 30th November").
   Skip newsletters, promotions and notifications, and skip travel
   advertisements, ticket or hotel offers and booking receipts.
2. For those, call read_email_body to get the details you need. Request
   ALL of the read_email_body calls together in a single turn, one call
   per email, instead of one at a time. If an email already shows its full
   text under "Body:", use that text and do not call read_email_body for it.
3. Call submit_meetings exactly once with every meeting you found
   (an empty list if there are none). That ends your work.

Use each email's own sent date to resolve relative dates such as
"tomorrow". Do not guess: leave date, time or location empty if the
email does not say.

A trip is an in-person event: use mode "offline", a short title such as
"Kerala trip", the date it starts, and the destination as `city` (and
`venue` if the email names a specific place). Leave the time empty if none
is given.

For an online meeting, put its link in `location`. For an in-person
meeting, fill `city`, `venue` and `address` from the email and leave
`location` empty. If the email only names a city, or is vague about the
place ("our office"), set the city and leave `venue` and `address` empty.
For an in-person meeting also fill `map_query`: a short, clean text a map
can find (venue name and city as a map spells them, no brackets, hall or
room numbers, or directions).
"""

# bind_tools sends each tool's name, description and argument schema to
# Gemini along with every request, so it can choose to call them.
agent_llm = llm.bind_tools(TOOLS)


def describe_email(email: dict) -> str:
    text = (
        f"[{email['id']}]\nFrom: {email['sender']}\nSent: {email['date']}\n"
        f"Subject: {email['subject']}\nSnippet: {email['snippet']}"
    )

    if email.get("body"):
        text += f"\nBody:\n{email['body']}"

    return text


def load_emails(state: MeetingState):
    """List the latest emails in code and hand the NEW ones to Gemini in its
    first message. Listing needs no Gemini call, and emails analyzed on an
    earlier check are not sent to Gemini again."""

    listed = state.get("supplied") or fetch_recent_emails()
    known = db.known_emails([e["id"] for e in listed])
    emails = [e for e in listed if e["id"] not in known]

    log.info("load_emails: %d listed, %d already analyzed, %d new", len(listed), len(known), len(emails))

    update = {"emails": emails, "listed_ids": [e["id"] for e in listed]}

    if emails:
        listing = "\n\n".join(describe_email(e) for e in emails)

        update["messages"] = [HumanMessage(f"Check my inbox for meetings. Latest emails:\n\n{listing}")]

    return update


def route_after_load(state: MeetingState):
    """Nothing to read means nothing to ask Gemini about."""

    return "agent" if state["emails"] else "finish"


def agent_step(state: MeetingState):
    return run_agent_turn(agent_llm, SYSTEM_PROMPT, state, log)


def route_after_agent(state: MeetingState):
    """Gemini asked for tools -> run them. Otherwise it answered in plain
    text without submitting, so stop."""

    last = last_ai_message(state["messages"])

    if last and last.tool_calls:
        return "tools"

    log.warning("Agent stopped without calling submit_meetings")
    return "finish"


def route_after_tools(state: MeetingState):
    """Once submit_meetings has run, we're done. Otherwise hand the tool
    results back to Gemini for its next decision."""

    last = last_ai_message(state["messages"])

    if last and any(c["name"] == "submit_meetings" for c in last.tool_calls):
        return "finish"

    return "agent"


def finish(state: MeetingState):
    """Turn what the agent submitted into the dashboard's meeting list,
    adding sender and subject from the emails it listed."""

    emails = {email["id"]: email for email in state["emails"]}

    submitted = []

    for message in state["messages"]:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                if call["name"] == "submit_meetings":
                    submitted = call["args"].get("meetings", [])

    meetings = []

    for item in submitted:
        email = emails.get(item["email_id"])

        if not email:
            log.warning("finish: unknown email_id %r, skipping", item["email_id"])
            continue

        meetings.append({
            "email_id": item["email_id"],
            "sender": email["sender"],
            "subject": email["subject"],
            "title": item["title"] or email["subject"],
            "date": item["date"],
            "time": item["time"],
            "mode": item["mode"],
            "location": item["location"],
            "city": item.get("city", ""),
            "venue": item.get("venue", ""),
            "address": item.get("address", ""),
            "map_query": item.get("map_query", ""),
        })

    log.info(
        "finish: %d email(s) analyzed, %d meeting(s) submitted",
        len(emails), len(meetings),
    )

    return {"meetings": meetings, "emails_checked": len(state["listed_ids"])}
