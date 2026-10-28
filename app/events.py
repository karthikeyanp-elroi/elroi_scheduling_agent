import email.utils
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import httplib2
from googleapiclient.errors import HttpError

from app import db
from app.agents.llm import llm
from app.graph.graph import graph as gmail_graph, initial_state as gmail_initial_state
from app.inbox import remember_analysis
from app.logger import get_logger
from app.tools.gmail_tool import MAX_BODY_CHARS, OWN_MAIL_PREFIX, fetch_emails_by_ids


log = get_logger("events")

MAX_PARALLEL = 2    
RETRY_DELAYS = (3, 10)   


class Broadcaster:
    def __init__(self):
        self._listeners = {}        
        self._lock = threading.Lock()

    def subscribe(self, deliver, user_id: str):
        """`deliver(event, data)` is called for every message of this user, from any thread."""
        with self._lock:
            self._listeners[deliver] = user_id

        return deliver

    def unsubscribe(self, deliver) -> None:
        with self._lock:
            self._listeners.pop(deliver, None)

    def publish(self, event: str, data: dict, user_id: str) -> None:
        """Tell the pages of this user (and nobody else's)."""
        with self._lock:
            listeners = [deliver for deliver, owner in self._listeners.items() if owner == user_id]

        for deliver in listeners:
            try:
                deliver(event, data)
            except Exception:
                log.exception("A listener of event %r failed", event)


broadcaster = Broadcaster()


def emails_from_event(event: dict) -> list[dict]:
    """The emails an event is about, in the shape the Gmail agent takes: an
    id of a real email (fetched from Gmail), or the email itself."""
    if event.get("message_id"):
        return fetch_emails_by_ids([event["message_id"]], event["user_id"])

    mail = event["email"]
    body = mail.get("body") or ""

    return [{
        "id": mail.get("id") or "fake-" + uuid.uuid4().hex[:10],
        "sender": mail.get("sender") or "unknown@example.com",
        "subject": mail.get("subject") or "(no subject)",
        "date": mail.get("date") or email.utils.format_datetime(datetime.now(timezone.utc)),
        "snippet": body[:150],
        "body": body[:MAX_BODY_CHARS],
    }]


MAX_REMEMBERED = 50
_found = OrderedDict()  


def meetings_of(event_id: str, user_id: str) -> list[dict] | None:
    owner, meetings = _found.get(event_id, (None, None))

    return meetings if owner == user_id else None


def is_network_error(exc: Exception) -> bool:
    """A failure that is likely to pass by itself: a dropped connection, a
    timeout, or Google being briefly busy."""
    if isinstance(exc, HttpError):
        return exc.resp.status in (429, 500, 502, 503, 504)

    return isinstance(exc, (OSError, httplib2.HttpLib2Error))


def read_emails_of(event: dict) -> list[dict]:
    """`emails_from_event`, tried again a few times if the network lets it down."""
    for delay in RETRY_DELAYS:
        try:
            return emails_from_event(event)
        except Exception as exc:
            if not is_network_error(exc):
                raise

            log.warning("Could not read the mail (%s). Trying again in %d s.", exc, delay)
            time.sleep(delay)

    return emails_from_event(event)


_in_flight = set()
_in_flight_lock = threading.Lock()


def handle_event(event_id: str, event: dict) -> None:
    """Analyze the email(s) of one event, save the result, and tell the page."""
    started = time.perf_counter()
    claimed = []
    user_id = event["user_id"]

    try:
        emails = read_emails_of(event)


        if emails and all(m["subject"].startswith(OWN_MAIL_PREFIX) for m in emails):
            broadcaster.publish("email_skipped", {"event_id": event_id, "subject": emails[0]["subject"], "reason": "sent by MeetPilot"}, user_id)
            return

        with _in_flight_lock:
            claimed = [e for e in emails if (user_id, e["id"]) not in _in_flight]
            _in_flight.update((user_id, e["id"]) for e in claimed)

        for e in emails:
            broadcaster.publish("email_received", {"event_id": event_id, "id": e["id"], "subject": e["subject"], "sender": e["sender"]}, user_id)

        if not claimed:
            broadcaster.publish("email_skipped", {"event_id": event_id, "subject": emails[0]["subject"], "reason": "already being analyzed"}, user_id)
            return

        final = gmail_graph.invoke(gmail_initial_state(claimed), config={"recursion_limit": 25, "configurable": {"user_id": user_id}})
        new, _ = remember_analysis(final, None, user_id)

        stats = final["llm_stats"]

        if stats["calls"] > 0:
            db.record_usage(
                None, f"event-{event_id}", "event", llm.model, int(stats["calls"]),
                int(stats["input_tokens"]), int(stats["output_tokens"]),
                int(stats["total_tokens"]), stats["latency_s"], user_id=user_id,
            )

        subject = claimed[0]["subject"]

        if not final.get("emails"):
            outcome = "email_skipped"
            broadcaster.publish(outcome, {"event_id": event_id, "subject": subject, "reason": "already analyzed"}, user_id)
        elif new:
            outcome = "new_meetings"
            _found[event_id] = (user_id, new)

            while len(_found) > MAX_REMEMBERED:
                _found.popitem(last=False)

            broadcaster.publish(outcome, {
                "event_id": event_id,
                "email": {"subject": subject, "sender": claimed[0]["sender"]},
                "meetings": new,
            }, user_id)
        else:
            outcome = "no_meeting"
            broadcaster.publish(outcome, {"event_id": event_id, "subject": subject}, user_id)

        db.record_run(
            None, f"event-{event_id}", "event", "New email: " + subject,
            [{"agent": "Gmail agent", "label": f"{stats['calls']} Gemini call(s) · {outcome.replace('_', ' ')}"}], user_id=user_id,
        )

        log.info(
            "[event %s] %s: %r in %.2fs | gemini_calls=%d total_tokens=%d",
            event_id, outcome, subject, time.perf_counter() - started, stats["calls"], stats["total_tokens"],
        )

    except Exception as exc:
        log.exception("[event %s] failed", event_id)
        broadcaster.publish("event_error", {"event_id": event_id, "message": str(exc)}, user_id)

    finally:
        with _in_flight_lock:
            _in_flight.difference_update((user_id, e["id"]) for e in claimed)


_executor = ThreadPoolExecutor(max_workers=MAX_PARALLEL, thread_name_prefix="email-event")


def submit(event: dict) -> str:
    """Queue an email event for event["user_id"]. Returns its id straight away; the work
    happens in the background and the result arrives on that user's event stream."""
    if not event.get("user_id"):
        raise ValueError("An email event needs a user_id.")

    event_id = uuid.uuid4().hex[:8]

    log.info("[event %s] received: %s", event_id,
             event["message_id"] if event.get("message_id") else repr(event["email"].get("subject")))

    _executor.submit(handle_event, event_id, event)

    return event_id
