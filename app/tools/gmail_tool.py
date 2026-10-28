import google_auth_httplib2
import httplib2
from google.auth.exceptions import RefreshError
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import base64
import json
from email.message import EmailMessage
import os
import socket
import threading
from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app import crypto, db
from app.logger import get_logger
from app.tools.stay_tool import current_user


log = get_logger("gmail")


def prefer_ipv4() -> None:
    """Try IPv4 addresses before IPv6 for every outgoing connection.

    Some networks hand out IPv6 addresses for Google's servers that can't be
    reached. Browsers and curl try IPv4 and IPv6 side by side and use the one
    that works, but Python tries the addresses in the order DNS gives them, so
    each attempt waited about 21 seconds for a timeout and then failed.
    IPv6 stays as the fallback, for networks that only have IPv6."""
    if getattr(socket.getaddrinfo, "ipv4_first", False):
        return

    original = socket.getaddrinfo

    def ipv4_first(*args, **kwargs):
        return sorted(original(*args, **kwargs), key=lambda found: found[0] != socket.AF_INET)

    ipv4_first.ipv4_first = True
    socket.getaddrinfo = ipv4_first


prefer_ipv4()


SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",       # the booking confirmation mail, to the user's own address
    "https://www.googleapis.com/auth/calendar.events",
]


class GmailNotConnected(Exception):
    """This user has no working Gmail connection. The message tells them what to do."""


OAUTH_URI = "https://oauth2.googleapis.com/token"


def oauth_client() -> dict:
    """The Google "Web application" client (GOOGLE_OAUTH_CLIENT_FILE in .env): id, secret and addresses."""
    path = os.getenv("GOOGLE_OAUTH_CLIENT_FILE", "")

    if not path or not os.path.isfile(path):
        raise GmailNotConnected("Gmail connection isn't set up on this server (GOOGLE_OAUTH_CLIENT_FILE in .env).")

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data["web"]
    except (OSError, ValueError, KeyError):
        raise GmailNotConnected("GOOGLE_OAUTH_CLIENT_FILE is not a Google 'Web application' client file.")


_creds = {}                     # user id -> that user's credentials (shared by their Gmail and Calendar clients)
_creds_lock = threading.Lock()


def forget_credentials(user_id: str) -> None:
    """Drop what is cached for a user, after they connect, disconnect or are found revoked."""
    with _creds_lock:
        _creds.pop(user_id, None)


class UserCredentials(Credentials):
    """Google credentials for one user. If Google refuses the refresh token (the user removed
    MeetPilot's access, or it expired) the connection is marked needs_reconnect, so the page can say so."""

    def __init__(self, *args, user_id=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.user_id = user_id

    def refresh(self, request):
        try:
            super().refresh(request)
        except RefreshError as exc:
            log.warning("Google refused the Gmail token of user %s: %s", self.user_id, exc)

            if self.user_id:
                forget_credentials(self.user_id)

                try:
                    db.set_gmail_status(self.user_id, "needs_reconnect", str(exc)[:300])
                except Exception:
                    log.exception("Could not record that the Gmail token was refused")

            raise GmailNotConnected("Google no longer allows MeetPilot to use your Gmail. Open Settings and connect it again.")


def get_credentials(user_id: str | None = None) -> UserCredentials:
    """The signed-in user's Google credentials, from their own saved (encrypted) refresh token."""
    user_id = user_id or current_user()

    if not user_id:
        raise GmailNotConnected("Sign in to use Gmail.")

    with _creds_lock:
        found = _creds.get(user_id)

    if found:
        return found

    row = db.get_gmail_secret(user_id)

    if not row:
        raise GmailNotConnected("Gmail isn't connected. Open Settings and choose Connect Gmail.")

    if row["status"] != "connected":
        raise GmailNotConnected("Your Gmail connection needs to be set up again. Open Settings and choose Connect Gmail.")

    client = oauth_client()
    creds = UserCredentials(
        token=None,
        refresh_token=crypto.decrypt(row["refresh_token_enc"]),
        token_uri=OAUTH_URI,
        client_id=client["client_id"],
        client_secret=client["client_secret"],
        scopes=SCOPES,
        user_id=user_id,
    )

    with _creds_lock:
        _creds[user_id] = creds

    return creds


def per_thread(factory):
    """Build one client per thread and user, and reuse it there. Google's HTTP clients aren't
    thread-safe, and we run tools on several threads (sharing one client caused SSL errors).
    The credentials themselves are shared per user. `factory(credentials)` makes the client."""
    local = threading.local()

    def get(user_id: str | None = None):
        creds = get_credentials(user_id)
        clients = getattr(local, "clients", None)

        if clients is None:
            clients = local.clients = {}

        found = clients.get(creds.user_id)

        # a new credentials object means the user connected again: the old client is stale
        if found is None or found[0] is not creds:
            found = (creds, factory(creds))
            clients[creds.user_id] = found
            log.info("%s ready (thread %s)", factory.__name__, threading.current_thread().name)

        return found[1]

    return get


HTTP_TIMEOUT = 20   # seconds: a stuck connection fails (and is retried) instead of hanging for ages
RETRIES = 3         # extra tries, with a growing pause, after a reset connection, a timeout or a 5xx/429


def timed_http(credentials):
    """An HTTP client for a Google API (Gmail, Calendar) that gives up on a stuck connection."""
    return google_auth_httplib2.AuthorizedHttp(credentials, http=httplib2.Http(timeout=HTTP_TIMEOUT))


def _build_gmail_service(credentials):
    return build("gmail", "v1", http=timed_http(credentials))


get_gmail_service = per_thread(_build_gmail_service)


def read_emails(service):
    results = service.users().messages().list(
        userId="me",
        maxResults=5
    ).execute(num_retries=RETRIES)

    messages = results.get("messages", [])
    log.info("Listed %d message(s): %s", len(messages), [m["id"] for m in messages])

    return messages


def get_email(service, message_id):
    message = service.users().messages().get(
        userId="me",
        id=message_id,
        format="full"
    ).execute(num_retries=RETRIES)
    log.info("Fetched message %s", message_id)

    return message


def parse_email(message):
    headers = message["payload"].get("headers", [])

    sender = ""
    subject = ""
    date = ""

    for header in headers:
        name = header["name"].lower()

        if name == "from":
            sender = header["value"]

        elif name == "subject":
            subject = header["value"]

        elif name == "date":
            date = header["value"]

    body = extract_body(message["payload"])
    log.info("Parsed email from=%r subject=%r body_chars=%d", sender, subject, len(body))
    if not body:
        log.warning("Empty body (no text/plain part found)")

    return {
        "sender": sender,
        "subject": subject,
        "date": date,
        "body": body,
    }


def extract_body(payload):
    if "parts" in payload:

        for part in payload["parts"]:
            mime_type = part.get("mimeType")

            if mime_type == "text/plain":
                data = part["body"].get("data")

                if data:
                    return base64.urlsafe_b64decode(data).decode("utf-8")

    data = payload.get("body", {}).get("data")

    if data:
        return base64.urlsafe_b64decode(data).decode("utf-8")

    return ""


# ---------------------------------------------------------------------------
# Tools the Gmail agent can call. Gemini sees each tool's name, docstring and
# argument types, and decides which to call. Our code only runs them.
# ---------------------------------------------------------------------------

MAX_BODY_CHARS = 3000


def email_summary(service, message_id: str, with_body: bool = False) -> dict:
    full = get_email(service, message_id)
    parsed = parse_email(full)

    summary = {
        "id": message_id,
        "sender": parsed["sender"],
        "subject": parsed["subject"],
        "date": parsed["date"],
        "snippet": full.get("snippet", ""),
    }

    if with_body:
        summary["body"] = parsed["body"][:MAX_BODY_CHARS]

    return summary


def fetch_recent_emails(user_id: str | None = None) -> list[dict]:
    """The 5 most recent emails: id, sender, subject, date (when it was
    sent) and a short snippet. Called by our own code, not by Gemini, since
    listing the inbox is always the first step."""

    try:
        service = get_gmail_service(user_id)

        return [email_summary(service, message["id"]) for message in read_emails(service)]

    except Exception:
        log.exception("fetch_recent_emails failed")
        raise


def fetch_emails_by_ids(message_ids: list[str], user_id: str | None = None) -> list[dict]:
    """The given emails with their text, for an email event that names them."""

    try:
        service = get_gmail_service(user_id)

        return [email_summary(service, message_id, with_body=True) for message_id in message_ids]

    except Exception:
        log.exception("fetch_emails_by_ids failed")
        raise


OWN_MAIL_PREFIX = "Booking confirmed"   # every mail MeetPilot sends starts like this


def send_mail(subject: str, body: str, user_id: str | None = None) -> str:
    """Send a mail from the user's own address to the same address. Returns the
    Gmail id of the sent mail. Not retried: a retry after a lost answer could
    send it twice."""
    service = get_gmail_service(user_id)
    me = service.users().getProfile(userId="me").execute(num_retries=RETRIES)["emailAddress"]

    message = EmailMessage()
    message["To"] = me
    message["From"] = me
    message["Subject"] = subject
    message.set_content(body)

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")
    sent = service.users().messages().send(userId="me", body={"raw": raw}).execute()
    log.info("Sent mail %s: %r", sent["id"], subject)

    return sent["id"]


def watch_inbox(topic: str, user_id: str | None = None) -> dict:
    """Ask Gmail to publish a notification to the Pub/Sub topic whenever the
    inbox changes. The watch lasts 7 days, so it is renewed every day. Returns
    {"history_id": where Gmail is now, "expires_ms": when the watch ends}."""
    body = {"topicName": topic, "labelIds": ["INBOX"], "labelFilterBehavior": "INCLUDE"}
    reply = get_gmail_service(user_id).users().watch(userId="me", body=body).execute(num_retries=RETRIES)

    return {"history_id": str(reply["historyId"]), "expires_ms": int(reply["expiration"])}


def new_message_ids(start_history_id: str, user_id: str | None = None) -> tuple[list[str], str] | None:
    """(ids of the messages that arrived in the inbox since `start_history_id`,
    the newest history id). None if Gmail no longer keeps that far back (it
    keeps about a week): the caller then starts again from now."""
    service = get_gmail_service(user_id)
    ids, page, latest = [], None, str(start_history_id)

    while True:
        try:
            reply = service.users().history().list(
                userId="me", startHistoryId=start_history_id, historyTypes=["messageAdded"],
                labelId="INBOX", pageToken=page,
            ).execute(num_retries=RETRIES)
        except HttpError as exc:
            if exc.resp.status == 404:
                return None
            raise

        for entry in reply.get("history", []):
            for added in entry.get("messagesAdded", []):
                message_id = added["message"]["id"]

                if message_id not in ids:
                    ids.append(message_id)

        latest = str(reply.get("historyId", latest))
        page = reply.get("nextPageToken")

        if not page:
            return ids, latest


@tool
def read_email_body(email_id: str) -> str:
    """Read the full text of one email, given its id from the list of recent emails."""

    try:
        service = get_gmail_service()

        body = parse_email(get_email(service, email_id))["body"]

        return body[:MAX_BODY_CHARS] or "(this email has no plain-text body)"

    except Exception:
        log.exception("read_email_body failed for %s", email_id)
        raise


class MeetingItem(BaseModel):
    email_id: str = Field(description="The id of the email the meeting came from")
    title: str = Field(description="Short title of the meeting")
    date: str = Field(
        description="Meeting date as YYYY-MM-DD, resolving words like "
                    "'tomorrow' from the email's sent date. Empty if not stated."
    )
    time: str = Field(
        description="Start time in 24-hour HH:MM. Empty if not stated."
    )
    mode: Literal["online", "offline", "unknown"] = Field(
        description="online for video/phone calls (Meet, Zoom, Teams, call), "
                    "offline for in-person, unknown if unclear"
    )
    location: str = Field(
        description="The meeting link for an online meeting. Empty for an "
                    "in-person meeting, or if no link is given."
    )
    city: str = Field(
        description="City of an in-person meeting. Empty for online meetings, "
                    "or if not stated."
    )
    venue: str = Field(
        description="Name of the place of an in-person meeting, such as an "
                    "office, building or restaurant. Empty if the email "
                    "only names a city or is vague, like 'our office'."
    )
    address: str = Field(
        description="Street address of an in-person meeting. Empty if not stated."
    )
    map_query: str = Field(
        default="",
        description="For an in-person meeting: a short text a map search can find, "
                    "the venue name and the city as a map would spell them, for example "
                    "'Bangalore International Exhibition Centre, Bengaluru'. Leave out "
                    "brackets, hall, room or floor numbers and directions like 'near'. If "
                    "only a city is known, use the city. Empty for online meetings.",
    )


@tool
def submit_meetings(meetings: list[MeetingItem]) -> str:
    """Submit the final list of meetings you found. Call this exactly once,
    when you have finished checking the emails. Pass an empty list if none
    of the emails are about a meeting."""

    return f"Recorded {len(meetings)} meeting(s)."


TOOLS = [read_email_body, submit_meetings]
