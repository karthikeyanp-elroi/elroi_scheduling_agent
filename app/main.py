import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import activity, auth, db, events, gmail_connect, inbox_push, settings
from app.chat import chat_events
from app.graph.meetpilot_graph import meetpilot
from app.streaming import sse


STATIC_DIR = Path(__file__).parent / "static"

@asynccontextmanager
async def lifespan(_app):
    """Gmail push runs for as long as the server does (when it is set up in .env)."""
    try:
        db.delete_expired_login_sessions()      # tidy up login sessions that ran out
    except db.AccountsUnavailable:
        pass

    inbox_push.start()
    yield
    inbox_push.stop()


app = FastAPI(title="MeetPilot", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# /login, /register, /auth/* and /api/me
app.include_router(auth.router)

# /api/settings: each person's preferences
app.include_router(settings.router)

# /connect/gmail and /api/gmail/status: each user connects their own Gmail
app.include_router(gmail_connect.router)

# /activity and /api/activity: what the agents did
app.include_router(activity.router)


@app.get("/")
def index(request: Request):
    """The chat page, for signed-in users. Everyone else goes to the login page."""
    if auth.optional_user(request) is None:
        return RedirectResponse("/login", 303)

    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})


class ChatRequest(BaseModel):
    thread_id: str
    message: str | None = None      # a new user message ...
    resume: Any = None              # ... or the answer to a pause: {"approved": true} or text
    timezone: str = "UTC"
    debug: bool = False


@app.post("/api/chat")
def chat(request: ChatRequest, user: dict = Depends(auth.require_user)):
    """One turn of the MeetPilot chat, streamed as server-sent events."""
    if (request.message is None) == (request.resume is None):
        raise HTTPException(400, "Send either `message` or `resume`.")

    auth.claim_thread(user, request.thread_id)

    return StreamingResponse(
        chat_events(
            request.thread_id, request.message, request.resume,
            request.timezone, request.debug, user["id"],
        ),
        media_type="text/event-stream",
    )


# ---------------------------------------------------------------------------
# Email events: something tells MeetPilot a new email arrived, it analyzes the
# email in the background, and the page hears about it on a live stream.
# ---------------------------------------------------------------------------

class FakeEmail(BaseModel):
    id: str | None = None
    sender: str | None = None
    subject: str | None = None
    body: str = ""
    date: str | None = None


class EmailEvent(BaseModel):
    user_id: str                    # whose mailbox it arrived in
    message_id: str | None = None   # a real email, fetched from Gmail ...
    email: FakeEmail | None = None  # ... or a made-up one


LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}


@app.post("/api/dev/email-event", status_code=202)
def email_event(event: EmailEvent, request: Request):
    """Development only: pretend a new email arrived. Switched off (404)
    unless DEV_FAKE_EVENTS=1 is in .env, and answers only to this machine."""
    if os.getenv("DEV_FAKE_EVENTS") != "1":
        raise HTTPException(404)

    if (request.client.host if request.client else "") not in LOCAL_HOSTS:
        raise HTTPException(403)

    if (event.message_id is None) == (event.email is None):
        raise HTTPException(400, "Send either `message_id` or `email`.")

    return {"event_id": events.submit(event.model_dump(exclude_none=True))}


class ShowMeetings(BaseModel):
    thread_id: str
    event_id: str


@app.post("/api/show-meetings")
def show_meetings(request: ShowMeetings, user: dict = Depends(auth.require_user)):
    """The "Show meeting" button of an email alert: put that email's meetings
    in the chat's memory, so "Book meeting 1" works on them, and return them."""
    auth.claim_thread(user, request.thread_id)
    found = events.meetings_of(request.event_id, user["id"])

    if found is None:
        raise HTTPException(404, "That email is no longer available. Use Check my inbox.")

    config = {"configurable": {"thread_id": request.thread_id}}
    meetpilot.update_state(config, {"meetings": found}, as_node="tools")

    return {"meetings": found}


class ChooseHotel(BaseModel):
    thread_id: str
    index: int      # position in the hotel list the user was shown ...
    name: str       # ... and its name, to be sure it is the same list


@app.post("/api/choose-hotel")
def choose_hotel(request: ChooseHotel, user: dict = Depends(auth.require_user)):
    """The "Choose" button on a hotel card: remember the hotel in the chat's
    memory and in the database. No Gemini call."""
    auth.claim_thread(user, request.thread_id)
    config = {"configurable": {"thread_id": request.thread_id}}
    options = meetpilot.get_state(config).values.get("stay_options") or {}
    hotels = options.get("hotels", [])

    if not (0 <= request.index < len(hotels)) or hotels[request.index]["name"] != request.name:
        raise HTTPException(409, "Those hotels are out of date. Search again to choose one.")

    hotel = hotels[request.index]
    meetpilot.update_state(config, {"stay_options": {**options, "chosen": hotel}}, as_node="tools")

    search = options.get("search", {})
    db.update_meeting(
        {"email_id": options.get("email_id"), "title": options.get("title")},
        user_id=user["id"],
        stay={"hotel": hotel, "check_in": search.get("check_in"), "check_out": search.get("check_out"), "guests": search.get("guests")},
    )

    return {"hotel": hotel, "meeting_number": options.get("meeting_number")}


@app.get("/api/events")
async def event_stream(user: dict = Depends(auth.require_user)):
    """A live stream the page keeps open. Each email event reports here:
    email_received, then new_meetings, no_meeting, email_skipped or event_error."""

    async def generate():
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()

        # Called from the worker threads; hands the message to this stream.
        def deliver(event, data):
            loop.call_soon_threadsafe(queue.put_nowait, (event, data))

        events.broadcaster.subscribe(deliver, user["id"])

        try:
            yield ": connected\n\n"

            while True:
                try:
                    event, data = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"          # keeps the connection alive
                    continue

                yield sse(event, data)

        finally:
            events.broadcaster.unsubscribe(deliver)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )
