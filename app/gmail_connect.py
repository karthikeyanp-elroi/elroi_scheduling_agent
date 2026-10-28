"""Connect a user's own Gmail to MeetPilot.

Logging in to MeetPilot and letting it read Gmail are two separate permissions. This is the second one:
the user is sent to Google's consent screen, comes back with a one-time code, and MeetPilot exchanges it
for a refresh token which is stored ENCRYPTED in gmail_connections (see app/crypto.py).

Everything that talks to Google is in the four small `google_*` functions at the top, so tests can replace
them. In the running app they are always the real ones: there is no setting that switches Google off."""
import os
import secrets
import threading
import time

import requests
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse

from app import auth, crypto, db
from app.logger import get_logger
from app.tools.gmail_tool import SCOPES, GmailNotConnected, forget_credentials, get_gmail_service, oauth_client, timed_http


log = get_logger("gmail-connect")

router = APIRouter()

os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")

STATE_MINUTES = 10   
DEFAULT_REDIRECT = "http://127.0.0.1:8000/connect/gmail/callback"



def redirect_uri() -> str:
    if os.getenv("OAUTH_REDIRECT_URI"):
        return os.environ["OAUTH_REDIRECT_URI"]

    for uri in oauth_client().get("redirect_uris", []):
        if uri.endswith("/connect/gmail/callback"):
            return uri

    return DEFAULT_REDIRECT


def _flow(verifier: str | None = None):
    from google_auth_oauthlib.flow import Flow

    flow = Flow.from_client_config(
        {"web": oauth_client()}, scopes=SCOPES, redirect_uri=redirect_uri(),
        autogenerate_code_verifier=verifier is None, code_verifier=verifier,
    )

    return flow


def google_authorize_url(state: str) -> tuple[str, str]:
    """(the Google consent address, the PKCE verifier to keep until the user comes back)."""
    flow = _flow()
    url, _ = flow.authorization_url(
        state=state, access_type="offline", prompt="consent", include_granted_scopes="false",
    )

    return url, flow.code_verifier


def google_exchange(code: str, verifier: str) -> dict:
    """Trade the one-time code for tokens. {"refresh_token", "scopes", "email"}."""
    from googleapiclient.discovery import build

    flow = _flow(verifier)
    flow.fetch_token(code=code)
    creds = flow.credentials

    profile = build("gmail", "v1", http=timed_http(creds)).users().getProfile(userId="me").execute(num_retries=2)

    return {
        "refresh_token": creds.refresh_token,
        "scopes": sorted(creds.granted_scopes or creds.scopes or []),
        "email": profile["emailAddress"],
    }


def google_revoke(token: str) -> None:
    """Tell Google MeetPilot's access is over (best effort)."""
    requests.post("https://oauth2.googleapis.com/revoke", params={"token": token}, timeout=10)



_pending = {}      
_pending_lock = threading.Lock()


def _new_state(user_id: str, verifier_for) -> tuple[str, str]:
    """(state, consent address). The state is random, single use and expires."""
    state = secrets.token_urlsafe(32)
    url, verifier = verifier_for(state)
    now = time.time()

    with _pending_lock:
        for key in [k for k, (_, _, until) in _pending.items() if until < now]:
            del _pending[key]

        if len(_pending) > 500:
            _pending.clear()

        _pending[state] = (user_id, verifier, now + STATE_MINUTES * 60)

    return state, url


def _take_state(state: str, user_id: str) -> str | None:
    """The verifier for this state, if it is valid and was started by this user. Used up either way."""
    with _pending_lock:
        found = _pending.pop(state, None)

    if not found:
        return None

    owner, verifier, until = found

    return verifier if owner == user_id and until >= time.time() else None



def _problem_with_setup() -> str | None:
    """Why connecting can't work on this server right now, if it can't."""
    if not crypto.available():
        return "Gmail connection isn't set up on this server: TOKEN_ENCRYPTION_KEY is missing in .env."

    try:
        oauth_client()
    except GmailNotConnected as exc:
        return str(exc)

    return None


def _no_store(body: dict, status: int = 200) -> JSONResponse:
    return JSONResponse(body, status_code=status, headers={"Cache-Control": "no-store"})


@router.get("/api/gmail/status")
def status(user: dict = Depends(auth.require_user)):
    try:
        row = db.get_gmail_connection(user["id"])
    except db.AccountsUnavailable as exc:
        raise HTTPException(503, str(exc))

    problem = _problem_with_setup()
    answer = {"available": problem is None, "problem": problem, "state": row["status"] if row else "not_connected"}

    if row:
        answer["email"] = row["google_email"]
        answer["connected_at"] = row["connected_at"].isoformat()

    return _no_store(answer)


@router.get("/connect/gmail")
def start(request: Request):
    """Send the signed-in user to Google's consent screen."""
    user = auth.optional_user(request)

    if user is None:
        return RedirectResponse("/login", status_code=303)

    if _problem_with_setup():
        return RedirectResponse("/?gmail=unavailable", status_code=303)

    _, url = _new_state(user["id"], google_authorize_url)

    return RedirectResponse(url, status_code=303, headers={"Cache-Control": "no-store"})


def _done(code: str) -> RedirectResponse:
    return RedirectResponse(f"/?gmail={code}", status_code=303, headers={"Cache-Control": "no-store"})


@router.get("/connect/gmail/callback")
def callback(request: Request, code: str | None = None, state: str | None = None, error: str | None = None):
    """Google sends the user back here."""
    user = auth.optional_user(request)

    if user is None:
        return RedirectResponse("/login", status_code=303)

    verifier = _take_state(state or "", user["id"])

    if verifier is None:
        log.warning("Gmail callback with an unknown or expired state (user %s)", user["id"])
        return _done("expired")

    if error or not code:
        log.info("User %s did not allow Gmail access (%s)", user["id"], error)
        return _done("denied")

    try:
        got = google_exchange(code, verifier)
    except Exception as exc:
        log.warning("Google token exchange failed for user %s: %s", user["id"], exc)
        return _done("failed")

    if not got.get("refresh_token"):
        return _done("failed")

    if not set(SCOPES) <= set(got["scopes"]):
        log.info("User %s allowed only part of what MeetPilot needs: %s", user["id"], got["scopes"])
        return _done("scopes")

    try:
        saved = db.save_gmail_connection(user["id"], got["email"], got["scopes"], crypto.encrypt(got["refresh_token"]))
    except (db.AccountsUnavailable, crypto.KeyProblem) as exc:
        log.error("Could not save the Gmail connection: %s", exc)
        return _done("failed")

    if not saved:
        log.warning("Gmail address already belongs to another MeetPilot user (asked by %s)", user["id"])
        return _done("taken")

    forget_credentials(user["id"])
    log.info("Gmail connected for user %s", user["id"])

    # start listening to this mailbox without waiting for a restart
    from app import inbox_push

    inbox_push.user_connected(user["id"])

    return _done("connected")


@router.post("/connect/gmail/disconnect")
def disconnect(request: Request, user: dict = Depends(auth.require_user)):
    if request.headers.get("X-MeetPilot") != "1":
        raise HTTPException(403, "Missing header.")

    try:
        secret = db.get_gmail_secret(user["id"])

        if secret:
            token = None

            try:
                token = crypto.decrypt(secret["refresh_token_enc"])
            except crypto.KeyProblem:
                pass

            # stop Gmail's notifications, then take MeetPilot's access away at Google; both best effort
            try:
                get_gmail_service(user["id"]).users().stop(userId="me").execute()
            except Exception as exc:
                log.info("Could not stop the Gmail watch for user %s: %s", user["id"], exc)

            if token:
                try:
                    google_revoke(token)
                except Exception as exc:
                    log.info("Could not revoke the Gmail token at Google for user %s: %s", user["id"], exc)

        db.delete_gmail_connection(user["id"])
    except db.AccountsUnavailable as exc:
        raise HTTPException(503, str(exc))

    forget_credentials(user["id"])

    return _no_store({"state": "not_connected"})
