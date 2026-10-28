"""Accounts: register, log in, log out, and who is signed in.

How it fits together:
  * a password is stored only as a salted scrypt hash (never the password itself);
  * a successful login creates a login session: a random cookie value goes to the browser, and only the
    SHA-256 hash of it is stored, so a leaked database cannot be used to take over a session;
  * too many wrong passwords lock the account for a while;
  * every page and API (except the login and register pages) needs a valid session. There is no
    "login off" mode and no placeholder user: whatever runs, runs for a real signed-in user.
"""
import hashlib
import hmac
import os
import re
import secrets
import threading
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel

from app import db
from app.logger import get_logger


log = get_logger("auth")

STATIC_DIR = Path(__file__).parent / "static"

COOKIE = "meetpilot_session"
SESSION_DAYS = 7
MAX_FAILURES = 5            # wrong passwords before the account is locked ...
LOCK_MINUTES = 15           # ... for this long
MIN_PASSWORD = 12

# scrypt: memory-hard, so guessing passwords on a graphics card is expensive. The settings are stored in
# the hash itself, so they can be raised later without breaking old passwords.
SCRYPT = {"n": 2 ** 15, "r": 8, "p": 1}

COMMON_PASSWORDS = {
    "password", "password1", "password12", "password123", "passw0rd", "123456789012", "1234567890123",
    "qwertyuiop12", "qwerty123456", "iloveyou1234", "admin1234567", "welcome12345", "letmein12345",
    "meetpilot123", "meetpilot1234", "changeme1234", "abcdefghijkl", "111111111111", "000000000000",
}

EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]{2,}$")


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, maxmem=128 * 1024 * 1024, dklen=32, **SCRYPT)

    return f"scrypt${SCRYPT['n']}${SCRYPT['r']}${SCRYPT['p']}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """True if the password matches the stored hash. Compares in constant time."""
    try:
        scheme, n, r, p, salt, digest = stored.split("$")

        if scheme != "scrypt":
            return False

        expected = bytes.fromhex(digest)
        actual = hashlib.scrypt(password.encode("utf-8"), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p),
                                maxmem=128 * 1024 * 1024, dklen=len(expected))
    except (ValueError, TypeError):
        return False        # not a real hash (for example the placeholder user's "!")

    return hmac.compare_digest(actual, expected)


def needs_rehash(stored: str) -> bool:
    """Was this hash made with weaker settings than we use now?"""
    try:
        _, n, r, p, _, _ = stored.split("$")
        return (int(n), int(r), int(p)) != (SCRYPT["n"], SCRYPT["r"], SCRYPT["p"])
    except ValueError:
        return False


# a hash to check against when the email is unknown, so "no such user" takes as long as "wrong password"
_DUMMY_HASH = hash_password(secrets.token_urlsafe(16))


def password_problem(password: str, email: str) -> str | None:
    """Why this password is not allowed, or None. Length matters more than odd rules."""
    if len(password) < MIN_PASSWORD:
        return f"Use at least {MIN_PASSWORD} characters. A short phrase of a few words works well."

    if len(password) > 200:
        return "That password is too long."

    lowered = password.lower()
    local = email.split("@")[0].lower()

    if lowered in COMMON_PASSWORDS or len(set(password)) < 5:
        return "That password is too easy to guess. Try a few unrelated words together."

    if len(local) >= 4 and local in lowered:
        return "Don't use your email name inside the password."

    return None


# ---------------------------------------------------------------------------
# Login sessions
# ---------------------------------------------------------------------------

def _hash_cookie(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def cookie_secure() -> bool:
    """Set COOKIE_SECURE=1 once the site is served over HTTPS."""
    return os.getenv("COOKIE_SECURE", "0") == "1"


def start_session(user_id: str, request: Request) -> str:
    """A fresh cookie value for this user (a new one on every login)."""
    value = secrets.token_urlsafe(32)
    db.create_login_session(user_id, _hash_cookie(value), SESSION_DAYS, request.headers.get("user-agent"))

    return value


def set_cookie(response, value: str) -> None:
    response.set_cookie(
        COOKIE, value, max_age=SESSION_DAYS * 86400, httponly=True, samesite="lax",
        secure=cookie_secure(), path="/",
    )


def session_user(request: Request) -> dict | None:
    value = request.cookies.get(COOKIE)

    if not value:
        return None

    try:
        return db.get_login_session_user(_hash_cookie(value))
    except db.AccountsUnavailable:
        return None


def optional_user(request: Request) -> dict | None:
    """The signed-in user, or None."""
    return session_user(request)


def require_user(request: Request) -> dict:
    """For the APIs: the signed-in user, or 401."""
    user = optional_user(request)

    if user is None:
        raise HTTPException(401, "Please sign in.")

    return user


# ---------------------------------------------------------------------------
# A chat belongs to the first user who uses its id: nobody else may use it
# ---------------------------------------------------------------------------

_threads = {}
_threads_lock = threading.Lock()


def claim_thread(user: dict, thread_id: str) -> None:
    with _threads_lock:
        owner = _threads.setdefault(thread_id, user["id"])

    if owner != user["id"]:
        raise HTTPException(403, "That chat belongs to another account.")


# ---------------------------------------------------------------------------
# Too many attempts for an email that has no account (an account locks in the database)
# ---------------------------------------------------------------------------

_unknown = {}
_unknown_lock = threading.Lock()


def _unknown_locked(email: str) -> bool:
    now = time.time()

    with _unknown_lock:
        hits = [t for t in _unknown.get(email, []) if now - t < LOCK_MINUTES * 60]
        _unknown[email] = hits

        return len(hits) >= MAX_FAILURES


def _unknown_failed(email: str) -> None:
    with _unknown_lock:
        _unknown.setdefault(email, []).append(time.time())


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

router = APIRouter()


def _page(name: str) -> FileResponse:
    return FileResponse(STATIC_DIR / name, headers={"Cache-Control": "no-store"})


def _failure(status: int, message: str, field: str | None = None) -> JSONResponse:
    return JSONResponse({"error": message, "field": field}, status_code=status, headers={"Cache-Control": "no-store"})


def _check_header(request: Request) -> None:
    """A form on another website cannot add this header, so it can't make a signed-in browser act for us."""
    if request.headers.get("x-meetpilot") != "1":
        raise HTTPException(403, "Bad request.")


@router.get("/login")
def login_page(request: Request):
    return RedirectResponse("/", 303) if optional_user(request) else _page("login.html")


@router.get("/register")
def register_page(request: Request):
    return RedirectResponse("/", 303) if optional_user(request) else _page("register.html")


class Register(BaseModel):
    name: str = ""
    email: str
    password: str


class Login(BaseModel):
    email: str
    password: str


def _signed_in(user: dict, request: Request) -> JSONResponse:
    response = JSONResponse({"ok": True, "user": {"email": user["email"], "name": user.get("display_name") or ""}},
                            headers={"Cache-Control": "no-store"})
    set_cookie(response, start_session(user["id"], request))

    return response


@router.post("/auth/register")
def register(body: Register, request: Request):
    _check_header(request)

    email = body.email.strip()
    name = body.name.strip()[:80]

    if len(email) > 254 or not EMAIL.match(email):
        return _failure(400, "That doesn't look like an email address.", "email")

    problem = password_problem(body.password, email)

    if problem:
        return _failure(400, problem, "password")

    try:
        user = db.create_user(email, hash_password(body.password), name or None)
    except db.AccountsUnavailable as exc:
        log.error("register: %s", exc)
        return _failure(503, "Accounts are unavailable right now: the database can't be reached.")

    if user is None:
        return _failure(409, "An account with this email already exists. Try signing in.", "email")

    log.info("Registered %s", user["email"])

    return _signed_in(user, request)


@router.post("/auth/login")
def login(body: Login, request: Request):
    _check_header(request)

    email = body.email.strip()
    same_answer = "Wrong email or password."
    locked = "Too many attempts. Try again in a few minutes."

    try:
        user = db.get_user_by_email(email)

        if user is None:
            # Same work and the same answer as for a wrong password, so the page can't be used to find out who has an account.
            verify_password(body.password, _DUMMY_HASH)

            if _unknown_locked(email.lower()):
                return _failure(429, locked)

            _unknown_failed(email.lower())
            return _failure(401, same_answer)

        if user["locked_until"] and user["locked_until"].timestamp() > time.time():
            return _failure(429, locked)

        if user["status"] != "active" or not verify_password(body.password, user["password_hash"]):
            if user["status"] == "active":
                until = db.note_login_failure(user["id"], MAX_FAILURES, LOCK_MINUTES)
                log.warning("Wrong password for %s%s", user["email"], " (now locked)" if until else "")

                if until and until.timestamp() > time.time():
                    return _failure(429, locked)

            return _failure(401, same_answer)

        db.note_login_success(user["id"], hash_password(body.password) if needs_rehash(user["password_hash"]) else None)
    except db.AccountsUnavailable as exc:
        log.error("login: %s", exc)
        return _failure(503, "Accounts are unavailable right now: the database can't be reached.")

    log.info("Signed in %s", user["email"])

    return _signed_in(user, request)


@router.post("/auth/logout")
def logout(request: Request):
    _check_header(request)
    value = request.cookies.get(COOKIE)

    if value:
        try:
            db.delete_login_session(_hash_cookie(value))
        except db.AccountsUnavailable:
            pass

    response = JSONResponse({"ok": True}, headers={"Cache-Control": "no-store"})
    response.delete_cookie(COOKIE, path="/")

    return response


@router.get("/api/me")
def me(user: dict = Depends(require_user)):
    from app.activity import can_view      # imported here: activity.py imports this module

    return JSONResponse(
        {"email": user["email"], "name": user.get("display_name") or "", "can_view_activity": can_view(user)},
        headers={"Cache-Control": "no-store"},
    )
