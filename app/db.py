import os
import threading
import time
import uuid
from datetime import date, datetime
from functools import wraps

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool, PoolTimeout

from app.logger import get_logger
from app.tools.stay_tool import current_user, exchange_rate


load_dotenv()

log = get_logger("db")

RETRY_AFTER = 30    # seconds to leave the database alone after a connection failure

_pool = None
_pool_lock = threading.Lock()
_down_until = 0.0


def enabled() -> bool:
    return bool(os.getenv("DATABASE_URL"))


def get_pool() -> ConnectionPool:
    global _pool

    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ConnectionPool(
                    os.environ["DATABASE_URL"],
                    min_size=1, max_size=5, timeout=5,
                    kwargs={"connect_timeout": 5},
                    open=True,
                )

    return _pool


def close() -> None:
    """Close the pool (used by tests that switch databases)."""
    global _pool, _down_until

    if _pool is not None:
        _pool.close()

    _pool = None
    _down_until = 0.0


class NoUser(Exception):
    """A database call that belongs to a user was made without one. That is a bug in the code, so it
    is never hidden by safe(): it fails where it happens instead of quietly losing or misplacing data."""


def safe(default_factory=None):
    """The app must work without the database. A failing call logs a warning
    and returns an empty result; a connection failure also pauses all
    database calls for a short while, so a dead server doesn't slow every
    message."""

    def decorate(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            global _down_until

            if not enabled() or time.time() < _down_until:
                return default_factory() if default_factory else None

            try:
                return fn(*args, **kwargs)

            except NoUser:
                log.error("Database call %s was made without a user", fn.__name__)
                raise

            except (psycopg.OperationalError, PoolTimeout) as exc:
                _down_until = time.time() + RETRY_AFTER
                log.warning("Database unreachable (%s): %s. Pausing for %ds.",
                            fn.__name__, str(exc).strip().splitlines()[0] if str(exc).strip() else exc, RETRY_AFTER)

            except Exception:
                log.exception("Database call %s failed", fn.__name__)

            return default_factory() if default_factory else None

        return inner

    return decorate


def _user(user_id) -> str:
    """The user a call is for: the one given, else the user of the chat it runs inside.
    There is no default owner: a call with neither is a bug, so it fails instead of guessing."""
    user_id = user_id or current_user()

    if not user_id:
        raise NoUser("No user for this database call.")

    return str(user_id)


def as_uuid(value) -> str | None:
    """The value as a normal UUID string, or None if it isn't one."""
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, TypeError):
        return None


def _date(text):
    try:
        return date.fromisoformat(text) if text else None
    except ValueError:
        return None


def _time(text):
    try:
        return datetime.strptime(text, "%H:%M").time() if text else None
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# sessions and messages
# ---------------------------------------------------------------------------

@safe()
def ensure_session(session_id: str, timezone: str, title: str | None = None, user_id: str | None = None) -> None:
    """Create the chat, or touch it. A chat that belongs to another user is left alone: the WHERE stops
    anyone from taking over a chat by knowing its id."""
    with get_pool().connection() as conn:
        conn.execute(
            """
            INSERT INTO sessions (id, user_id, title, timezone)
            VALUES (%s::uuid, %s::uuid, %s, %s)
            ON CONFLICT (id) DO UPDATE
               SET updated_at = now(),
                   timezone   = EXCLUDED.timezone,
                   title      = COALESCE(sessions.title, EXCLUDED.title)
             WHERE sessions.user_id = EXCLUDED.user_id
            """,
            (session_id, _user(user_id), title, timezone),
        )


@safe(bool)
def session_belongs_to(session_id: str, user_id: str) -> bool:
    """Is this chat this user's? (False for a chat that doesn't exist.)"""
    with get_pool().connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM sessions WHERE id = %s::uuid AND user_id = %s::uuid", (session_id, _user(user_id))
        ).fetchone()

    return row is not None


@safe()
def add_message(session_id: str, role: str, kind: str = "text", content: str | None = None, payload=None,
                user_id: str | None = None) -> None:
    """Add a message to a chat, but only if the chat is this user's."""
    with get_pool().connection() as conn:
        done = conn.execute(
            """
            INSERT INTO messages (session_id, role, kind, content, payload)
            SELECT %s::uuid, %s, %s, %s, %s
             WHERE EXISTS (SELECT 1 FROM sessions WHERE id = %s::uuid AND user_id = %s::uuid)
            """,
            (session_id, role, kind, content, Jsonb(payload) if payload is not None else None, session_id, _user(user_id)),
        )

        if done.rowcount == 0:
            log.warning("add_message: chat %s does not belong to user %s; nothing saved", session_id, user_id)


# ---------------------------------------------------------------------------
# Gemini usage and cost, one row per chat request
# ---------------------------------------------------------------------------

@safe()
def record_usage(session_id, request_id, kind, model, calls, input_tokens, output_tokens, total_tokens, seconds,
                 user_id: str | None = None) -> None:
    try:
        input_price = float(os.environ["GEMINI_INPUT_USD_PER_1M"])
        output_price = float(os.environ["GEMINI_OUTPUT_USD_PER_1M"])
    except (KeyError, ValueError):
        log.warning("GEMINI_INPUT_USD_PER_1M / GEMINI_OUTPUT_USD_PER_1M are not set in .env: usage not recorded")
        return

    with get_pool().connection() as conn:
        found = exchange_rate("USD", "INR")

        if found is None:
            # The rate service is down: reuse the last rate that was stored.
            last = conn.execute("SELECT usd_to_inr, rate_date FROM gemini_usage ORDER BY id DESC LIMIT 1").fetchone()

            if last is None:
                log.warning("No USD->INR rate available: usage not recorded")
                return

            found = (float(last[0]), str(last[1]))

        rate, rate_date = found

        conn.execute(
            """
            INSERT INTO gemini_usage
                (session_id, request_id, request_kind, model, gemini_calls, input_tokens, output_tokens,
                 total_tokens, gemini_seconds, input_usd_per_1m, output_usd_per_1m, usd_to_inr, rate_date, user_id)
            VALUES (%s::uuid, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::date, %s::uuid)
            ON CONFLICT (request_id) DO NOTHING
            """,
            (session_id, request_id, kind, model, calls, input_tokens, output_tokens,
             total_tokens, seconds, input_price, output_price, rate, rate_date, _user(user_id)),
        )


# ---------------------------------------------------------------------------
# processed emails and meetings
# ---------------------------------------------------------------------------

@safe(set)
def known_emails(message_ids: list[str], user_id: str | None = None) -> set[str]:
    """Which of these Gmail messages this user's mailbox had analyzed before."""
    if not message_ids:
        return set()

    with get_pool().connection() as conn:
        rows = conn.execute(
            "SELECT message_id FROM processed_emails WHERE user_id = %s::uuid AND message_id = ANY(%s)",
            (_user(user_id), list(message_ids)),
        ).fetchall()

    return {row[0] for row in rows}


@safe()
def mark_processed(emails: list[tuple], session_id: str | None, user_id: str | None = None) -> None:
    """emails: (message_id, subject, is_meeting) for each email Gemini analyzed."""
    if not emails:
        return

    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO processed_emails (user_id, message_id, subject, is_meeting, session_id)
                VALUES (%s::uuid, %s, %s, %s, %s::uuid)
                ON CONFLICT DO NOTHING
                """,
                [(_user(user_id), mid, subject, is_meeting, session_id) for mid, subject, is_meeting in emails],
            )


@safe()
def save_meetings(meetings: list[dict], user_id: str | None = None) -> None:
    """Keep meetings Gemini found. A meeting already saved is left alone, so
    what happened to it since (booked, venue given) isn't overwritten."""
    rows = [m for m in meetings if m.get("email_id")]

    if not rows:
        return

    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO meetings
                    (user_id, email_id, title, sender, subject, meeting_date, meeting_time, mode,
                     location, city, venue, address)
                VALUES (%s::uuid, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                [
                    (_user(user_id), m["email_id"], m["title"], m.get("sender"), m.get("subject"),
                     _date(m.get("date")), _time(m.get("time")), m["mode"],
                     m.get("location") or None, m.get("city") or None, m.get("venue") or None, m.get("address") or None)
                    for m in rows
                ],
            )


def _meeting_dicts(rows) -> list[dict]:
    """Saved meetings shaped like the ones Gemini returns."""
    return [
        {
            "email_id": r["email_id"],
            "sender": r["sender"] or "",
            "subject": r["subject"] or "",
            "title": r["title"],
            "date": r["meeting_date"].isoformat() if r["meeting_date"] else "",
            "time": r["meeting_time"].strftime("%H:%M") if r["meeting_time"] else "",
            "mode": r["mode"],
            "location": r["location"] or "",
            "city": r["city"] or "",
            "venue": r["venue"] or "",
            "address": r["address"] or "",
            "status": r["status"],
        }
        for r in rows
    ]


@safe(list)
def saved_meetings(email_ids: list[str], user_id: str | None = None) -> list[dict]:
    """Meetings found earlier for these emails, this user's only."""
    if not email_ids:
        return []

    with get_pool().connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            rows = cur.execute(
                "SELECT * FROM meetings WHERE user_id = %s::uuid AND email_id = ANY(%s) ORDER BY id",
                (_user(user_id), list(email_ids)),
            ).fetchall()

    return _meeting_dicts(rows)


# ---------------------------------------------------------------------------
# Hotel bookings (demo or real), one row per booking reference
# ---------------------------------------------------------------------------

@safe()
def get_booking(reference: str, user_id: str | None = None) -> dict | None:
    with get_pool().connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            return cur.execute(
                "SELECT reference, status, hotel_name, total_price, currency FROM hotel_bookings WHERE user_id = %s::uuid AND reference = %s",
                (_user(user_id), reference),
            ).fetchone()


@safe()
def save_booking(reference, meeting, hotel, search, provider, payment_id, total, currency, status="confirmed",
                 user_id: str | None = None) -> None:
    with get_pool().connection() as conn:
        conn.execute(
            """
            INSERT INTO hotel_bookings
                (user_id, reference, email_id, meeting_title, hotel_name, hotel, check_in, check_out, guests,
                 total_price, currency, provider, payment_id, status)
            VALUES (%s::uuid, %s, %s, %s, %s, %s, %s::date, %s::date, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            """,
            (_user(user_id), reference, meeting.get("email_id"), meeting.get("title"), hotel.get("name"), Jsonb(hotel),
             search.get("check_in"), search.get("check_out"), search.get("guests"),
             total, currency, provider, payment_id, status),
        )


# ---------------------------------------------------------------------------
# Accounts and login sessions. Unlike the functions above these are NOT wrapped
# in safe(): if the database is down nobody can sign in, and the caller must
# say so instead of pretending nothing happened.
# ---------------------------------------------------------------------------

class AccountsUnavailable(Exception):
    """The database that holds the accounts can't be reached."""


def _accounts_pool() -> ConnectionPool:
    if not enabled():
        raise AccountsUnavailable("Accounts need the database. Set DATABASE_URL in .env.")

    try:
        return get_pool()
    except Exception as exc:
        raise AccountsUnavailable(f"The database is not reachable: {exc}") from exc


def _guard(fn):
    """Turn any connection problem into AccountsUnavailable."""
    @wraps(fn)
    def inner(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (psycopg.OperationalError, PoolTimeout) as exc:
            raise AccountsUnavailable(f"The database is not reachable: {exc}") from exc

    return inner


@_guard
def create_user(email: str, password_hash: str, display_name: str | None) -> dict | None:
    """Create an account. Returns None if the email is already used (ignoring case)."""
    try:
        with _accounts_pool().connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                return cur.execute(
                    "INSERT INTO users (email, display_name, password_hash) VALUES (%s, %s, %s) "
                    "RETURNING id::text AS id, email, display_name",
                    (email, display_name, password_hash),
                ).fetchone()
    except psycopg.errors.UniqueViolation:
        return None


@_guard
def get_user_by_email(email: str) -> dict | None:
    with _accounts_pool().connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            return cur.execute(
                "SELECT id::text AS id, email, display_name, password_hash, failed_logins, locked_until, status "
                "FROM users WHERE lower(email) = lower(%s)",
                (email,),
            ).fetchone()


@_guard
def note_login_failure(user_id: str, max_failures: int, lock_minutes: int):
    """Count a wrong password; after `max_failures` the account is locked for a while.
    Returns locked_until (or None)."""
    with _accounts_pool().connection() as conn:
        row = conn.execute(
            """
            UPDATE users
               SET failed_logins = failed_logins + 1,
                   locked_until  = CASE WHEN failed_logins + 1 >= %s
                                        THEN now() + make_interval(mins => %s) ELSE locked_until END
             WHERE id = %s::uuid
            RETURNING locked_until
            """,
            (max_failures, lock_minutes, user_id),
        ).fetchone()

    return row[0] if row else None


@_guard
def note_login_success(user_id: str, new_hash: str | None = None) -> None:
    """A correct password: clear the failures. `new_hash` replaces an old-style hash."""
    with _accounts_pool().connection() as conn:
        conn.execute(
            """
            UPDATE users
               SET failed_logins = 0, locked_until = NULL, last_login_at = now(),
                   password_hash = COALESCE(%s::text, password_hash),
                   password_changed_at = CASE WHEN %s::text IS NULL THEN password_changed_at ELSE now() END
             WHERE id = %s::uuid
            """,
            (new_hash, new_hash, user_id),
        )


@_guard
def create_login_session(user_id: str, id_hash: str, lifetime_days: int, user_agent: str | None) -> None:
    with _accounts_pool().connection() as conn:
        conn.execute(
            "INSERT INTO login_sessions (id_hash, user_id, expires_at, user_agent) "
            "VALUES (%s, %s::uuid, now() + make_interval(days => %s), %s)",
            (id_hash, user_id, lifetime_days, (user_agent or "")[:300] or None),
        )


@_guard
def get_login_session_user(id_hash: str) -> dict | None:
    """The user a login cookie belongs to, if the session exists, has not expired and the account is active."""
    with _accounts_pool().connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            return cur.execute(
                "SELECT u.id::text AS id, u.email, u.display_name FROM login_sessions s "
                "JOIN users u ON u.id = s.user_id "
                "WHERE s.id_hash = %s AND s.expires_at > now() AND u.status = 'active'",
                (id_hash,),
            ).fetchone()


@_guard
def delete_login_session(id_hash: str) -> None:
    with _accounts_pool().connection() as conn:
        conn.execute("DELETE FROM login_sessions WHERE id_hash = %s", (id_hash,))


@_guard
def delete_expired_login_sessions() -> int:
    with _accounts_pool().connection() as conn:
        return conn.execute("DELETE FROM login_sessions WHERE expires_at <= now()").rowcount


UPDATABLE = ("status", "venue", "calendar_event_id", "event_link", "stay")


@safe()
def update_meeting(meeting: dict, user_id: str | None = None, **fields) -> None:
    """Record what happened to a saved meeting (status, venue, calendar event,
    the hotel the user chose)."""
    fields = {k: (Jsonb(v) if k == "stay" else v) for k, v in fields.items() if k in UPDATABLE and v is not None}

    if not meeting.get("email_id") or not fields:
        return

    assignments = ", ".join(f"{column} = %s" for column in fields)

    with get_pool().connection() as conn:
        conn.execute(
            f"UPDATE meetings SET {assignments}, updated_at = now() "
            "WHERE user_id = %s::uuid AND email_id = %s AND title = %s",
            (*fields.values(), _user(user_id), meeting["email_id"], meeting["title"]),
        )


# ---------------------------------------------------------------------------
# Settings: each person's own preferences (the popup). Only the values the person
# changed are stored; everything else comes from data/preferences.json.
# ---------------------------------------------------------------------------

@safe(dict)
def get_user_settings(user_id: str) -> dict:
    """What this user changed: {"home_location": {...}, "travel_preferences": {...}, "hotel_preferences": {...}}.
    Empty if nothing was saved (or the database is down: the defaults are used then)."""
    with get_pool().connection() as conn:
        row = conn.execute(
            "SELECT home_location, travel_preferences, hotel_preferences FROM user_settings WHERE user_id = %s",
            (user_id,),
        ).fetchone()

    if not row:
        return {}

    return {"home_location": row[0], "travel_preferences": row[1], "hotel_preferences": row[2]}


@_guard
def save_user_settings(user_id: str, home: dict, travel: dict, hotel: dict) -> None:
    with _accounts_pool().connection() as conn:
        conn.execute(
            """
            INSERT INTO user_settings (user_id, home_location, travel_preferences, hotel_preferences)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (user_id) DO UPDATE SET
                home_location = EXCLUDED.home_location,
                travel_preferences = EXCLUDED.travel_preferences,
                hotel_preferences = EXCLUDED.hotel_preferences,
                updated_at = now()
            """,
            (user_id, Jsonb(home), Jsonb(travel), Jsonb(hotel)),
        )


@_guard
def reset_user_settings(user_id: str) -> None:
    with _accounts_pool().connection() as conn:
        conn.execute("DELETE FROM user_settings WHERE user_id = %s", (user_id,))


# ---------------------------------------------------------------------------
# Gmail connections: each user's own link to their Gmail (the encrypted token lives here).
# These are not wrapped in safe(): without the database there is no Gmail for anyone.
# ---------------------------------------------------------------------------

@_guard
def get_gmail_connection(user_id: str) -> dict | None:
    """What is safe to show: the address and the state. Never the token."""
    with _accounts_pool().connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            return cur.execute(
                "SELECT google_email, scopes, status, connected_at, last_sync_at, last_error "
                "FROM gmail_connections WHERE user_id = %s",
                (user_id,),
            ).fetchone()


@_guard
def get_gmail_secret(user_id: str) -> dict | None:
    """The encrypted token and scopes, for the code that talks to Google."""
    with _accounts_pool().connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            return cur.execute(
                "SELECT refresh_token_enc, scopes, status, google_email FROM gmail_connections WHERE user_id = %s",
                (user_id,),
            ).fetchone()


@_guard
def save_gmail_connection(user_id: str, google_email: str, scopes: list[str], token_enc: bytes) -> bool:
    """Connect (or reconnect) a user. False if that Gmail address already belongs to another user."""
    try:
        with _accounts_pool().connection() as conn:
            conn.execute(
                """
                INSERT INTO gmail_connections (user_id, google_email, scopes, refresh_token_enc, status)
                VALUES (%s, %s, %s, %s, 'connected')
                ON CONFLICT (user_id) DO UPDATE SET
                    google_email = EXCLUDED.google_email,
                    scopes = EXCLUDED.scopes,
                    refresh_token_enc = EXCLUDED.refresh_token_enc,
                    status = 'connected',
                    last_error = NULL,
                    connected_at = now(),
                    history_id = CASE WHEN lower(gmail_connections.google_email) = lower(EXCLUDED.google_email)
                                      THEN gmail_connections.history_id END,
                    watch_expires_at = CASE WHEN lower(gmail_connections.google_email) = lower(EXCLUDED.google_email)
                                            THEN gmail_connections.watch_expires_at END
                """,
                (user_id, google_email, scopes, token_enc),
            )
        return True
    except psycopg.errors.UniqueViolation:
        return False


@_guard
def set_gmail_status(user_id: str, status: str, error: str | None = None) -> None:
    with _accounts_pool().connection() as conn:
        conn.execute(
            "UPDATE gmail_connections SET status = %s, last_error = %s WHERE user_id = %s",
            (status, error, user_id),
        )


@_guard
def delete_gmail_connection(user_id: str) -> bool:
    with _accounts_pool().connection() as conn:
        return conn.execute("DELETE FROM gmail_connections WHERE user_id = %s", (user_id,)).rowcount > 0


@_guard
def gmail_user_for_address(google_email: str) -> str | None:
    """The user a Gmail address belongs to, if that connection is working."""
    with _accounts_pool().connection() as conn:
        row = conn.execute(
            "SELECT user_id::text FROM gmail_connections WHERE lower(google_email) = lower(%s) AND status = 'connected'",
            (google_email,),
        ).fetchone()

    return row[0] if row else None


@_guard
def connected_gmail_users() -> list[str]:
    with _accounts_pool().connection() as conn:
        return [r[0] for r in conn.execute("SELECT user_id::text FROM gmail_connections WHERE status = 'connected'").fetchall()]


@_guard
def get_gmail_position(user_id: str) -> str | None:
    with _accounts_pool().connection() as conn:
        row = conn.execute("SELECT history_id FROM gmail_connections WHERE user_id = %s", (user_id,)).fetchone()

    return row[0] if row else None


@_guard
def save_gmail_position(user_id: str, history_id: str | None = None, watch_expires_ms: int | None = None) -> None:
    """Remember how far this user's mailbox has been read, and when their watch ends."""
    with _accounts_pool().connection() as conn:
        conn.execute(
            """
            UPDATE gmail_connections SET
                history_id = COALESCE(%s, history_id),
                watch_expires_at = COALESCE(to_timestamp(%s / 1000.0), watch_expires_at),
                last_sync_at = now()
            WHERE user_id = %s
            """,
            (history_id, watch_expires_ms, user_id),
        )


# ---------------------------------------------------------------------------
# Agent activity: the steps of each run, for the Agent activity page
# ---------------------------------------------------------------------------

@safe()
def record_run(session_id, request_id: str, kind: str, label: str, steps: list, user_id: str | None = None) -> None:
    with get_pool().connection() as conn:
        conn.execute(
            """
            INSERT INTO agent_runs (user_id, session_id, request_id, kind, label, steps)
            VALUES (%s::uuid, %s::uuid, %s, %s, %s, %s)
            ON CONFLICT (request_id) DO NOTHING
            """,
            (_user(user_id), session_id, request_id, kind, (label or "")[:300] or "(no text)", Jsonb(steps)),
        )


@_guard
def list_runs(user_id: str, limit: int = 50) -> dict:
    """This user's latest runs (newest first) with their Gemini numbers, and their all-time totals."""
    with _accounts_pool().connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            runs = cur.execute(
                """
                SELECT r.request_id, r.kind, r.label, r.steps, r.created_at,
                       COALESCE(u.gemini_calls, 0) AS calls, COALESCE(u.total_tokens, 0) AS tokens,
                       COALESCE(u.gemini_seconds, 0) AS seconds, COALESCE(u.cost_inr, 0) AS cost_inr
                FROM agent_runs r
                LEFT JOIN gemini_usage u ON u.request_id = r.request_id AND u.user_id = r.user_id
                WHERE r.user_id = %s
                ORDER BY r.created_at DESC, r.id DESC
                LIMIT %s
                """,
                (user_id, limit),
            ).fetchall()

            totals = cur.execute(
                """
                SELECT COUNT(*) AS requests, COALESCE(SUM(gemini_calls), 0) AS calls,
                       COALESCE(SUM(total_tokens), 0) AS tokens, COALESCE(SUM(cost_inr), 0) AS cost_inr
                FROM gemini_usage WHERE user_id = %s
                """,
                (user_id,),
            ).fetchone()

            runs_total = cur.execute("SELECT COUNT(*) AS n FROM agent_runs WHERE user_id = %s", (user_id,)).fetchone()["n"]

    return {"runs": runs, "totals": {**totals, "runs": runs_total}}
