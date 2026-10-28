"""The Agent activity page: what MeetPilot and its sub-agents did, run by run, with the Gemini numbers."""
from datetime import datetime
from decimal import Decimal
import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

from app import auth, db


router = APIRouter()

PAGE = Path(__file__).parent / "static" / "activity.html"


def can_view(user: dict) -> bool:
    """Only the emails listed in ACTIVITY_ALLOWED_EMAILS (.env, comma separated) may see this page.
    If the setting is empty, nobody can."""
    allowed = {e.strip().lower() for e in os.getenv("ACTIVITY_ALLOWED_EMAILS", "").split(",") if e.strip()}

    return (user.get("email") or "").lower() in allowed


def _plain(value):
    """JSON-friendly: Decimal -> float, datetime -> text."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


@router.get("/activity")
def page(request: Request):
    user = auth.optional_user(request)

    if user is None:
        return RedirectResponse("/login", 303)

    if not can_view(user):
        raise HTTPException(404)        # for everyone else the page does not exist

    return FileResponse(PAGE, headers={"Cache-Control": "no-store"})


@router.get("/api/activity")
def activity(limit: int = Query(50, ge=1, le=200), user: dict = Depends(auth.require_user)):
    if not can_view(user):
        raise HTTPException(403, "Not allowed.")

    try:
        found = db.list_runs(user["id"], limit)
    except db.AccountsUnavailable as exc:
        raise HTTPException(503, str(exc))

    return JSONResponse(_plain(found), headers={"Cache-Control": "no-store"})
