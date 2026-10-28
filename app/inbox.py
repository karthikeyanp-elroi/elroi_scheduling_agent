from app import db


def remember_analysis(final: dict, session_id: str | None, user_id: str | None = None) -> tuple[list[dict], list[dict]]:
    """Save what the Gmail agent just analyzed, so Gemini is never paid for
    the same email twice. Returns (the new meetings, every meeting to show):
    the new ones plus the ones found earlier for the other listed emails."""
    new = final.get("meetings", [])
    analyzed = final.get("emails", [])
    listed = final.get("listed_ids", [])

    with_meeting = {m["email_id"] for m in new}
    db.mark_processed([(e["id"], e["subject"], e["id"] in with_meeting) for e in analyzed], session_id, user_id)
    db.save_meetings(new, user_id)

    # Emails analyzed on an earlier check come from the database.
    analyzed_ids = {e["id"] for e in analyzed}
    earlier = db.saved_meetings([i for i in listed if i not in analyzed_ids], user_id)
    position = {email_id: i for i, email_id in enumerate(listed)}
    found = sorted(new + earlier, key=lambda m: position.get(m.get("email_id"), len(position)))

    return new, found
