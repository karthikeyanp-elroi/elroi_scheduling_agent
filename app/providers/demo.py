"""A pretend hotel supplier and payment gateway, for the POC.

Nothing is charged and no room is reserved. It follows the same four calls a
real provider would (see app/providers/__init__.py), with short pauses so the
page can show the payment and booking steps as they happen.
"""
import hashlib
import time


class DemoProvider:
    name = "demo"
    demo = True

    def __init__(self, step_seconds: float = 1.0):
        self.step_seconds = step_seconds

    def reference(self, meeting: dict, hotel: dict, search: dict) -> str:
        """The same booking always gets the same reference, so repeating it can't book twice."""
        key = "|".join(str(part) for part in (
            meeting.get("email_id"), meeting.get("title"), hotel.get("name"),
            search.get("check_in"), search.get("check_out"), search.get("guests"),
        ))

        return "DEMO-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:8].upper()

    def quote(self, hotel: dict, search: dict) -> dict:
        """Re-check the price and availability. The demo trusts the search."""
        total = hotel.get("total_price")

        if total is None and hotel.get("price_per_night") is not None:
            total = hotel["price_per_night"] * int(search.get("nights") or 1)

        return {"available": total is not None, "total": total, "currency": hotel.get("currency"),
                "note": "" if total is not None else "The search gave no price for these dates."}

    def pay(self, quote: dict, reference: str, progress) -> dict:
        for label in ("Securing your payment (demo)…", "Authorising the card (demo)…"):
            progress(label)
            time.sleep(self.step_seconds)

        return {"status": "paid", "payment_id": "PAY-" + reference.split("-")[-1],
                "amount": quote["total"], "currency": quote["currency"]}

    def book(self, hotel: dict, search: dict, reference: str, payment: dict, progress) -> dict:
        progress("Confirming with the hotel (demo)…")
        time.sleep(self.step_seconds)

        return {"status": "confirmed", "reference": reference}
