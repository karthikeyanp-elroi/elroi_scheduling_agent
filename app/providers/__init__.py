"""Where hotels get booked and paid for.

The agents talk to a provider through four calls, so a real supplier or a
payment gateway (Razorpay or Stripe in test mode, then live) can replace the
demo one without touching the agents or the cards:

    provider.reference(meeting, hotel, search)    a stable id, so a retry can never book twice
    provider.quote(hotel, search)                 re-check price and availability just before paying
    provider.pay(quote, reference, progress)      take the payment
    provider.book(hotel, search, reference, payment, progress)    reserve the room

`progress(label)` tells the page what is happening. A failure is raised as an
exception with a message the user can read. `provider.name` is stored with the
booking, and `provider.demo` is True when nothing real happens (no charge, no
reservation), so every screen and email can say so.
"""
from app.tools.stay_tool import StayError, load_preferences


def get_provider():
    """The provider named in data/preferences.json (booking.provider)."""
    settings = load_preferences().get("booking") or {}
    name = settings.get("provider", "demo")

    if name == "demo":
        from app.providers.demo import DemoProvider

        return DemoProvider(float(settings.get("demo_step_seconds", 1.0)))

    raise StayError(f"Unknown booking provider {name!r}. Set booking.provider in data/preferences.json.")
