"""Gmail push: Gmail -> Pub/Sub topic -> pull subscription -> this worker -> events.submit().

A Pub/Sub message only says "the mailbox of this address changed" (with a history number), not which
mail arrived. The worker finds the MeetPilot user who connected that address, asks Gmail what is new
since the last history number handled for THAT user, and sends each new mail to the same pipeline the
fake terminal events use, for that user only. Every connected user has their own Gmail watch and their
own position. Nothing here needs a public address: the worker pulls from Pub/Sub itself.
"""
import json
import os
import threading

from app import db, events
from app.logger import get_logger
from app.tools.gmail_tool import GmailNotConnected, new_message_ids, watch_inbox


log = get_logger("push")

RENEW_SECONDS = 24 * 3600   # Gmail's watch lasts 7 days; it is renewed daily
RETRY_SECONDS = 60          # wait before trying again after a failure


def default_subscriber():
    from google.cloud import pubsub_v1      # uses GOOGLE_APPLICATION_CREDENTIALS

    return pubsub_v1.SubscriberClient()


class PushWorker:
    def __init__(self, topic: str, subscription: str, subscriber_factory=default_subscriber):
        self.topic = topic
        self.subscription = subscription
        self._make_subscriber = subscriber_factory
        self._locks = {}                    # one notification per user is handled at a time
        self._locks_guard = threading.Lock()
        self._stop = threading.Event()
        self._subscriber = None
        self._future = None

    def _lock_for(self, user_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(user_id, threading.Lock())
    
    def start(self) -> None:
        threading.Thread(target=self._pull_loop, name="gmail-push", daemon=True).start()
        threading.Thread(target=self._renew_loop, name="gmail-watch", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

        try:
            if self._future:
                self._future.cancel()

            if self._subscriber:
                self._subscriber.close()
        except Exception:
            log.exception("Gmail push: could not stop cleanly")

    # ---- starting up -----------------------------------------------------------------------

    def start_user(self, user_id: str) -> None:
        """Register this user's watch and decide where to start: where the last run stopped
        (and catch up on mail that arrived meanwhile), or from now."""
        try:
            info = watch_inbox(self.topic, user_id)
            saved = db.get_gmail_position(user_id)
            db.save_gmail_position(user_id, None if saved else info["history_id"], info["expires_ms"])

            if saved:
                log.info("Gmail push: user %s continues from history %s", user_id, saved)
                self.sync(user_id)
            else:
                log.info("Gmail push: user %s starts from now (history %s)", user_id, info["history_id"])

        except GmailNotConnected as exc:
            log.warning("Gmail push: user %s is not reachable: %s", user_id, exc)

    def begin(self) -> None:
        for user_id in db.connected_gmail_users():
            self.start_user(user_id)

    def _pull_loop(self) -> None:
        begun = False

        while not self._stop.is_set():
            try:
                if not begun:
                    self.begin()
                    begun = True

                self._subscriber = self._make_subscriber()
                self._future = self._subscriber.subscribe(self.subscription, callback=self.on_message)
                log.info("Gmail push: listening on %s", self.subscription)
                self._future.result()       # blocks until stopped, or a failure that can't recover

            except Exception as exc:
                if self._stop.is_set():
                    return

                log.error("Gmail push failed: %s. Trying again in %d s.", exc, RETRY_SECONDS)

            if self._stop.wait(RETRY_SECONDS):
                return

    def _renew_loop(self) -> None:
        while not self._stop.wait(RENEW_SECONDS):
            try:
                users = db.connected_gmail_users()
            except Exception:
                log.exception("Gmail push: could not list the connected users; trying again tomorrow")
                continue

            for user_id in users:
                try:
                    info = watch_inbox(self.topic, user_id)
                    db.save_gmail_position(user_id, None, info["expires_ms"])
                    log.info("Gmail push: watch renewed for user %s (expires at %s ms)", user_id, info["expires_ms"])
                except Exception:
                    log.exception("Gmail push: could not renew the watch of user %s; trying again tomorrow", user_id)

    # ---- handling a notification -----------------------------------------------------------

    def on_message(self, message) -> None:
        """A Pub/Sub message arrived. It is always acknowledged: if reading Gmail failed, the
        next notification picks up from the same position, so nothing is lost."""
        try:
            note = json.loads(message.data.decode("utf-8"))
            address = note.get("emailAddress") or ""
            user_id = db.gmail_user_for_address(address)

            if not user_id:
                log.info("Gmail notification for an address nobody has connected: ignored")
            else:
                log.info("Gmail notification for user %s: history %s", user_id, note.get("historyId"))
                self.sync(user_id)
        except Exception:
            log.exception("Gmail push: could not read a notification")
        finally:
            message.ack()

    def sync(self, user_id: str) -> None:
        """Find the mail that arrived in this user's inbox since their last position and hand each to the agent."""
        with self._lock_for(user_id):
            try:
                position = db.get_gmail_position(user_id)

                if position is None:
                    position = watch_inbox(self.topic, user_id)["history_id"]
                    db.save_gmail_position(user_id, position)
                    log.info("Gmail push: user %s had no position: starting from now (%s)", user_id, position)
                    return

                found = new_message_ids(position, user_id)

                if found is None:
                    # Gmail no longer keeps that far back: start again from now.
                    position = watch_inbox(self.topic, user_id)["history_id"]
                    log.warning("Gmail no longer has the old history of user %s: starting again from now (%s)", user_id, position)
                else:
                    ids, position = found

                    for message_id in ids:
                        log.info("Gmail push: new mail %s for user %s", message_id, user_id)
                        events.submit({"message_id": message_id, "user_id": user_id})

                db.save_gmail_position(user_id, position)

            except GmailNotConnected as exc:
                log.warning("Gmail push: user %s: %s", user_id, exc)
            except Exception:
                log.exception("Gmail push: could not read the new mail of user %s; will try again on the next notification", user_id)


_worker = None


def start() -> None:
    """Start the worker if Gmail push is set up in .env; otherwise do nothing."""
    global _worker

    topic = os.getenv("GMAIL_PUBSUB_TOPIC")
    subscription = os.getenv("GMAIL_PUBSUB_SUBSCRIPTION")

    if not (topic and subscription):
        log.info("Gmail push is off: GMAIL_PUBSUB_TOPIC / GMAIL_PUBSUB_SUBSCRIPTION are not set in .env")
        return

    key = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

    if not os.path.isfile(key):
        log.error("Gmail push is off: GOOGLE_APPLICATION_CREDENTIALS does not point to a key file (%r)", key)
        return

    _worker = PushWorker(topic, subscription)
    _worker.start()


def stop() -> None:
    global _worker

    if _worker:
        _worker.stop()
        _worker = None


def user_connected(user_id: str) -> None:
    """A user just connected Gmail: start watching their inbox now (when push is running)."""
    if _worker:
        threading.Thread(target=_worker.start_user, args=(user_id,), name="gmail-connect", daemon=True).start()
