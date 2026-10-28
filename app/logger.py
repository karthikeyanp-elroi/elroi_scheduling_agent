import faulthandler
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOG_DIR = Path("logs")
FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Named logger under `meetpilot`. Console shows INFO and up;
    logs/agent.log keeps everything (DEBUG includes email bodies and raw LLM output)."""
    root = logging.getLogger("meetpilot")

    if not root.handlers:
        root.setLevel(logging.DEBUG)
        root.propagate = False

        formatter = logging.Formatter(FORMAT)

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        root.addHandler(console)

        LOG_DIR.mkdir(exist_ok=True)

        # Native crashes and hangs never reach logging; this dumps the
        # Python stack of every thread to logs/crash.log instead.
        faulthandler.enable(open(LOG_DIR / "crash.log", "a"))
        file = RotatingFileHandler(
            LOG_DIR / "agent.log",
            maxBytes=2_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file.setLevel(logging.DEBUG)
        file.setFormatter(formatter)
        root.addHandler(file)

    return root.getChild(name)
