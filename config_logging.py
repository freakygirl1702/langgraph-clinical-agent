"""Central logging configuration for the medical chatbot."""

import logging
import sys
from pathlib import Path

# Format: timestamp, level, module, message
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger; call once at app startup (e.g. in app.py)."""
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(level)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(handler)
