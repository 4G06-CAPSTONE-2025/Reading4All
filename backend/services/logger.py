"""
Logging Module M7 for Reading4All - Bridging Gaps AI for Diagram Accessibility
============================================================================

**Module Guide (MG) Description** [MG-3.pdf]:
- **Secrets**: The internal details of which events and errors are logged and the specific
  format used for log entries.
- **Services**: Records system
"""


# services/logger.py - Logging Module M7

import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import List

# Exported Constants (from MIS)
LOG_FILE_PATH: str = "./logs/app.log"  # configurable via env var later
LOG_TYPE: List[str] = ["Event", "Error"]

class LogWriteException(Exception):
    """Raised when log file cannot be written (from MIS semantics)."""
    pass

# One-time setup (internal secret)
_logger = logging.getLogger("reading4all")
_logger.setLevel(logging.INFO)

# Ensure logs dir exists
import os
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

_handler = RotatingFileHandler(
    LOG_FILE_PATH,
    maxBytes=5 * 1024 * 1024,  # 5MB per file
    backupCount=3
)
_formatter = logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%SZ'
)
_handler.setFormatter(_formatter)
_logger.addHandler(_handler)

def logEvent(eventMssg: str, logType: str) -> None:
    """
    Exported Access Program (from MIS).
    Appends timestamp + type + message to LOG_FILE_PATH.
    Raises LogWriteException on failure.
    """
    if logType not in LOG_TYPE:
        raise LogWriteException(f"Invalid logType: {logType}")
    
    try:
        msg = f"type={logType} | {eventMssg}"
        if logType == "Error":
            _logger.error(msg)
        else:
            _logger.info(msg)
    except Exception as e:
        raise LogWriteException(f"Failed to write log: {e}") from e
