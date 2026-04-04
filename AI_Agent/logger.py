"""
logger.py
─────────
Centralised logger using `rich` for coloured terminal output
and plain-text file logging under ./logs/
"""

import logging
import os
from datetime import datetime
from rich.logging import RichHandler
from config import LOG_DIR

os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, f"agent_{datetime.now().strftime('%Y%m%d')}.log")

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(message)s",
    datefmt = "[%H:%M:%S]",
    handlers= [
        RichHandler(rich_tracebacks=True, show_path=False),          # coloured terminal
        logging.FileHandler(log_file, encoding="utf-8")              # plain file
    ]
)

log = logging.getLogger("agent")