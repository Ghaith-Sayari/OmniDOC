"""
Shared utilities — structured logging.
"""

from __future__ import annotations

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger for *name* with a consistent format.

    Format: ``YYYY-MM-DD HH:MM:SS | LEVEL     | module:line — message``
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
