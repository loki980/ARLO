"""
Logging configuration for ARLO.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Milliseconds follow the seconds so lines within one second can be ordered
CONSOLE_FORMAT = "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s"
CONSOLE_DATEFMT = "%H:%M:%S"
FILE_FORMAT = "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s - %(message)s"
FILE_DATEFMT = "%Y-%m-%d %H:%M:%S"


def console_formatter() -> logging.Formatter:
    """Formatter for the console handler, for example 19:42:07.381."""
    return logging.Formatter(CONSOLE_FORMAT, datefmt=CONSOLE_DATEFMT)


def file_formatter() -> logging.Formatter:
    """Formatter for the file handler, for example 2026-10-14 19:42:07.381."""
    return logging.Formatter(FILE_FORMAT, datefmt=FILE_DATEFMT)


def setup_logging(app_dir: Path, *, debug_mode: bool = False) -> logging.Logger:
    """
    Configure and return the application logger.

    Call this once at application startup.
    """
    logger = logging.getLogger("arc_helper")

    if debug_mode:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(console_formatter())
    logger.addHandler(console_handler)

    # File handler (only in debug mode)
    if debug_mode:
        log_file = app_dir / "arc_helper.log"
        file_handler = RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=2, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter())
        logger.addHandler(file_handler)

    return logger
