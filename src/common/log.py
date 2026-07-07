"""Shared logging configuration (CLAUDE.md §7).

Console + rotating file handler under the log directory. No bare ``print``
in library code — every module gets its logger via :func:`get_logger`.
"""
import logging
import logging.handlers
import os
from pathlib import Path

LOG_DIR_ENV = "CURVE_EXTRACTOR_LOG_DIR"
DEFAULT_LOG_DIR = "logs"
LOG_FILE_NAME = "pipeline.log"
LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
MAX_LOG_BYTES = 1_000_000
LOG_BACKUP_COUNT = 3

_configured = False


def _configure_root() -> None:
    """Attach console + rotating-file handlers to the package root logger once."""
    global _configured
    if _configured:
        return
    root = logging.getLogger("src")
    root.setLevel(logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    log_dir = Path(os.environ.get(LOG_DIR_ENV, DEFAULT_LOG_DIR))
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / LOG_FILE_NAME,
        maxBytes=MAX_LOG_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger wired to the shared console + rotating-file config.

    Args:
        name: Module name, normally ``__name__``.
    """
    _configure_root()
    return logging.getLogger(name)
