"""Central logging configuration helpers."""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


DEFAULT_LOGGER_NAME = "automl_gui"


def configure_logging(
    *,
    debug: bool = False,
    log_to_file: bool = False,
    logger_name: str = DEFAULT_LOGGER_NAME,
) -> logging.Logger:
    """Configure app logger with console and optional rotating-file handlers."""
    env_level = os.getenv("AUTOML_LOG_LEVEL", "").upper()
    if env_level in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        level = getattr(logging, env_level)
    else:
        level = logging.DEBUG if debug else logging.INFO

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    has_stream_handler = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    if not has_stream_handler:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    file_logging_enabled = log_to_file or os.getenv("AUTOML_LOG_TO_FILE", "0") == "1"
    if file_logging_enabled:
        log_path = Path(os.getenv("AUTOML_LOG_FILE", "automl_gui.log"))
        has_file_handler = any(isinstance(h, RotatingFileHandler) for h in logger.handlers)
        if not has_file_handler:
            file_handler = RotatingFileHandler(
                filename=log_path,
                maxBytes=5 * 1024 * 1024,
                backupCount=3,
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    for handler in logger.handlers:
        handler.setLevel(level)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get child logger under app namespace."""
    return logging.getLogger(f"{DEFAULT_LOGGER_NAME}.{name}")
