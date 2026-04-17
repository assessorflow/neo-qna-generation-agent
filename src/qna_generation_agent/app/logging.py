"""Structured logging helpers."""

from __future__ import annotations

# Stdlib logging is imported for structlog's stdlib integration.
# structlog uses stdlib's LoggerFactory and requires basicConfig setup.
# This is the correct pattern for structlog>=25.5 stdlib compatibility.
import logging
import sys
from typing import Any, cast

import orjson
import structlog


def _render_json(value: Any, **_: Any) -> str:
    """Render logs using orjson.

    Note: orjson>=3.11 dumps() returns bytes, which must be decoded to str
    for structlog's JSONRenderer. This is the correct pattern per orjson>=3.11.
    """
    return orjson.dumps(value, option=orjson.OPT_SORT_KEYS).decode("utf-8")


def configure_logging(level_name: str) -> None:
    """Configure stdlib logging and structlog."""
    level = getattr(logging, level_name.upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        level=level,
        stream=sys.stdout,
        force=True,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(serializer=_render_json),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a typed structlog logger."""
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))


def clear_context() -> None:
    """Clear bound context variables."""
    structlog.contextvars.clear_contextvars()


def bind_context(**values: Any) -> None:
    """Bind request or worker context."""
    clean_values = {key: value for key, value in values.items() if value is not None}
    structlog.contextvars.bind_contextvars(**clean_values)
