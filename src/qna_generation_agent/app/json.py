"""Shared orjson helpers."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any

import orjson
from pydantic import BaseModel


def _default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def dumps(value: Any) -> bytes:
    """Serialize a value to JSON bytes with orjson."""
    return orjson.dumps(value, default=_default, option=orjson.OPT_SORT_KEYS)


def loads(value: bytes | str) -> Any:
    """Deserialize JSON using orjson."""
    return orjson.loads(value)
