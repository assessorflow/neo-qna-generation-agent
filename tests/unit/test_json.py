"""Unit tests for app/json module."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum

import pytest
from pydantic import BaseModel

from qna_generation_agent.app.json import _default, dumps, loads


class TestDefault:
    """Tests for _default function."""

    @pytest.mark.unit
    def test_handles_pydantic_model(self) -> None:
        """Test that Pydantic models are serialized."""

        class TestModel(BaseModel):
            name: str
            value: int

        model = TestModel(name="test", value=42)
        result = _default(model)

        assert result == {"name": "test", "value": 42}

    @pytest.mark.unit
    def test_handles_dataclass(self) -> None:
        """Test that dataclasses are serialized."""

        @dataclass
        class TestData:
            name: str
            value: int

        data = TestData(name="test", value=42)
        result = _default(data)

        assert result == {"name": "test", "value": 42}

    @pytest.mark.unit
    def test_handles_enum(self) -> None:
        """Test that enums are serialized."""

        class TestEnum(Enum):
            VALUE = "test_value"

        enum_val = TestEnum.VALUE
        result = _default(enum_val)

        assert result == "test_value"

    @pytest.mark.unit
    def test_handles_datetime(self) -> None:
        """Test that datetime is serialized."""
        dt = datetime(2025, 1, 1, 12, 0, 0)
        result = _default(dt)

        assert result == "2025-01-01T12:00:00"

    @pytest.mark.unit
    def test_handles_date(self) -> None:
        """Test that date is serialized."""
        d = date(2025, 1, 1)
        result = _default(d)

        assert result == "2025-01-01"

    @pytest.mark.unit
    def test_raises_on_unsupported_type(self) -> None:
        """Test that unsupported types raise TypeError."""
        with pytest.raises(TypeError):
            _default(object())


class TestDumps:
    """Tests for dumps function."""

    @pytest.mark.unit
    def test_serializes_dict(self) -> None:
        """Test that dict is serialized to JSON bytes."""
        data = {"key": "value"}
        result = dumps(data)

        assert isinstance(result, bytes)
        assert b"key" in result
        assert b"value" in result

    @pytest.mark.unit
    def test_serializes_pydantic_model(self) -> None:
        """Test that Pydantic model is serialized."""

        class TestModel(BaseModel):
            name: str
            value: int

        model = TestModel(name="test", value=42)
        result = dumps(model)

        assert isinstance(result, bytes)
        assert b"test" in result
        assert b"42" in result


class TestLoads:
    """Tests for loads function."""

    @pytest.mark.unit
    def test_deserializes_bytes(self) -> None:
        """Test that bytes are deserialized."""
        data = b'{"key": "value"}'
        result = loads(data)

        assert result == {"key": "value"}

    @pytest.mark.unit
    def test_deserializes_string(self) -> None:
        """Test that string is deserialized."""
        data = '{"key": "value"}'
        result = loads(data)

        assert result == {"key": "value"}
