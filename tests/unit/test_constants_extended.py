"""Unit tests for app/constants module."""

from __future__ import annotations

import pytest

from qna_generation_agent.app.constants import (
    HTTP_DEFAULT_PORT,
    HTTP_DEFAULT_WORKERS,
    MAX_QUESTION_TEXT_LENGTH,
    SERVICE_NAME,
)


class TestConstants:
    """Tests for constants."""

    @pytest.mark.unit
    def test_default_port(self) -> None:
        """Test default port value."""
        assert HTTP_DEFAULT_PORT == 8000

    @pytest.mark.unit
    def test_default_workers(self) -> None:
        """Test default workers value."""
        assert HTTP_DEFAULT_WORKERS == 1

    @pytest.mark.unit
    def test_max_question_text_length(self) -> None:
        """Test max question text length value."""
        assert MAX_QUESTION_TEXT_LENGTH == 10000

    @pytest.mark.unit
    def test_service_name(self) -> None:
        """Test service name value."""
        assert SERVICE_NAME == "qna-generation-agent"
