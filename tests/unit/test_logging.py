"""Unit tests for app/logging module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from qna_generation_agent.app.logging import (
    bind_context,
    clear_context,
    configure_logging,
    get_logger,
)


class TestConfigureLogging:
    """Tests for configure_logging."""

    @pytest.mark.unit
    def test_configures_logging(self) -> None:
        """Test that logging is configured."""
        with patch("qna_generation_agent.app.logging.logging") as mock_logging:
            with patch("qna_generation_agent.app.logging.structlog") as mock_structlog:
                configure_logging("info")

                mock_logging.basicConfig.assert_called_once()
                mock_structlog.configure.assert_called_once()

    @pytest.mark.unit
    def test_handles_invalid_level(self) -> None:
        """Test that invalid level defaults to INFO."""
        with patch("qna_generation_agent.app.logging.logging") as mock_logging:
            with patch("qna_generation_agent.app.logging.structlog"):
                configure_logging("invalid_level")

                # Should still call basicConfig (with INFO level)
                mock_logging.basicConfig.assert_called_once()


class TestGetLogger:
    """Tests for get_logger."""

    @pytest.mark.unit
    def test_returns_logger(self) -> None:
        """Test that logger is returned."""
        with patch("qna_generation_agent.app.logging.structlog") as mock_structlog:
            mock_logger = MagicMock()
            mock_structlog.get_logger.return_value = mock_logger

            result = get_logger("test_name")

            assert result is mock_logger


class TestClearContext:
    """Tests for clear_context."""

    @pytest.mark.unit
    def test_clears_context(self) -> None:
        """Test that context is cleared."""
        with patch("qna_generation_agent.app.logging.structlog") as mock_structlog:
            clear_context()

            mock_structlog.contextvars.clear_contextvars.assert_called_once()


class TestBindContext:
    """Tests for bind_context."""

    @pytest.mark.unit
    def test_binds_values(self) -> None:
        """Test that values are bound to context."""
        with patch("qna_generation_agent.app.logging.structlog") as mock_structlog:
            bind_context(request_id="test-123", trace_id="trace-456")

            mock_structlog.contextvars.bind_contextvars.assert_called_once()

    @pytest.mark.unit
    def test_filters_none_values(self) -> None:
        """Test that None values are filtered out."""
        with patch("qna_generation_agent.app.logging.structlog") as mock_structlog:
            bind_context(request_id="test-123", trace_id=None)

            # Should only bind non-None values
            call_args = mock_structlog.contextvars.bind_contextvars.call_args
            assert "request_id" in call_args.kwargs
            assert "trace_id" not in call_args.kwargs
