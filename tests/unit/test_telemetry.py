"""Unit tests for Langfuse telemetry client."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from qna_generation_agent.infrastructure.telemetry.langfuse_client import (
    LangfuseTelemetry,
    _LangfuseSpan,
)


class TestLangfuseSpan:
    """Tests for _LangfuseSpan."""

    @pytest.mark.unit
    def test_set_attribute_updates_metadata(self) -> None:
        """Test that set_attribute updates span metadata."""
        mock_client = MagicMock()
        span = _LangfuseSpan(mock_client)

        span.set_attribute("key1", "value1")

        mock_client.update_current_span.assert_called_with(metadata={"key1": "value1"})

    @pytest.mark.unit
    def test_record_error_updates_span(self) -> None:
        """Test that record_error records error metadata."""
        mock_client = MagicMock()
        span = _LangfuseSpan(mock_client)
        error = ValueError("Test error")

        span.record_error(error)

        mock_client.update_current_span.assert_called_with(
            metadata={"error_type": "ValueError"},
            status_message="Test error",
        )


class TestLangfuseTelemetry:
    """Tests for LangfuseTelemetry."""

    @pytest.mark.unit
    def test_init_creates_client(self) -> None:
        """Test that initialization creates the Langfuse client."""
        with patch(
            "qna_generation_agent.infrastructure.telemetry.langfuse_client.Langfuse"
        ) as mock_langfuse:
            mock_instance = MagicMock()
            mock_langfuse.return_value = mock_instance

            telemetry = LangfuseTelemetry(
                public_key="test_public_key",
                secret_key="test_secret_key",
                host="https://test.langfuse.com",
                environment="test",
                release="v1.0.0",
            )

            assert telemetry is not None
            mock_langfuse.assert_called_once_with(
                public_key="test_public_key",
                secret_key="test_secret_key",
                host="https://test.langfuse.com",
                environment="test",
                release="v1.0.0",
            )

    @pytest.mark.unit
    def test_trace_creates_span(self) -> None:
        """Test that trace creates a span context manager."""
        with patch(
            "qna_generation_agent.infrastructure.telemetry.langfuse_client.Langfuse"
        ) as mock_langfuse:
            mock_instance = MagicMock()
            mock_langfuse.return_value = mock_instance

            telemetry = LangfuseTelemetry(
                public_key="test_public_key",
                secret_key="test_secret_key",
                host="https://test.langfuse.com",
                environment="test",
                release="v1.0.0",
            )

            # Mock the context manager
            @contextmanager
            def mock_context(*args: Any, **kwargs: Any) -> Generator[None]:
                yield

            mock_instance.start_as_current_observation.return_value = mock_context()
            mock_instance.get_current_trace_id.return_value = "trace_123"

            with telemetry.trace("test_trace", metadata={"key": "value"}) as span:
                assert span is not None

    @pytest.mark.unit
    def test_trace_binds_trace_id(self) -> None:
        """Test that trace binds trace_id to structlog context."""
        with patch(
            "qna_generation_agent.infrastructure.telemetry.langfuse_client.Langfuse"
        ) as mock_langfuse:
            with patch(
                "qna_generation_agent.infrastructure.telemetry.langfuse_client.bind_context"
            ) as mock_bind:
                mock_instance = MagicMock()
                mock_langfuse.return_value = mock_instance

                telemetry = LangfuseTelemetry(
                    public_key="test_public_key",
                    secret_key="test_secret_key",
                    host="https://test.langfuse.com",
                    environment="test",
                    release="v1.0.0",
                )

                @contextmanager
                def mock_context(*args: Any, **kwargs: Any) -> Generator[None]:
                    yield

                mock_instance.start_as_current_observation.return_value = mock_context()
                mock_instance.get_current_trace_id.return_value = "trace_123"

                with telemetry.trace("test_trace"):
                    pass

                mock_bind.assert_called_with(trace_id="trace_123")

    @pytest.mark.unit
    def test_propagate_creates_trace_context(self) -> None:
        """Test that propagate creates a trace context."""
        with patch(
            "qna_generation_agent.infrastructure.telemetry.langfuse_client.Langfuse"
        ) as mock_langfuse:
            with patch(
                "qna_generation_agent.infrastructure.telemetry.langfuse_client.propagate_attributes"
            ) as mock_prop:
                with patch(
                    "qna_generation_agent.infrastructure.telemetry.langfuse_client.bind_context"
                ) as mock_bind:
                    mock_instance = MagicMock()
                    mock_langfuse.return_value = mock_instance

                    telemetry = LangfuseTelemetry(
                        public_key="test_public_key",
                        secret_key="test_secret_key",
                        host="https://test.langfuse.com",
                        environment="test",
                        release="v1.0.0",
                    )

                    @contextmanager
                    def mock_attrs(*args: Any, **kwargs: Any) -> Generator[None]:
                        yield

                    @contextmanager
                    def mock_context(*args: Any, **kwargs: Any) -> Generator[None]:
                        yield

                    mock_prop.return_value = mock_attrs()
                    mock_instance.start_as_current_observation.return_value = (
                        mock_context()
                    )
                    mock_instance.get_current_trace_id.return_value = "trace_123"

                    with telemetry.propagate(
                        trace_name="test_trace",
                        correlation_id="corr_123",
                        workflow_id="wf_123",
                        metadata={"extra": "data"},
                    ):
                        pass

                    mock_prop.assert_called_once_with(
                        trace_name="test_trace",
                        session_id="corr_123",
                        metadata={"workflow_id": "wf_123", "extra": "data"},
                    )
                    mock_bind.assert_called_with(trace_id="trace_123")

    @pytest.mark.unit
    def test_current_trace_id_returns_value(self) -> None:
        """Test that current_trace_id returns the trace ID."""
        with patch(
            "qna_generation_agent.infrastructure.telemetry.langfuse_client.Langfuse"
        ) as mock_langfuse:
            mock_instance = MagicMock()
            mock_langfuse.return_value = mock_instance

            telemetry = LangfuseTelemetry(
                public_key="test_public_key",
                secret_key="test_secret_key",
                host="https://test.langfuse.com",
                environment="test",
                release="v1.0.0",
            )

            mock_instance.get_current_trace_id.return_value = "trace_123"

            trace_id = telemetry.current_trace_id()

            assert trace_id == "trace_123"

    @pytest.mark.unit
    def test_current_trace_url_returns_value(self) -> None:
        """Test that current_trace_url returns the trace URL."""
        with patch(
            "qna_generation_agent.infrastructure.telemetry.langfuse_client.Langfuse"
        ) as mock_langfuse:
            mock_instance = MagicMock()
            mock_langfuse.return_value = mock_instance

            telemetry = LangfuseTelemetry(
                public_key="test_public_key",
                secret_key="test_secret_key",
                host="https://test.langfuse.com",
                environment="test",
                release="v1.0.0",
            )

            mock_instance.get_trace_url.return_value = (
                "https://test.langfuse.com/trace/trace_123"
            )

            trace_url = telemetry.current_trace_url()

            assert trace_url == "https://test.langfuse.com/trace/trace_123"

    @pytest.mark.unit
    async def test_shutdown_flushes_client(self) -> None:
        """Test that shutdown flushes the client."""
        with patch(
            "qna_generation_agent.infrastructure.telemetry.langfuse_client.Langfuse"
        ) as mock_langfuse:
            with patch(
                "qna_generation_agent.infrastructure.telemetry.langfuse_client.asyncio.to_thread"
            ) as mock_to_thread:
                mock_instance = MagicMock()
                mock_langfuse.return_value = mock_instance

                telemetry = LangfuseTelemetry(
                    public_key="test_public_key",
                    secret_key="test_secret_key",
                    host="https://test.langfuse.com",
                    environment="test",
                    release="v1.0.0",
                )

                mock_to_thread.return_value = None

                await telemetry.shutdown()

                mock_to_thread.assert_any_call(mock_instance.flush)

    @pytest.mark.unit
    async def test_shutdown_handles_network_errors(self) -> None:
        """Test that shutdown handles network errors gracefully."""
        with patch(
            "qna_generation_agent.infrastructure.telemetry.langfuse_client.Langfuse"
        ) as mock_langfuse:
            with patch(
                "qna_generation_agent.infrastructure.telemetry.langfuse_client.asyncio.to_thread"
            ) as mock_to_thread:
                mock_instance = MagicMock()
                mock_langfuse.return_value = mock_instance

                telemetry = LangfuseTelemetry(
                    public_key="test_public_key",
                    secret_key="test_secret_key",
                    host="https://test.langfuse.com",
                    environment="test",
                    release="v1.0.0",
                )

                mock_to_thread.side_effect = [ConnectionError("Network error"), None]

                # Should not raise
                await telemetry.shutdown()

    @pytest.mark.unit
    async def test_shutdown_handles_other_errors(self) -> None:
        """Test that shutdown handles other errors gracefully."""
        with patch(
            "qna_generation_agent.infrastructure.telemetry.langfuse_client.Langfuse"
        ) as mock_langfuse:
            with patch(
                "qna_generation_agent.infrastructure.telemetry.langfuse_client.asyncio.to_thread"
            ) as mock_to_thread:
                mock_instance = MagicMock()
                mock_langfuse.return_value = mock_instance

                telemetry = LangfuseTelemetry(
                    public_key="test_public_key",
                    secret_key="test_secret_key",
                    host="https://test.langfuse.com",
                    environment="test",
                    release="v1.0.0",
                )

                mock_to_thread.side_effect = [RuntimeError("Unexpected error"), None]

                # Should not raise
                await telemetry.shutdown()
