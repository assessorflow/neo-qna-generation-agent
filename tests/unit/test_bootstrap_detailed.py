"""Unit tests for bootstrap module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qna_generation_agent.app.bootstrap import (
    ApplicationContainer,
    _build_knowledge_client,
    _build_llm,
    _build_prompt_provider,
    _build_submission_client,
    _build_subscriber,
    _build_telemetry,
    bootstrap_serve,
)
from qna_generation_agent.app.settings import RuntimeEnvironment
from qna_generation_agent.errors import ConfigurationError


class TestBuildSubmissionClient:
    """Tests for _build_submission_client."""

    @pytest.mark.unit
    def test_creates_client(self) -> None:
        """Test that client is created."""
        settings = MagicMock()
        settings.submission_service_url = "localhost:50051"
        settings.grpc_tls_enabled = False
        settings.grpc_tls_cert_path = None

        with patch(
            "qna_generation_agent.app.bootstrap.GrpcSubmissionClient"
        ) as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            result = _build_submission_client(settings)

            assert result is mock_instance


class TestBuildKnowledgeClient:
    """Tests for _build_knowledge_client."""

    @pytest.mark.unit
    def test_creates_client(self) -> None:
        """Test that client is created."""
        settings = MagicMock()
        settings.knowledge_service_url = "localhost:50052"
        settings.grpc_tls_enabled = False
        settings.grpc_tls_cert_path = None

        with patch(
            "qna_generation_agent.app.bootstrap.GrpcKnowledgeClient"
        ) as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            result = _build_knowledge_client(settings)

            assert result is mock_instance


class TestBuildTelemetry:
    """Tests for _build_telemetry."""

    @pytest.mark.unit
    def test_returns_none_if_disabled(self) -> None:
        """Test that None is returned if Langfuse is disabled."""
        settings = MagicMock()
        settings.langfuse_enabled = False

        result = _build_telemetry(settings)

        assert result is None

    @pytest.mark.unit
    def test_creates_telemetry(self) -> None:
        """Test that telemetry is created."""
        settings = MagicMock()
        settings.langfuse_enabled = True
        settings.langfuse_public_key = "test_public"
        settings.langfuse_secret_key = "test_secret"
        settings.langfuse_base_url = "https://test.langfuse.com"
        settings.environment = RuntimeEnvironment.LOCAL
        settings.release = "v1.0.0"

        with patch(
            "qna_generation_agent.app.bootstrap.LangfuseTelemetry"
        ) as mock_telemetry:
            mock_instance = MagicMock()
            mock_telemetry.return_value = mock_instance

            result = _build_telemetry(settings)

            assert result is mock_instance


class TestBuildLlm:
    """Tests for _build_llm."""

    @pytest.mark.unit
    def test_creates_llm_provider(self) -> None:
        """Test that LLM provider is created."""
        settings = MagicMock()
        settings.model_id = "gpt-4o"
        settings.llm_api_key = "test-key"
        settings.llm_base_url = "https://api.openai.com"
        settings.llm_workflow_mode = False

        with patch(
            "qna_generation_agent.app.bootstrap.StrandsLLMProvider"
        ) as mock_provider:
            mock_instance = MagicMock()
            mock_provider.return_value = mock_instance

            result = _build_llm(settings, "gpt-4o")

            assert result is mock_instance


class TestBuildPromptProvider:
    """Tests for _build_prompt_provider."""

    @pytest.mark.unit
    def test_returns_none_if_disabled(self) -> None:
        """Test that None is returned if Langfuse is disabled."""
        settings = MagicMock()
        settings.langfuse_enabled = False

        result = _build_prompt_provider(settings)

        assert result is None

    @pytest.mark.unit
    def test_creates_provider(self) -> None:
        """Test that provider is created."""
        settings = MagicMock()
        settings.langfuse_enabled = True
        settings.langfuse_public_key = "test_public"
        settings.langfuse_secret_key = "test_secret"
        settings.langfuse_base_url = "https://test.langfuse.com"
        settings.environment = RuntimeEnvironment.LOCAL
        settings.release = "v1.0.0"

        with patch(
            "qna_generation_agent.app.bootstrap.LangfusePromptProvider"
        ) as mock_provider:
            mock_instance = MagicMock()
            mock_provider.return_value = mock_instance

            result = _build_prompt_provider(settings)

            assert result is mock_instance


class TestBuildSubscriber:
    """Tests for _build_subscriber."""

    @pytest.mark.unit
    def test_returns_none_if_pubsub_disabled(self) -> None:
        """Test that None is returned if Pub/Sub is disabled."""
        settings = MagicMock()
        settings.pubsub_enabled = False

        generate_service = MagicMock()

        result = _build_subscriber(settings, generate_service)

        assert result is None

    @pytest.mark.unit
    def test_returns_none_if_worker_not_ready(self) -> None:
        """Test that None is returned if worker is not ready."""
        settings = MagicMock()
        settings.pubsub_enabled = True
        settings.worker_ready = False

        generate_service = MagicMock()

        result = _build_subscriber(settings, generate_service)

        assert result is None


class TestApplicationContainer:
    """Tests for ApplicationContainer."""

    @pytest.mark.unit
    async def test_startup_skips_subscriber_if_none(self) -> None:
        """Test that startup works without subscriber."""
        settings = MagicMock()

        container = ApplicationContainer(
            settings=settings,
            llm_provider=None,
            question_set_repo=MagicMock(),
            idempotency_store=MagicMock(),
            event_publisher=MagicMock(),
            telemetry=None,
            subscriber=None,
            submission_client=MagicMock(),
            knowledge_client=MagicMock(),
        )

        await container.startup()

    @pytest.mark.unit
    async def test_startup_starts_subscriber(self) -> None:
        """Test that startup starts subscriber if present."""
        settings = MagicMock()
        subscriber = MagicMock()
        subscriber.start = AsyncMock()

        container = ApplicationContainer(
            settings=settings,
            llm_provider=None,
            question_set_repo=MagicMock(),
            idempotency_store=MagicMock(),
            event_publisher=MagicMock(),
            telemetry=None,
            subscriber=subscriber,
            submission_client=MagicMock(),
            knowledge_client=MagicMock(),
        )

        await container.startup()

        subscriber.start.assert_awaited_once()

    @pytest.mark.unit
    async def test_shutdown_closes_clients(self) -> None:
        """Test that shutdown closes all clients."""
        settings = MagicMock()
        subscriber = MagicMock()
        subscriber.shutdown = AsyncMock()
        telemetry = MagicMock()
        telemetry.shutdown = AsyncMock()
        submission_client = MagicMock()
        submission_client.close = AsyncMock()
        knowledge_client = MagicMock()
        knowledge_client.close = AsyncMock()
        event_publisher = MagicMock()
        event_publisher.close = AsyncMock()
        prompt_provider = MagicMock()
        prompt_provider.shutdown = AsyncMock()

        container = ApplicationContainer(
            settings=settings,
            llm_provider=None,
            question_set_repo=MagicMock(),
            idempotency_store=MagicMock(),
            event_publisher=event_publisher,
            telemetry=telemetry,
            subscriber=subscriber,
            submission_client=submission_client,
            knowledge_client=knowledge_client,
            prompt_provider=prompt_provider,
        )

        await container.shutdown()

        subscriber.shutdown.assert_awaited_once()
        telemetry.shutdown.assert_awaited_once()
        submission_client.close.assert_awaited_once()
        knowledge_client.close.assert_awaited_once()
        event_publisher.close.assert_awaited_once()
        prompt_provider.shutdown.assert_awaited_once()

    @pytest.mark.unit
    async def test_shutdown_handles_errors_gracefully(self) -> None:
        """Test that shutdown handles errors gracefully."""
        settings = MagicMock()
        subscriber = MagicMock()
        subscriber.shutdown = AsyncMock(side_effect=Exception("Shutdown error"))

        container = ApplicationContainer(
            settings=settings,
            llm_provider=None,
            question_set_repo=MagicMock(),
            idempotency_store=MagicMock(),
            event_publisher=MagicMock(),
            telemetry=None,
            subscriber=subscriber,
            submission_client=MagicMock(),
            knowledge_client=MagicMock(),
        )

        # Should not raise
        await container.shutdown()


class TestBuildSubscriberBranchSelection:
    """Tests for _build_subscriber branch selection logic."""

    @pytest.mark.unit
    def test_returns_none_when_pubsub_disabled(self) -> None:
        """Test that subscriber is None when pubsub is disabled."""
        settings = MagicMock()
        settings.pubsub_enabled = False
        settings.worker_ready = True

        generate_service = MagicMock()

        result = _build_subscriber(settings, generate_service)

        assert result is None

    @pytest.mark.unit
    def test_returns_none_when_worker_not_ready(self) -> None:
        """Test that subscriber is None when worker is not ready."""
        settings = MagicMock()
        settings.pubsub_enabled = True
        settings.worker_ready = False

        generate_service = MagicMock()

        result = _build_subscriber(settings, generate_service)

        assert result is None

    @pytest.mark.unit
    def test_returns_worker_when_enabled_and_ready(self) -> None:
        """Test that subscriber is created when enabled and ready."""
        settings = MagicMock()
        settings.pubsub_enabled = True
        settings.worker_ready = True
        settings.pubsub_project_id = "test-project"
        settings.pubsub_subscription_trigger = "test-subscription"
        settings.pubsub_max_workers = 1

        generate_service = MagicMock()

        with patch(
            "qna_generation_agent.app.bootstrap.PubSubSubscriptionWorker"
        ) as mock_worker_class:
            mock_instance = MagicMock()
            mock_worker_class.return_value = mock_instance

            result = _build_subscriber(settings, generate_service)

            assert result is mock_instance
            # Verify config values were passed correctly
            mock_worker_class.assert_called_once()
            call_args = mock_worker_class.call_args
            config = call_args[0][0]
            assert config.project_id == "test-project"
            assert config.subscription_id == "test-subscription"
            assert config.max_messages == 1


class TestApplicationContainerShutdown:
    """Tests for container shutdown continuation behavior."""

    @pytest.mark.unit
    async def test_shutdown_continues_after_individual_failures(self) -> None:
        """Test that shutdown continues even if individual clients fail."""
        settings = MagicMock()

        # Create clients that will fail during shutdown
        failing_subscriber = MagicMock()
        failing_subscriber.shutdown = AsyncMock(
            side_effect=RuntimeError("Subscriber failed")
        )

        failing_telemetry = MagicMock()
        failing_telemetry.shutdown = AsyncMock(
            side_effect=RuntimeError("Telemetry failed")
        )

        # Other clients that should still be called
        submission_client = MagicMock()
        submission_client.close = AsyncMock()

        knowledge_client = MagicMock()
        knowledge_client.close = AsyncMock()

        event_publisher = MagicMock()
        event_publisher.close = AsyncMock()

        container = ApplicationContainer(
            settings=settings,
            llm_provider=None,
            question_set_repo=MagicMock(),
            idempotency_store=MagicMock(),
            event_publisher=event_publisher,
            telemetry=failing_telemetry,
            subscriber=failing_subscriber,
            submission_client=submission_client,
            knowledge_client=knowledge_client,
        )

        # Should not raise - should continue after each failure
        await container.shutdown()

        # All close methods should have been called despite failures
        failing_subscriber.shutdown.assert_awaited_once()
        failing_telemetry.shutdown.assert_awaited_once()
        submission_client.close.assert_awaited_once()
        knowledge_client.close.assert_awaited_once()
        event_publisher.close.assert_awaited_once()


class TestBootstrapServe:
    """Tests for bootstrap_serve."""

    @pytest.mark.unit
    def test_validates_settings(self) -> None:
        """Test that settings are validated."""
        settings = MagicMock()
        settings.validate_settings = MagicMock()

        with patch("qna_generation_agent.app.bootstrap._build_container") as mock_build:
            mock_container = MagicMock()
            mock_container.llm_provider = MagicMock()
            mock_container.subscriber = MagicMock()
            mock_build.return_value = mock_container

            bootstrap_serve(settings)

        settings.validate_settings.assert_called_once()

    @pytest.mark.unit
    def test_raises_if_no_llm_provider(self) -> None:
        """Test that error is raised if no LLM provider."""
        settings = MagicMock()
        settings.validate_settings = MagicMock()

        with patch("qna_generation_agent.app.bootstrap._build_container") as mock_build:
            mock_container = MagicMock()
            mock_container.llm_provider = None
            mock_build.return_value = mock_container

            with pytest.raises(ConfigurationError):
                bootstrap_serve(settings)

    @pytest.mark.unit
    def test_raises_if_no_subscriber(self) -> None:
        """Test that error is raised if no subscriber."""
        settings = MagicMock()
        settings.validate_settings = MagicMock()

        with patch("qna_generation_agent.app.bootstrap._build_container") as mock_build:
            mock_container = MagicMock()
            mock_container.llm_provider = MagicMock()
            mock_container.subscriber = None
            mock_build.return_value = mock_container

            with pytest.raises(ConfigurationError):
                bootstrap_serve(settings)
