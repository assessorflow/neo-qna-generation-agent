"""Unit tests for bootstrap and settings validation."""

from __future__ import annotations

import pytest
from pytest import MonkeyPatch

from qna_generation_agent.app.bootstrap import (
    _build_container,
)
from qna_generation_agent.app.settings import RuntimeEnvironment, Settings
from qna_generation_agent.errors import ConfigurationError


def set_minimal_env(monkeypatch: MonkeyPatch) -> None:
    """Set minimal required environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com")
    monkeypatch.setenv("SUBMISSION_SERVICE_URL", "grpc://localhost:50051")
    monkeypatch.setenv("KNOWLEDGE_SERVICE_URL", "grpc://localhost:50052")


@pytest.mark.unit
class TestSettingsValidation:
    """Tests for settings validation methods."""

    def test_validate_succeeds_with_full_config(self, monkeypatch: MonkeyPatch) -> None:
        """Serve bootstrap succeeds with all required configuration."""
        set_minimal_env(monkeypatch)
        monkeypatch.setenv("PUBSUB_PROJECT_ID", "test-project")
        monkeypatch.setenv("PUBSUB_SUBSCRIPTION_TRIGGER", "test-subscription")
        monkeypatch.setenv("PUBSUB_TOPIC_COMPLETE", "test-topic")

        settings = Settings()  # type: ignore[call-arg]
        # Should not raise
        settings.validate_settings()
        assert settings.worker_ready is True

    def test_validate_rejects_missing_llm_config(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        """Serve bootstrap requires LLM configuration."""
        monkeypatch.setenv("OPENAI_API_KEY", "")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com")
        monkeypatch.setenv("SUBMISSION_SERVICE_URL", "grpc://localhost:50051")
        monkeypatch.setenv("KNOWLEDGE_SERVICE_URL", "grpc://localhost:50052")
        monkeypatch.setenv("PUBSUB_PROJECT_ID", "test-project")
        monkeypatch.setenv("PUBSUB_SUBSCRIPTION_TRIGGER", "test-sub")
        monkeypatch.setenv("PUBSUB_TOPIC_COMPLETE", "test-topic")

        settings = Settings()  # type: ignore[call-arg]
        with pytest.raises(ConfigurationError):
            settings.validate_settings()

    def test_validate_rejects_missing_pubsub_config(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        """Serve bootstrap requires Pub/Sub configuration."""
        set_minimal_env(monkeypatch)
        # Missing pubsub config - explicitly clear to override .env file
        monkeypatch.setenv("PUBSUB_PROJECT_ID", "")
        monkeypatch.setenv("PUBSUB_SUBSCRIPTION_TRIGGER", "")
        monkeypatch.setenv("PUBSUB_TOPIC_COMPLETE", "")

        settings = Settings()  # type: ignore[call-arg]
        with pytest.raises(ConfigurationError):
            settings.validate_settings()


@pytest.mark.unit
class TestSettingsProperties:
    """Tests for settings computed properties."""

    def test_pubsub_enabled_when_configured(self, monkeypatch: MonkeyPatch) -> None:
        """pubsub_enabled is True when project and topic are set."""
        set_minimal_env(monkeypatch)
        monkeypatch.setenv("PUBSUB_PROJECT_ID", "test-project")
        monkeypatch.setenv("PUBSUB_TOPIC_COMPLETE", "test-topic")

        settings = Settings()  # type: ignore[call-arg]
        assert settings.pubsub_enabled is True

    def test_pubsub_enabled_defaults_to_true(self, monkeypatch: MonkeyPatch) -> None:
        """pubsub_enabled defaults to True."""
        set_minimal_env(monkeypatch)

        settings = Settings()  # type: ignore[call-arg]
        assert settings.pubsub_enabled is True

    def test_pubsub_enabled_when_explicitly_disabled(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        """pubsub_enabled is False when explicitly disabled."""
        set_minimal_env(monkeypatch)
        monkeypatch.setenv("PUBSUB_ENABLED", "false")

        settings = Settings()  # type: ignore[call-arg]
        assert settings.pubsub_enabled is False

    def test_langfuse_enabled_when_configured(self, monkeypatch: MonkeyPatch) -> None:
        """langfuse_enabled is True when keys are set."""
        set_minimal_env(monkeypatch)
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

        settings = Settings()  # type: ignore[call-arg]
        assert settings.langfuse_enabled is True

    def test_langfuse_enabled_when_not_configured(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        """langfuse_enabled is False when keys are not set."""
        set_minimal_env(monkeypatch)
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")

        settings = Settings()  # type: ignore[call-arg]
        assert settings.langfuse_enabled is False

    def test_worker_ready_when_pubsub_configured(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        """worker_ready is True when Pub/Sub subscription is configured."""
        set_minimal_env(monkeypatch)
        monkeypatch.setenv("PUBSUB_PROJECT_ID", "test-project")
        monkeypatch.setenv("PUBSUB_SUBSCRIPTION_TRIGGER", "test-subscription")
        monkeypatch.setenv("PUBSUB_TOPIC_COMPLETE", "test-topic")

        settings = Settings()  # type: ignore[call-arg]
        assert settings.worker_ready is True

    def test_worker_ready_when_pubsub_not_configured(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        """worker_ready is False when subscription is not set."""
        set_minimal_env(monkeypatch)
        monkeypatch.setenv("PUBSUB_PROJECT_ID", "")
        monkeypatch.setenv("PUBSUB_SUBSCRIPTION_TRIGGER", "")
        monkeypatch.setenv("PUBSUB_TOPIC_COMPLETE", "")

        settings = Settings()  # type: ignore[call-arg]
        assert settings.worker_ready is False

    def test_environment_defaults_to_local(self, monkeypatch: MonkeyPatch) -> None:
        """Environment defaults to local."""
        set_minimal_env(monkeypatch)

        settings = Settings()  # type: ignore[call-arg]
        assert settings.environment == RuntimeEnvironment.LOCAL

    def test_is_development_returns_true_for_local(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        """is_development returns True for local environment."""
        set_minimal_env(monkeypatch)
        monkeypatch.setenv("ENVIRONMENT", "local")

        settings = Settings()  # type: ignore[call-arg]
        assert settings.is_development is True

    def test_is_development_returns_true_for_dev(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        """is_development returns True for dev environment."""
        set_minimal_env(monkeypatch)
        monkeypatch.setenv("ENVIRONMENT", "dev")

        settings = Settings()  # type: ignore[call-arg]
        assert settings.is_development is True

    def test_is_development_returns_false_for_prod(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        """is_development returns False for prod environment."""
        set_minimal_env(monkeypatch)
        monkeypatch.setenv("ENVIRONMENT", "prod")

        settings = Settings()  # type: ignore[call-arg]
        assert settings.is_development is False

    def test_service_name_returns_canonical_name(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        """service_name returns the canonical service identity."""
        set_minimal_env(monkeypatch)

        settings = Settings()  # type: ignore[call-arg]
        assert settings.service_name == "qna-generation-agent"

    def test_swarm_size_reads_env_var(self, monkeypatch: MonkeyPatch) -> None:
        """swarm_size reads QNA_SWARM_SIZE."""
        set_minimal_env(monkeypatch)
        monkeypatch.setenv("QNA_SWARM_SIZE", "5")

        settings = Settings()  # type: ignore[call-arg]
        assert settings.swarm_size == 5


@pytest.mark.unit
class TestBootstrapContainer:
    """Tests for bootstrap container building."""

    def test_build_container_creates_llm_provider(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        """Container creates LLM provider."""
        set_minimal_env(monkeypatch)

        settings = Settings()  # type: ignore[call-arg]
        container = _build_container(settings)

        assert container.llm_provider is not None

    def test_build_container_creates_submission_client(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        """Container creates submission client."""
        set_minimal_env(monkeypatch)

        settings = Settings()  # type: ignore[call-arg]
        container = _build_container(settings)

        assert container.submission_client is not None

    def test_build_container_creates_knowledge_client(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        """Container creates knowledge client."""
        set_minimal_env(monkeypatch)

        settings = Settings()  # type: ignore[call-arg]
        container = _build_container(settings)

        assert container.knowledge_client is not None

    def test_build_container_with_null_publisher(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        """Container uses null publisher when Pub/Sub is not configured."""
        set_minimal_env(monkeypatch)
        monkeypatch.setenv("PUBSUB_PROJECT_ID", "")
        monkeypatch.setenv("PUBSUB_TOPIC_COMPLETE", "")

        settings = Settings()  # type: ignore[call-arg]
        container = _build_container(settings)

        from qna_generation_agent.infrastructure.messaging.pubsub_publisher import (
            NullEventPublisher,
        )

        assert isinstance(container.event_publisher, NullEventPublisher)

    def test_build_container_without_langfuse(self, monkeypatch: MonkeyPatch) -> None:
        """Container has no telemetry when prompt management is not configured."""
        set_minimal_env(monkeypatch)
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")

        settings = Settings()  # type: ignore[call-arg]
        container = _build_container(settings)

        assert container.telemetry is None

    def test_build_container_deprecated_workflow_mode_ignored(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        """LLM_WORKFLOW_MODE is deprecated and has no effect - always uses standard provider."""
        set_minimal_env(monkeypatch)
        monkeypatch.setenv("LLM_WORKFLOW_MODE", "true")

        settings = Settings()  # type: ignore[call-arg]
        # Should succeed and use StrandsLLMProvider (not raise ConfigurationError)
        container = _build_container(settings)
        assert container.llm_provider is not None


@pytest.mark.unit
class TestValidateHost:
    """Tests for host validation."""

    def test_validate_rejects_empty_host(self, monkeypatch: MonkeyPatch) -> None:
        """validate raises when host is empty."""
        set_minimal_env(monkeypatch)
        monkeypatch.setenv("PUBSUB_PROJECT_ID", "test-project")
        monkeypatch.setenv("PUBSUB_SUBSCRIPTION_TRIGGER", "test-sub")
        monkeypatch.setenv("PUBSUB_TOPIC_COMPLETE", "test-topic")
        monkeypatch.setenv("HOST", "   ")

        settings = Settings()  # type: ignore[call-arg]
        with pytest.raises(ConfigurationError):
            settings.validate_settings()


@pytest.mark.unit
class TestSettingsWorkerValidation:
    """Tests for strict worker validation."""

    def test_rejects_workers_greater_than_1(self, monkeypatch: MonkeyPatch) -> None:
        """Settings validation rejects workers > 1."""
        set_minimal_env(monkeypatch)
        monkeypatch.setenv("WORKERS", "4")

        with pytest.raises(ConfigurationError) as exc_info:
            Settings()  # type: ignore[call-arg]
        assert "workers must be 1" in str(exc_info.value)

    def test_rejects_pubsub_max_workers_greater_than_1(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        """Settings validation rejects pubsub_max_workers > 1."""
        set_minimal_env(monkeypatch)
        monkeypatch.setenv("PUBSUB_MAX_WORKERS", "4")

        with pytest.raises(ConfigurationError) as exc_info:
            Settings()  # type: ignore[call-arg]
        assert "pubsub_max_workers must be 1" in str(exc_info.value)

    def test_accepts_single_worker(self, monkeypatch: MonkeyPatch) -> None:
        """Settings validation accepts workers=1."""
        set_minimal_env(monkeypatch)
        monkeypatch.setenv("WORKERS", "1")
        monkeypatch.setenv("PUBSUB_MAX_WORKERS", "1")

        settings = Settings()  # type: ignore[call-arg]
        assert settings.workers == 1
        assert settings.pubsub_max_workers == 1


@pytest.mark.unit
class TestSettingsTimeoutValidation:
    """Tests for timeout bounds validation."""

    def test_rejects_llm_timeout_below_minimum(self, monkeypatch: MonkeyPatch) -> None:
        """Settings validation rejects LLM timeout below 30 seconds."""
        set_minimal_env(monkeypatch)
        monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "10")  # Below minimum of 30

        with pytest.raises(ValueError) as exc_info:
            Settings()  # type: ignore[call-arg]
        assert "LLM_TIMEOUT_SECONDS" in str(exc_info.value)

    def test_rejects_llm_timeout_above_maximum(self, monkeypatch: MonkeyPatch) -> None:
        """Settings validation rejects LLM timeout above 600 seconds."""
        set_minimal_env(monkeypatch)
        monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "700")  # Above maximum of 600

        with pytest.raises(ValueError) as exc_info:
            Settings()  # type: ignore[call-arg]
        assert "LLM_TIMEOUT_SECONDS" in str(exc_info.value)

    def test_rejects_grpc_timeout_below_minimum(self, monkeypatch: MonkeyPatch) -> None:
        """Settings validation rejects gRPC timeout below 5 seconds."""
        set_minimal_env(monkeypatch)
        monkeypatch.setenv("GRPC_TIMEOUT_SECONDS", "1")  # Below minimum of 5

        with pytest.raises(ValueError) as exc_info:
            Settings()  # type: ignore[call-arg]
        assert "GRPC_TIMEOUT_SECONDS" in str(exc_info.value)

    def test_rejects_grpc_timeout_above_maximum(self, monkeypatch: MonkeyPatch) -> None:
        """Settings validation rejects gRPC timeout above 120 seconds."""
        set_minimal_env(monkeypatch)
        monkeypatch.setenv("GRPC_TIMEOUT_SECONDS", "200")  # Above maximum of 120

        with pytest.raises(ValueError) as exc_info:
            Settings()  # type: ignore[call-arg]
        assert "GRPC_TIMEOUT_SECONDS" in str(exc_info.value)

    def test_accepts_valid_timeouts(self, monkeypatch: MonkeyPatch) -> None:
        """Settings validation accepts valid timeout values."""
        set_minimal_env(monkeypatch)
        monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "60")
        monkeypatch.setenv("GRPC_TIMEOUT_SECONDS", "30")

        settings = Settings()  # type: ignore[call-arg]
        assert settings.llm_timeout_seconds == 60
        assert settings.grpc_timeout_seconds == 30
