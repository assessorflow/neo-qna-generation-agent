"""Typed runtime settings."""

from __future__ import annotations

from enum import StrEnum
from functools import cached_property
from typing import Any

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from qna_generation_agent.errors import ConfigurationError


class RuntimeEnvironment(StrEnum):
    """Runtime environment values."""

    LOCAL = "local"
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


class LogLevel(StrEnum):
    """Supported log levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Settings(BaseSettings):
    """Typed environment-backed settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        strict=True,
    )

    environment: RuntimeEnvironment = Field(
        default=RuntimeEnvironment.LOCAL,
        validation_alias=AliasChoices("ENVIRONMENT", "QNA_ENVIRONMENT"),
    )
    host: str = Field(
        default="0.0.0.0", validation_alias=AliasChoices("HOST", "QNA_HOST")
    )
    port: int = Field(
        default=8000, ge=1, le=65535, validation_alias=AliasChoices("PORT", "QNA_PORT")
    )
    # Workers are strictly limited to 1 for both HTTP (Granian) and Pub/Sub
    # to ensure single-threaded operation and prevent concurrency issues.
    # Validation happens in _enforce_single_worker model_validator.
    workers: int = Field(default=1)
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        validation_alias=AliasChoices("LOG_LEVEL", "QNA_LOG_LEVEL"),
    )

    # Unified serve mode: HTTP is always enabled (required for health checks).
    # Pub/Sub is optional and can be disabled for local development without GCP.
    pubsub_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("PUBSUB_ENABLED", "QNA_PUBSUB_ENABLED"),
    )
    pubsub_project_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("PUBSUB_PROJECT_ID", "QNA_PUBSUB_PROJECT_ID"),
    )
    pubsub_subscription_trigger: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "PUBSUB_SUBSCRIPTION_TRIGGER",
            "QNA_PUBSUB_SUBSCRIPTION_TRIGGER",
        ),
    )
    pubsub_topic_complete: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "PUBSUB_TOPIC_COMPLETE", "QNA_PUBSUB_TOPIC_COMPLETE"
        ),
    )
    pubsub_topic_decision_audit: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "PUBSUB_TOPIC_DECISION_AUDIT", "QNA_PUBSUB_TOPIC_DECISION_AUDIT"
        ),
    )
    pubsub_topic_token_usage: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "PUBSUB_TOPIC_TOKEN_USAGE", "QNA_PUBSUB_TOPIC_TOKEN_USAGE"
        ),
    )
    pubsub_topic_trigger_dlq: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "PUBSUB_TOPIC_TRIGGER_DLQ", "QNA_PUBSUB_TOPIC_TRIGGER_DLQ"
        ),
    )
    # Pub/Sub flow control: strictly limited to 1 concurrent message
    # to ensure single-threaded processing aligned with HTTP worker count.
    # Validation happens in _enforce_single_worker model_validator.
    pubsub_max_workers: int = Field(default=1)

    langfuse_public_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "LANGFUSE_PUBLIC_KEY",
            "QNA_LANGFUSE_PUBLIC_KEY",
        ),
    )
    langfuse_secret_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "LANGFUSE_SECRET_KEY",
            "QNA_LANGFUSE_SECRET_KEY",
        ),
    )
    langfuse_base_url: str = Field(
        default="https://cloud.langfuse.com",
        validation_alias=AliasChoices(
            "LANGFUSE_BASE_URL",
            "QNA_LANGFUSE_BASE_URL",
        ),
    )

    model_id: str = Field(validation_alias="OPENAI_MODEL")
    cheap_model_id: str = Field(
        default="",
        validation_alias=AliasChoices("CHEAP_MODEL_ID", "QNA_CHEAP_MODEL_ID"),
    )
    expensive_model_id: str = Field(
        default="",
        validation_alias=AliasChoices("EXPENSIVE_MODEL_ID", "QNA_EXPENSIVE_MODEL_ID"),
    )
    llm_api_key: str = Field(validation_alias="OPENAI_API_KEY")
    llm_base_url: str = Field(validation_alias="OPENAI_BASE_URL")
    # Deprecated: WorkflowLLMProvider was removed due to production-safety issues
    # (timing-dependent parsing, broken ID aggregation, brittle exception handling).
    # This setting is kept for backwards compatibility but has no effect.
    llm_workflow_mode: bool = Field(
        default=False,
        validation_alias=AliasChoices("LLM_WORKFLOW_MODE", "QNA_LLM_WORKFLOW_MODE"),
    )
    llm_timeout_seconds: int = Field(
        default=120,
        ge=30,
        le=600,
        validation_alias=AliasChoices("LLM_TIMEOUT_SECONDS", "QNA_LLM_TIMEOUT_SECONDS"),
    )
    openai_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        validation_alias=AliasChoices("OPENAI_TEMPERATURE", "QNA_OPENAI_TEMPERATURE"),
    )
    openai_max_output_tokens: int = Field(
        default=4096,
        ge=1,
        le=8192,
        validation_alias=AliasChoices(
            "OPENAI_MAX_OUTPUT_TOKENS", "QNA_OPENAI_MAX_OUTPUT_TOKENS"
        ),
    )

    qa_gen_max_retries: int = Field(
        default=3,
        ge=1,
        validation_alias=AliasChoices("QA_GEN_MAX_RETRIES", "QNA_QA_GEN_MAX_RETRIES"),
    )
    qa_gen_timeout_ms: int = Field(
        default=30000,
        ge=1000,
        validation_alias=AliasChoices("QA_GEN_TIMEOUT_MS", "QNA_QA_GEN_TIMEOUT_MS"),
    )
    qa_gen_max_iterations: int = Field(
        default=3,
        ge=1,
        validation_alias=AliasChoices(
            "QA_GEN_MAX_ITERATIONS", "QNA_QA_GEN_MAX_ITERATIONS"
        ),
    )

    submission_service_url: str = Field(validation_alias="SUBMISSION_SERVICE_URL")
    knowledge_service_url: str = Field(validation_alias="KNOWLEDGE_SERVICE_URL")

    release: str | None = Field(
        default=None,
        validation_alias=AliasChoices("RELEASE", "QNA_RELEASE"),
    )

    cors_allowed_origins: str = Field(
        default="",
        validation_alias=AliasChoices(
            "CORS_ALLOWED_ORIGINS",
            "QNA_CORS_ALLOWED_ORIGINS",
        ),
    )
    cors_allow_credentials: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "CORS_ALLOW_CREDENTIALS",
            "QNA_CORS_ALLOW_CREDENTIALS",
        ),
    )

    grpc_tls_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "GRPC_TLS_ENABLED",
            "QNA_GRPC_TLS_ENABLED",
        ),
    )
    grpc_tls_cert_path: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "GRPC_TLS_CERT_PATH",
        ),
    )
    grpc_timeout_seconds: float = Field(
        default=30.0,
        ge=5.0,
        le=120.0,
        validation_alias=AliasChoices(
            "GRPC_TIMEOUT_SECONDS",
            "QNA_GRPC_TIMEOUT_SECONDS",
        ),
    )

    enable_test_routes: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "ENABLE_TEST_ROUTES",
            "QNA_ENABLE_TEST_ROUTES",
        ),
    )
    prompt_label: str = Field(
        default="production",
        validation_alias=AliasChoices(
            "PROMPT_LABEL",
            "QNA_PROMPT_LABEL",
        ),
    )

    @field_validator(
        "pubsub_project_id",
        "pubsub_subscription_trigger",
        "pubsub_topic_complete",
        "pubsub_topic_decision_audit",
        "pubsub_topic_token_usage",
        "pubsub_topic_trigger_dlq",
        "langfuse_public_key",
        "langfuse_secret_key",
        "release",
        "grpc_tls_cert_path",
        mode="before",
    )
    @classmethod
    def _normalize_blank_strings(cls, value: Any) -> Any:
        """Convert blank strings to None for optional fields."""
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    @model_validator(mode="after")
    def _enforce_single_worker(self) -> Settings:
        """Validate single-worker constraint.

        Both HTTP (Granian) and Pub/Sub are strictly limited to 1 worker
        to ensure single-threaded operation and prevent concurrency issues.
        Raises ConfigurationError if unsupported values are provided.
        """
        if self.workers != 1:
            raise ConfigurationError(
                "workers must be 1 (single-worker architecture enforced)",
                workers=self.workers,
            )
        if self.pubsub_max_workers != 1:
            raise ConfigurationError(
                "pubsub_max_workers must be 1 (single-worker architecture enforced)",
                pubsub_max_workers=self.pubsub_max_workers,
            )
        return self

    @model_validator(mode="after")
    def _default_model_ids(self) -> Settings:
        """Default cheap and expensive model IDs to model_id if not set."""
        if not self.cheap_model_id or self.cheap_model_id.strip() == "":
            self.cheap_model_id = self.model_id
        if not self.expensive_model_id or self.expensive_model_id.strip() == "":
            self.expensive_model_id = self.model_id
        return self

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def cheap_model_enabled(self) -> bool:
        """Return True if a distinct cheap model is configured."""
        return self.cheap_model_id != self.model_id

    @property
    def worker_ready(self) -> bool:
        return all(
            (
                self.pubsub_project_id,
                self.pubsub_subscription_trigger,
                self.pubsub_topic_complete,
                self.llm_api_key,
            )
        )

    @property
    def is_development(self) -> bool:
        return self.environment in {RuntimeEnvironment.LOCAL, RuntimeEnvironment.DEV}

    @cached_property
    def service_name(self) -> str:
        """Return the canonical service identity used in logs and events."""
        return "qna-generation-agent"

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS allowed origins from comma-separated string.

        Returns empty list if no origins configured (no CORS restrictions).
        Use '*' to allow all origins (not recommended for production).
        """
        if not self.cors_allowed_origins:
            return []
        return [
            origin.strip()
            for origin in self.cors_allowed_origins.split(",")
            if origin.strip()
        ]

    def validate_settings(self) -> None:
        """Validate serve-process settings."""
        if not self.host.strip():
            raise ConfigurationError("HTTP host cannot be empty")
        if not self.worker_ready:
            raise ConfigurationError(
                "Serve runtime is missing required configuration",
                pubsub_project_id=self.pubsub_project_id,
                pubsub_subscription_trigger=self.pubsub_subscription_trigger,
                pubsub_topic_complete=self.pubsub_topic_complete,
                llm_api_key=bool(self.llm_api_key),
            )


def load_settings() -> Settings:
    """Load a fresh settings object from environment."""
    return Settings()  # type: ignore[call-arg]
