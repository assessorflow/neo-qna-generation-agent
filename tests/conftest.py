"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from qna_generation_agent.app.settings import LogLevel, RuntimeEnvironment, Settings


@pytest.fixture
def test_settings() -> Settings:
    """Return deterministic process settings for tests."""
    return Settings.model_construct(
        environment=RuntimeEnvironment.LOCAL,
        host="127.0.0.1",
        port=8080,
        workers=1,
        log_level=LogLevel.DEBUG,
        pubsub_project_id=None,
        pubsub_subscription_trigger=None,
        pubsub_topic_complete=None,
        llm_api_key="test-key",
        model_id="gpt-4o-mini",
        llm_base_url="https://api.openai.com/v1",
        submission_service_url="localhost:50052",
        knowledge_service_url="localhost:9030",
    )
