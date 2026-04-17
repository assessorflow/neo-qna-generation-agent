"""Integration tests for the HTTP application wiring."""

from __future__ import annotations

from typing import Any

import pytest
from blacksheep.testing import TestClient

from qna_generation_agent.app.bootstrap import ApplicationContainer
from qna_generation_agent.app.settings import Settings
from qna_generation_agent.application.dto import AssessmentContext, QuestionBatch
from qna_generation_agent.application.ports.llm import LLMProvider
from qna_generation_agent.infrastructure.messaging.pubsub_publisher import (
    NullEventPublisher,
)
from qna_generation_agent.infrastructure.persistence.in_memory import (
    InMemoryIdempotencyStore,
    InMemoryQuestionSetRepository,
)
from qna_generation_agent.interfaces.http.factory import create_blacksheep_app


class FakeLLMProvider(LLMProvider):
    """Minimal stub to satisfy readiness checks."""

    @property
    def model_id(self) -> str:
        return "fake-model"

    async def generate_structured(
        self,
        *,
        context: AssessmentContext,
        count: int,
        difficulty_level: str | None,
        correlation_id: str,
    ) -> QuestionBatch:
        del context, count, difficulty_level, correlation_id
        return QuestionBatch(questions=[], model_name="fake", prompt_version="test")

    async def generate_non_structured(
        self,
        *,
        context: AssessmentContext,
        count: int,
        difficulty_level: str | None,
        correlation_id: str,
    ) -> QuestionBatch:
        del context, count, difficulty_level, correlation_id
        return QuestionBatch(questions=[], model_name="fake", prompt_version="test")

    async def generate_with_prompt(
        self,
        *,
        prompt: str,
        count: int,
        difficulty_level: str | None,
        correlation_id: str,
        question_type: str,
    ) -> QuestionBatch:
        del prompt, count, difficulty_level, correlation_id, question_type
        return QuestionBatch(questions=[], model_name="fake", prompt_version="test")

    async def health_check(self) -> bool:
        """Return True for readiness probe tests."""
        return True

    async def invoke_with_schema(
        self, prompt: str, *, structured_output_model: type[Any]
    ) -> Any | None:
        """Invoke LLM with structured output schema - not used in HTTP tests."""
        del prompt, structured_output_model
        return None


class FakeSubmissionClient:
    """Minimal stub for submission client."""

    async def close(self) -> None:
        pass

    async def health_check(self) -> bool:
        """Return True for readiness probe tests."""
        return True


class FakeKnowledgeClient:
    """Minimal stub for knowledge client."""

    async def close(self) -> None:
        pass

    async def health_check(self) -> bool:
        """Return True for readiness probe tests."""
        return True


@pytest.mark.integration
async def test_health_endpoints_return_json(test_settings: Settings) -> None:
    container = ApplicationContainer(
        settings=test_settings,
        llm_provider=FakeLLMProvider(),
        question_set_repo=InMemoryQuestionSetRepository(),
        idempotency_store=InMemoryIdempotencyStore(),
        event_publisher=NullEventPublisher(),
        telemetry=None,
        subscriber=None,
        submission_client=FakeSubmissionClient(),  # type: ignore[arg-type]
        knowledge_client=FakeKnowledgeClient(),  # type: ignore[arg-type]
    )
    app = create_blacksheep_app(container)

    await app.start()  # type: ignore[no-untyped-call]
    try:
        client = TestClient(app)

        health = await client.get("/healthz")
        live = await client.get("/livez")
        ready = await client.get("/readyz")
        version = await client.get("/version")
        health_payload = await health.json()
        live_payload = await live.json()
        ready_payload = await ready.json()
        version_payload = await version.json()

        assert health.status == 200
        assert health_payload["status"] == "healthy"
        assert live_payload["alive"] is True
        assert ready_payload["ready"] is True
        assert version_payload["version"] == "0.1.0"
    finally:
        await app.stop()  # type: ignore[no-untyped-call]


@pytest.mark.integration
async def test_request_id_is_echoed(test_settings: Settings) -> None:
    container = ApplicationContainer(
        settings=test_settings,
        llm_provider=FakeLLMProvider(),
        question_set_repo=InMemoryQuestionSetRepository(),
        idempotency_store=InMemoryIdempotencyStore(),
        event_publisher=NullEventPublisher(),
        telemetry=None,
        subscriber=None,
        submission_client=FakeSubmissionClient(),  # type: ignore[arg-type]
        knowledge_client=FakeKnowledgeClient(),  # type: ignore[arg-type]
    )
    app = create_blacksheep_app(container)

    await app.start()  # type: ignore[no-untyped-call]
    try:
        client = TestClient(app)
        response = await client.get("/healthz", headers={"x-request-id": "req-test"})

        assert response.status == 200
        assert response.get_first_header(b"x-request-id") == b"req-test"
    finally:
        await app.stop()  # type: ignore[no-untyped-call]
