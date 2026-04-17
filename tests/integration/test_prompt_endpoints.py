"""Integration tests for prompt testing endpoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from blacksheep.contents import JSONContent
from blacksheep.testing import TestClient

from qna_generation_agent.app.bootstrap import ApplicationContainer
from qna_generation_agent.app.settings import LogLevel, RuntimeEnvironment, Settings
from qna_generation_agent.application.dto import AssessmentContext, QuestionBatch
from qna_generation_agent.application.ports.llm import LLMProvider
from qna_generation_agent.application.services.prompt_test_service import (
    PromptTestResult,
)
from qna_generation_agent.infrastructure.messaging.pubsub_publisher import (
    NullEventPublisher,
)
from qna_generation_agent.infrastructure.persistence.in_memory import (
    InMemoryIdempotencyStore,
    InMemoryQuestionSetRepository,
)
from qna_generation_agent.interfaces.http.factory import create_blacksheep_app


def _has_langfuse_credentials() -> bool:
    """Check if Langfuse credentials are available in environment or .env file.

    Returns True if LANGFUSE_PUBLIC_KEY is set in environment or .env file.
    """
    # Check environment variable first
    if os.environ.get("LANGFUSE_PUBLIC_KEY"):
        return True

    # Check .env file if it exists
    env_file = Path(".env")
    if env_file.exists():
        content = env_file.read_text()
        if "LANGFUSE_PUBLIC_KEY" in content and "=" in content:
            # Check if it's not commented out
            for line in content.split("\n"):
                if line.strip().startswith("LANGFUSE_PUBLIC_KEY="):
                    return True

    return False


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


def _create_fresh_settings(
    *,
    langfuse_public_key: str | None = "pk-test",
    langfuse_secret_key: str | None = "sk-test",
) -> Settings:
    """Create a completely fresh Settings object for testing.

    This avoids any state leakage from the test_settings fixture
    or environment variables.
    """
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
        is_development=True,
        enable_test_routes=True,
        langfuse_public_key=langfuse_public_key,
        langfuse_secret_key=langfuse_secret_key,
    )


def _create_test_container(settings: Settings) -> ApplicationContainer:
    """Create a test container with fake dependencies."""
    return ApplicationContainer(
        settings=settings,
        llm_provider=FakeLLMProvider(),
        question_set_repo=InMemoryQuestionSetRepository(),
        idempotency_store=InMemoryIdempotencyStore(),
        event_publisher=NullEventPublisher(),
        telemetry=None,
        subscriber=None,
        submission_client=FakeSubmissionClient(),  # type: ignore[arg-type]
        knowledge_client=FakeKnowledgeClient(),  # type: ignore[arg-type]
    )


@pytest.mark.integration
async def test_assessment_generator_endpoint_with_defaults() -> None:
    """Test assessment generator endpoint with default values."""
    test_settings = _create_fresh_settings()

    result = PromptTestResult(
        result={
            "questions": [
                {
                    "question_id": "q-001",
                    "content": "Test question",
                    "question_type": "structured",
                }
            ]
        },
        prompt_version="Assessment Generator@v3",
        execution_time_ms=1250,
        raw_output='{"questions": [...]}',
    )

    container = _create_test_container(test_settings)
    app = create_blacksheep_app(container)

    await app.start()  # type: ignore[no-untyped-call]
    try:
        with patch(
            "qna_generation_agent.interfaces.http.routes_prompts._get_test_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.test_assessment_generator.return_value = result
            mock_get_service.return_value = mock_service

            client = TestClient(app)

            # POST with minimal content (BlackSheep treats empty JSON as missing body)
            response = await client.post(
                "/test/prompt/assessment",
                content=JSONContent({"structured_count": 2}),
            )

            assert response.status == 200
            data = await response.json()
            assert data["success"] is True
            assert data["prompt_version"] == "Assessment Generator@v3"
            assert data["execution_time_ms"] == 1250
            assert data["result"] is not None

            # Verify the service was called with defaults
            call_kwargs = mock_service.test_assessment_generator.call_args.kwargs
            assert call_kwargs["structured_count"] == 2
            assert call_kwargs["non_structured_count"] == 1
            assert call_kwargs["difficulty"] == "medium"
    finally:
        await app.stop()  # type: ignore[no-untyped-call]


@pytest.mark.integration
async def test_assessment_generator_endpoint_custom_values() -> None:
    """Test assessment generator endpoint with custom values."""
    test_settings = _create_fresh_settings()

    result = PromptTestResult(
        result={"questions": []},
        prompt_version="Assessment Generator@v3",
        execution_time_ms=1000,
    )

    container = _create_test_container(test_settings)
    app = create_blacksheep_app(container)

    await app.start()  # type: ignore[no-untyped-call]
    try:
        with patch(
            "qna_generation_agent.interfaces.http.routes_prompts._get_test_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.test_assessment_generator.return_value = result
            mock_get_service.return_value = mock_service

            client = TestClient(app)

            custom_request = {
                "structured_count": 5,
                "non_structured_count": 2,
                "difficulty": "hard",
                "topics": "Advanced Grammar, Academic Writing",
                "chunks": ["Custom chunk 1", "Custom chunk 2"],
            }

            response = await client.post(
                "/test/prompt/assessment",
                content=JSONContent(custom_request),
            )

            assert response.status == 200

            # Verify custom values were passed
            call_kwargs = mock_service.test_assessment_generator.call_args.kwargs
            assert call_kwargs["structured_count"] == 5
            assert call_kwargs["non_structured_count"] == 2
            assert call_kwargs["difficulty"] == "hard"
            assert call_kwargs["topics"] == "Advanced Grammar, Academic Writing"
            assert call_kwargs["chunks"] == ["Custom chunk 1", "Custom chunk 2"]
    finally:
        await app.stop()  # type: ignore[no-untyped-call]


@pytest.mark.integration
async def test_mcq_answer_generator_endpoint() -> None:
    """Test MCQ answer generator endpoint."""
    test_settings = _create_fresh_settings()

    result = PromptTestResult(
        result={
            "question_stem": "Test question",
            "correct_answer": {"option_letter": "A", "option_text": "was walking"},
            "distractors": [
                {"option_letter": "B", "option_text": "walked"},
                {"option_letter": "C", "option_text": "is walking"},
            ],
        },
        prompt_version="MCQ Answer Generator@v2",
        execution_time_ms=980,
    )

    container = _create_test_container(test_settings)
    app = create_blacksheep_app(container)

    await app.start()  # type: ignore[no-untyped-call]
    try:
        with patch(
            "qna_generation_agent.interfaces.http.routes_prompts._get_test_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.test_mcq_answer_generator.return_value = result
            mock_get_service.return_value = mock_service

            client = TestClient(app)

            response = await client.post(
                "/test/prompt/mcq-answer",
                content=JSONContent({"question_stem": "test"}),
            )

            assert response.status == 200
            data = await response.json()
            assert data["success"] is True
            assert data["prompt_version"] == "MCQ Answer Generator@v2"

            # Verify the request value was passed (we sent "test" in the request)
            call_kwargs = mock_service.test_mcq_answer_generator.call_args.kwargs
            assert call_kwargs["question_stem"] == "test"
            # Grammar target should use the default since we didn't provide it
            assert call_kwargs["grammar_target"] == "past continuous tense"
    finally:
        await app.stop()  # type: ignore[no-untyped-call]


@pytest.mark.integration
async def test_mcq_explanation_generator_endpoint() -> None:
    """Test MCQ explanation generator endpoint."""
    test_settings = _create_fresh_settings()

    result = PromptTestResult(
        result={
            "question_analysis": "This tests article usage with countable nouns",
            "cefr_level": "A2",
            "teaching_tip": "Focus on when to use 'a' vs 'the'",
            "option_explanations": [
                {"option_letter": "A", "is_correct": True, "explanation": "Correct!"},
                {"option_letter": "B", "is_correct": False, "explanation": "Wrong!"},
            ],
        },
        prompt_version="MCQ Explanation Generator@v4",
        execution_time_ms=1450,
    )

    container = _create_test_container(test_settings)
    app = create_blacksheep_app(container)

    await app.start()  # type: ignore[no-untyped-call]
    try:
        with patch(
            "qna_generation_agent.interfaces.http.routes_prompts._get_test_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.test_mcq_explanation_generator.return_value = result
            mock_get_service.return_value = mock_service

            client = TestClient(app)

            response = await client.post(
                "/test/prompt/mcq-explanation",
                content=JSONContent({"question": "test"}),
            )

            assert response.status == 200
            data = await response.json()
            assert data["success"] is True
            assert data["prompt_version"] == "MCQ Explanation Generator@v4"

            # Verify the request value was passed (we sent "test" in the request)
            call_kwargs = mock_service.test_mcq_explanation_generator.call_args.kwargs
            assert call_kwargs["question"] == "test"
            # Default values should be used for fields not in request
            assert call_kwargs["correct_answer"] == "A"
            assert "A" in call_kwargs["options"]
    finally:
        await app.stop()  # type: ignore[no-untyped-call]


@pytest.mark.integration
@pytest.mark.skipif(
    _has_langfuse_credentials(),
    reason="Test requires clean environment without Langfuse credentials. "
    "Skips when LANGFUSE_PUBLIC_KEY is set in environment or .env file.",
)
async def test_prompt_endpoint_returns_503_when_langfuse_not_configured() -> None:
    """Test that endpoints return 503 when Langfuse is not available.

    Note: This test must not be run in parallel with other tests that set
    LANGFUSE_* environment variables, as it relies on clean env state.
    """
    # Create settings WITHOUT Langfuse keys
    test_settings = _create_fresh_settings(
        langfuse_public_key=None,
        langfuse_secret_key=None,
    )

    container = _create_test_container(test_settings)
    app = create_blacksheep_app(container)

    await app.start()  # type: ignore[no-untyped-call]
    try:
        client = TestClient(app)
        response = await client.post(
            "/test/prompt/assessment",
            content=JSONContent({"structured_count": 2}),
        )

        assert response.status == 503
        data = await response.json()
        assert "error" in data
        assert "Langfuse" in data["error"]
    finally:
        await app.stop()  # type: ignore[no-untyped-call]


@pytest.mark.integration
async def test_prompt_endpoint_returns_500_on_execution_error() -> None:
    """Test that endpoints return 500 when prompt execution fails."""
    test_settings = _create_fresh_settings()

    error_result = PromptTestResult(
        result=None,
        prompt_version="unknown",
        execution_time_ms=150,
        error="Prompt 'Assessment Generator' not found in Langfuse",
    )

    container = _create_test_container(test_settings)
    app = create_blacksheep_app(container)

    await app.start()  # type: ignore[no-untyped-call]
    try:
        with patch(
            "qna_generation_agent.interfaces.http.routes_prompts._get_test_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.test_assessment_generator.return_value = error_result
            mock_get_service.return_value = mock_service

            client = TestClient(app)

            response = await client.post(
                "/test/prompt/assessment",
                content=JSONContent({"structured_count": 2}),
            )

            assert response.status == 500
            data = await response.json()
            assert data["success"] is False
            assert "not found" in data["error"]
    finally:
        await app.stop()  # type: ignore[no-untyped-call]
