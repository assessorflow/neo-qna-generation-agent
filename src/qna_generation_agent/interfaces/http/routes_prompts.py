"""HTTP routes for testing Langfuse prompts.

These endpoints are available when ENABLE_TEST_ROUTES=true or in non-production
environments. They allow testing of individual prompts via Strands workflow execution.
"""

from __future__ import annotations

from blacksheep import Application, FromJSON, Response
from blacksheep.server.responses import json

from qna_generation_agent.app.bootstrap import ApplicationContainer
from qna_generation_agent.app.logging import get_logger
from qna_generation_agent.application.services.prompt_test_service import (
    PromptTestResult,
    PromptTestService,
)
from qna_generation_agent.interfaces.http.schemas import (
    AssessmentGeneratorTestRequest,
    MCQAnswerGeneratorTestRequest,
    MCQExplanationGeneratorTestRequest,
    PromptTestResponse,
)

logger = get_logger(__name__)


def _get_test_service(container: ApplicationContainer) -> PromptTestService:
    """Get or create the PromptTestService from container settings.

    Args:
        container: The application container with settings.

    Returns:
        Configured PromptTestService instance.

    Raises:
        RuntimeError: If Langfuse is not configured (prompt provider unavailable).
    """
    settings = container.settings

    # Check if we have a prompt provider (requires Langfuse)
    if not settings.langfuse_enabled:
        raise RuntimeError(
            "Prompt testing requires Langfuse. "
            "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY."
        )

    # We need to create a new PromptTestService (not shared with main workflow)
    # because it has different initialization requirements
    from qna_generation_agent.infrastructure.llm.langfuse_prompt_provider import (
        LangfusePromptProvider,
    )

    prompt_provider = LangfusePromptProvider(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_base_url,
        environment=settings.environment.value,
        release=settings.release,
    )

    return PromptTestService(
        model_id=settings.model_id,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        timeout_seconds=settings.llm_timeout_seconds,
        prompt_provider=prompt_provider,
    )


def _make_response(result: PromptTestResult) -> Response:
    """Create a JSON response from PromptTestResult.

    Args:
        result: The test result to convert.

    Returns:
        BlackSheep JSON Response.
    """
    response_data = PromptTestResponse(
        success=result.result is not None and result.error is None,
        prompt_version=result.prompt_version,
        execution_time_ms=result.execution_time_ms,
        result=result.result,
        raw_output=result.raw_output,
        error=result.error,
    )

    status = 200 if response_data.success else 500
    return json(response_data.model_dump(mode="json"), status=status)


def register_prompt_test_routes(app: Application) -> None:
    """Register prompt testing routes on the application.

    These routes are available for testing individual Langfuse prompts
    via direct HTTP calls. Useful for development and debugging.

    Args:
        app: The BlackSheep application instance.
    """

    @app.router.post("/test/prompt/assessment")
    async def test_assessment_generator(
        request_body: FromJSON[AssessmentGeneratorTestRequest],
        container: ApplicationContainer,
    ) -> Response:
        """Test the Assessment Generator prompt.

        Generates assessment questions (MCQ and/or open-ended) based on
        provided document chunks and topics.

        Example request body (with defaults):
        ```json
        {
            "structured_count": 2,
            "non_structured_count": 1,
            "difficulty": "medium",
            "topics": "Grammar, Vocabulary, Article Usage",
            "chunks": [
                "Many foreign students struggle with English articles...",
                "Past perfect tense describes an action completed before..."
            ]
        }
        ```

        Returns:
            PromptTestResponse with parsed AssessmentGeneratorOutputSchema.
        """
        try:
            service = _get_test_service(container)
        except RuntimeError as e:
            return json({"error": str(e)}, status=503)

        body = request_body.value

        result = await service.test_assessment_generator(
            structured_count=body.structured_count,
            non_structured_count=body.non_structured_count,
            difficulty=body.difficulty,
            topics=body.topics,
            chunks=body.chunks,
        )

        return _make_response(result)

    @app.router.post("/test/prompt/mcq-answer")
    async def test_mcq_answer_generator(
        request_body: FromJSON[MCQAnswerGeneratorTestRequest],
        container: ApplicationContainer,
    ) -> Response:
        """Test the MCQ Answer Generator prompt.

        Generates MCQ answers with L1-targeted distractors for a given
        question stem.

        Example request body (with defaults):
        ```json
        {
            "question_stem": "The student ____ to school yesterday...",
            "grammar_target": "past continuous tense",
            "difficulty": "medium",
            "l1_background": "Chinese"
        }
        ```

        Returns:
            PromptTestResponse with parsed MCQAnswerGeneratorOutputSchema.
        """
        try:
            service = _get_test_service(container)
        except RuntimeError as e:
            return json({"error": str(e)}, status=503)

        body = request_body.value

        result = await service.test_mcq_answer_generator(
            question_stem=body.question_stem,
            grammar_target=body.grammar_target,
            difficulty=body.difficulty,
            l1_background=body.l1_background,
        )

        return _make_response(result)

    @app.router.post("/test/prompt/mcq-explanation")
    async def test_mcq_explanation_generator(
        request_body: FromJSON[MCQExplanationGeneratorTestRequest],
        container: ApplicationContainer,
    ) -> Response:
        """Test the MCQ Explanation Generator prompt.

        Generates detailed explanations for why each MCQ option is
        correct or incorrect, with L1 interference notes.

        Example request body (with defaults):
        ```json
        {
            "question": "Choose the correct article: I bought ____ book...",
            "options": {
                "A": "a",
                "B": "an",
                "C": "the",
                "D": "(no article)"
            },
            "correct_answer": "A",
            "target_audience": "Chinese L1 students learning English..."
        }
        ```

        Returns:
            PromptTestResponse with parsed MCQExplanationOutputSchema.
        """
        try:
            service = _get_test_service(container)
        except RuntimeError as e:
            return json({"error": str(e)}, status=503)

        body = request_body.value

        result = await service.test_mcq_explanation_generator(
            question=body.question,
            options=body.options,
            correct_answer=body.correct_answer,
            target_audience=body.target_audience,
        )

        return _make_response(result)
