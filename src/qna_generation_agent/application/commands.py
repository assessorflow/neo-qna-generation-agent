"""Command handlers for inbound messages."""

from __future__ import annotations

from qna_generation_agent.application.dto import GenerationCommand, GenerationReceipt
from qna_generation_agent.application.services.generate_qna import GenerateQnAService
from qna_generation_agent.domain.enums import DifficultyLevel, Purpose, ValidationResult
from qna_generation_agent.domain.events import QnAGenerationTriggered


def _parse_difficulty(value: str | None) -> DifficultyLevel | None:
    """Parse difficulty string to enum."""
    if value is None:
        return None
    try:
        return DifficultyLevel(value.lower())
    except ValueError:
        return None


def _parse_purpose(value: str | None) -> Purpose | None:
    """Parse purpose string to enum."""
    if value is None:
        return None
    try:
        return Purpose(value.lower())
    except ValueError:
        return None


def _parse_validation_result(value: str | None) -> ValidationResult | None:
    """Parse validation result string to enum."""
    if value is None:
        return None
    try:
        return ValidationResult(value.lower())
    except ValueError:
        return None


class HandleGenerationTrigger:
    """Translate inbound events into a generation command."""

    def __init__(self, service: GenerateQnAService) -> None:
        """Initialize with the generation service."""
        self._service = service

    async def handle(self, event: QnAGenerationTriggered) -> GenerationReceipt:
        """Handle a QnAGenerationTriggered event.

        Translates the domain event into a generation command and
        delegates to the GenerateQnAService for execution.

        Args:
            event: The triggered generation event.

        Returns:
            Receipt of the generation execution.

        Raises:
            IdempotencyConflict: If generation is already in progress or completed.
            RetrievalError: If required data cannot be retrieved.
            GenerationError: If question generation fails.
        """
        command = GenerationCommand(
            request_id=event.event_id,
            workflow_id=event.workflow_id,
            correlation_id=event.correlation_id,
            trace_id=event.trace_id,
            assessment_id=event.assessment_id,
            question_set_id=event.question_set_id,
            validation_result=_parse_validation_result(event.validation_result),
            iteration=event.iteration,
            structured_count=event.structured_count,
            non_structured_count=event.non_structured_count,
            difficulty_level=_parse_difficulty(event.difficulty_level),
            purpose=_parse_purpose(event.purpose),
            feedback_issues=event.feedback_issues,
        )
        return await self._service.execute(command)
