"""LLM provider port."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

from qna_generation_agent.application.dto import AssessmentContext, QuestionBatch

T = TypeVar("T", bound=BaseModel)


class LLMProvider(ABC):
    """Strategy pattern: structured and open-ended generation backends."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return the model ID for this provider."""

    @abstractmethod
    async def generate_structured(
        self,
        *,
        context: AssessmentContext,
        count: int,
        difficulty_level: str | None,
        correlation_id: str,
    ) -> QuestionBatch:
        """Generate structured questions."""

    @abstractmethod
    async def generate_non_structured(
        self,
        *,
        context: AssessmentContext,
        count: int,
        difficulty_level: str | None,
        correlation_id: str,
    ) -> QuestionBatch:
        """Generate non-structured questions."""

    @abstractmethod
    async def generate_with_prompt(
        self,
        *,
        prompt: str,
        count: int,
        difficulty_level: str | None,
        correlation_id: str,
        question_type: str,
    ) -> QuestionBatch:
        """Generate questions using a pre-compiled prompt.

        Args:
            prompt: The compiled prompt text.
            count: Number of questions to generate.
            difficulty: Target difficulty level.
            correlation_id: Correlation ID for tracing.
            question_type: "structured" or "non_structured".

        Returns:
            Batch of generated question drafts.
        """

    @abstractmethod
    async def health_check(self) -> bool:
        """Check connectivity to the LLM provider.

        Returns True if the provider is reachable and responsive.
        """
        ...

    @abstractmethod
    async def invoke_with_system_and_user[T: BaseModel](
        self,
        system_message: str,
        user_message: str,
        *,
        structured_output_model: type[T],
        model_tier: str = "expensive",
    ) -> T | None:
        """Invoke LLM with separate system and user messages.

        This is the primary method for the new architecture. The system
        message (from the prompt asset) defines role and format rules.
        The user message (built from template) contains the specific
        request data.

        Args:
            system_message: Static system prompt from the prompt asset.
            user_message: Dynamic user prompt built from template.
            structured_output_model: Pydantic model for response validation.
            model_tier: Which model to use ("expensive" or "cheap").

        Returns:
            Parsed structured output or None if failed.

        Raises:
            LLMTransientError: Timeout, rate limit, server error (retry).
            LLMPermanentError: Auth failure, bad request, model not found.
        """
        ...
