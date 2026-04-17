"""Assessment Submission Service client port."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class AssessmentConfig:
    """Assessment configuration from Submission Service per spec."""

    assessment_id: str
    workflow_id: str
    assessor_id: str
    assessment_title: str
    purpose: str | None
    duration_minutes: int
    difficulty_level: str | None
    structured_question_count: int
    non_structured_question_count: int
    web_research_mode: str
    status: str
    deadline: str | None = None


@dataclass(frozen=True, slots=True)
class GetAssessmentConfigCommand:
    """Command to get assessment config."""

    assessment_id: str
    workflow_id: str


@dataclass(frozen=True, slots=True)
class Question:
    """Question data for Submission Service."""

    question_id: str
    question_type: str
    content: str
    structured_answer: str = ""
    non_structured_model_answer: str = ""
    metadata_json: str = ""
    topic_id: str = ""
    iteration: int = 1
    sort_order: int = 1


@dataclass(frozen=True, slots=True)
class QuestionSetRecord:
    """Question set record returned by Submission Service."""

    id: str
    workflow_id: str
    iteration_count: int
    status: str
    created_at: datetime | None


@dataclass(frozen=True, slots=True)
class CreateQuestionSetCommand:
    """Command to create a question set in Submission Service."""

    workflow_id: str


@dataclass(frozen=True, slots=True)
class WriteGeneratedQuestionsCommand:
    """Command to write generated questions to Submission Service."""

    question_set_id: str
    questions: list[Question]


@dataclass(frozen=True, slots=True)
class WriteGeneratedQuestionsResult:
    """Result of writing generated questions."""

    questions_written: int
    status: str


@dataclass(frozen=True, slots=True)
class IncrementIterationCommand:
    """Command to increment question set iteration."""

    question_set_id: str


@dataclass(frozen=True, slots=True)
class IncrementIterationResult:
    """Result of incrementing iteration."""

    question_set_id: str
    iteration_count: int
    status: str


class SubmissionClient(ABC):
    """Port for Assessment Submission Service gRPC client."""

    @abstractmethod
    async def get_assessment_config(
        self,
        command: GetAssessmentConfigCommand,
    ) -> AssessmentConfig:
        """Get assessment configuration from Submission Service per spec.

        Args:
            command: The get config command with assessment_id and workflow_id.

        Returns:
            The assessment configuration.

        Raises:
            StorageTransientError: If the request fails temporarily (retryable).
            StoragePermanentError: If the request fails permanently.
        """

    @abstractmethod
    async def create_question_set(
        self,
        command: CreateQuestionSetCommand,
    ) -> QuestionSetRecord:
        """Create a question set record in the Submission Service.

        Args:
            command: The create command with id and workflow_id.

        Returns:
            The created question set record.

        Raises:
            StorageTransientError: If the request fails temporarily (retryable).
            StoragePermanentError: If the request fails permanently.
        """

    @abstractmethod
    async def write_generated_questions(
        self,
        command: WriteGeneratedQuestionsCommand,
    ) -> WriteGeneratedQuestionsResult:
        """Write generated questions to the Submission Service.

        Args:
            command: The write command with question set ID and questions.

        Returns:
            The result with count of questions written.

        Raises:
            StorageTransientError: If the request fails temporarily (retryable).
            StoragePermanentError: If the request fails permanently.
        """

    @abstractmethod
    async def increment_iteration(
        self,
        command: IncrementIterationCommand,
    ) -> IncrementIterationResult:
        """Increment the iteration count for a question set.

        Args:
            command: The increment command with question set ID.

        Returns:
            The result with new iteration count.

        Raises:
            StorageTransientError: If the request fails temporarily (retryable).
            StoragePermanentError: If the request fails permanently.
        """

    @abstractmethod
    async def close(self) -> None:
        """Close the client connection."""
