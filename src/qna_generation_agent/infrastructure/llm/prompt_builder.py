"""Prompt templates and structured-output schemas for Strands and Langfuse."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from qna_generation_agent.application.dto import AssessmentContext
from qna_generation_agent.domain.enums import QuestionType

# =============================================================================
# Legacy GeneratedQuestion Schemas (for backward compatibility)
# =============================================================================


class GeneratedQuestionSchema(BaseModel):
    """Structured output schema returned by the model."""

    model_config = ConfigDict(strict=True)

    question_text: str = Field(min_length=1)
    answer_text: str = Field(min_length=1)
    explanation: str | None = None
    references: list[str] = Field(default_factory=list)
    topic_id: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class GeneratedQuestionBatchSchema(BaseModel):
    """Structured output schema for one generation batch."""

    model_config = ConfigDict(strict=True)

    questions: list[GeneratedQuestionSchema]


# =============================================================================
# Langfuse Prompt: Assessment Generator
# Generates ELP assessment questions for foreign students in Singapore
# Variables: {structured_count}, {non_structured_count}, {difficulty}, {topics}, {chunks}
# =============================================================================


class MCQOptionsSchema(BaseModel):
    """MCQ options with A, B, C, D choices."""

    model_config = ConfigDict(strict=True)

    A: str = Field(min_length=1, description="Option A text")
    B: str = Field(min_length=1, description="Option B text")
    C: str = Field(min_length=1, description="Option C text")
    D: str = Field(min_length=1, description="Option D text")


class StructuredQuestionMetadataSchema(BaseModel):
    """Metadata for structured (MCQ) questions."""

    model_config = ConfigDict(strict=True)

    question_type: str = Field(
        default="structured",
        pattern="^structured$",
        description="Question type identifier",
    )
    options: MCQOptionsSchema = Field(
        description="Four MCQ options with keys A, B, C, D"
    )
    source_chunk_ids: list[str] = Field(
        min_length=1,
        description="Source chunk IDs that ground this question",
    )
    difficulty: str = Field(
        pattern="^(easy|medium|hard)$",
        description="Question difficulty level",
    )
    topic: str = Field(
        min_length=1,
        description="Topic name from the source material",
    )


class NonStructuredQuestionMetadataSchema(BaseModel):
    """Metadata for non-structured (open-ended) questions."""

    model_config = ConfigDict(strict=True)

    question_type: str = Field(
        default="non_structured",
        pattern="^non_structured$",
        description="Question type identifier",
    )
    source_chunk_ids: list[str] = Field(
        min_length=1,
        description="Source chunk IDs that ground this question",
    )
    difficulty: str = Field(
        pattern="^(easy|medium|hard)$",
        description="Question difficulty level",
    )
    topic: str = Field(
        min_length=1,
        description="Topic name from the source material",
    )
    rubric: str = Field(
        min_length=1,
        description="Marking rubric with point allocations",
    )


class AssessmentQuestionSchema(BaseModel):
    """Single question in the Assessment Generator output."""

    model_config = ConfigDict(strict=True)

    question_id: str = Field(
        pattern=r"^q-\d+$",
        description="Unique question identifier (e.g., q-001)",
    )
    question_type: str = Field(
        pattern="^(structured|non_structured)$",
        description="Type of question: structured (MCQ) or non_structured (open-ended)",
    )
    content: str = Field(
        min_length=10,
        description="The question text/stem. For MCQ: must be self-contained without blanks or underscores.",
    )
    structured_answer: str | None = Field(
        default=None,
        pattern="^[ABCD]$",
        description="For MCQ only: the correct answer (A, B, C, or D)",
    )
    non_structured_model_answer: str | None = Field(
        default=None,
        min_length=1,
        description="For open-ended only: comprehensive model answer in clear, grammatically correct English",
    )
    # Union type with metadata variants - validated at runtime
    metadata: dict[str, object] = Field(
        description="Question metadata including source grounding and difficulty. Use StructuredQuestionMetadataSchema or NonStructuredQuestionMetadataSchema shape based on question_type.",
    )


class AssessmentGeneratorOutputSchema(BaseModel):
    """Structured output for the Assessment Generator prompt.

    Generates ELP assessment questions targeting common English difficulties
    for foreign students in Singapore.

    Prompt Variables:
        - structured_count: Number of MCQ questions to generate
        - non_structured_count: Number of open-ended questions to generate
        - difficulty: Target difficulty (easy, medium, hard)
        - topics: Comma-separated list of topics to cover
        - chunks: Formatted knowledge chunks from source material
    """

    model_config = ConfigDict(strict=True)

    questions: list[AssessmentQuestionSchema] = Field(
        min_length=1,
        description="List of generated questions (both MCQ and open-ended)",
    )


# =============================================================================
# Langfuse Prompt: MCQ Explanation Generator
# Generates detailed explanations for why MCQ answers are correct/incorrect
# Variables: {question}, {options}, {correct_answer}, {target_audience}
# =============================================================================


class MCQDistractorExplanationSchema(BaseModel):
    """Explanation for a single MCQ distractor option."""

    model_config = ConfigDict(strict=True)

    option_letter: str = Field(
        pattern="^[ABCD]$",
        description="The option letter (A, B, C, or D)",
    )
    option_text: str = Field(
        min_length=1,
        description="The text of this option",
    )
    is_correct: bool = Field(
        description="Whether this is the correct answer",
    )
    explanation: str = Field(
        min_length=1,
        description="Detailed explanation of why this option is correct or incorrect",
    )
    l1_interference_note: str | None = Field(
        default=None,
        description="For distractors: the L1 interference error this represents",
    )


class MCQExplanationOutputSchema(BaseModel):
    """Structured output for the MCQ Explanation Generator prompt.

    Generates detailed explanations for each MCQ option, highlighting:
    - Why the correct answer is correct
    - Why each distractor is incorrect
    - Common L1 interference errors (e.g., Chinese article confusion)

    Prompt Variables:
        - question: The MCQ question stem
        - options: JSON object with A, B, C, D options
        - correct_answer: The correct option letter (A, B, C, D)
        - target_audience: Description of target learners (e.g., Chinese L1, Vietnamese L1)
    """

    model_config = ConfigDict(strict=True)

    question_analysis: str = Field(
        min_length=1,
        description="Brief analysis of what the question tests (grammar point, skill, etc.)",
    )
    option_explanations: list[MCQDistractorExplanationSchema] = Field(
        min_length=4,
        max_length=4,
        description="Explanations for all 4 options",
    )
    teaching_tip: str = Field(
        min_length=1,
        description="A teaching tip for instructors on how to address common errors",
    )
    cefr_level: str = Field(
        pattern="^(A1|A2|B1|B2|C1|C2)$",
        description="Estimated CEFR level of this question",
    )


# =============================================================================
# Langfuse Prompt: MCQ Answer Generator
# Generates model answers and distractors for MCQ questions
# Variables: {question_stem}, {grammar_target}, {difficulty}, {l1_background}
# =============================================================================


class MCQAnswerGeneratorOutputSchema(BaseModel):
    """Structured output for the MCQ Answer Generator prompt.

    Generates complete MCQ answer sets including:
    - The correct answer (grammatically correct)
    - Three distractors reflecting plausible L1 interference errors
    - Rationale for each distractor design

    Prompt Variables:
        - question_stem: The incomplete MCQ question stem
        - grammar_target: The grammar point being tested (e.g., "article usage")
        - difficulty: Target difficulty (easy, medium, hard)
        - l1_background: Target learner L1 (e.g., "Chinese", "Vietnamese", "Mixed")
    """

    model_config = ConfigDict(strict=True)

    question_stem: str = Field(
        min_length=1,
        description="The complete, self-contained question stem",
    )
    correct_answer: MCQDistractorExplanationSchema = Field(
        description="The correct answer with explanation",
    )
    distractors: list[MCQDistractorExplanationSchema] = Field(
        min_length=3,
        max_length=3,
        description="Three incorrect but plausible distractor options",
    )
    grammar_point_tested: str = Field(
        min_length=1,
        description="Specific grammar point this question assesses",
    )
    difficulty_justification: str = Field(
        min_length=1,
        description="Why this difficulty level is appropriate for this content",
    )
    l1_considerations: list[str] = Field(
        min_length=1,
        description="Specific L1 interference errors targeted by distractors",
    )


# =============================================================================
# Prompt Input Schemas (for variable validation)
# =============================================================================


class AssessmentGeneratorInputSchema(BaseModel):
    """Input variables for the Assessment Generator prompt.

    Use this to validate inputs before compiling the prompt:
        prompt = await provider.get_assessment_generator_prompt()
        compiled = prompt.compile(**input_schema.model_dump())
    """

    model_config = ConfigDict(strict=True)

    structured_count: int = Field(
        ge=0,
        description="Number of MCQ questions to generate",
    )
    non_structured_count: int = Field(
        ge=0,
        description="Number of open-ended questions to generate",
    )
    difficulty: str = Field(
        pattern="^(easy|medium|hard)$",
        description="Target difficulty level",
    )
    topics: str = Field(
        min_length=1,
        description="Comma-separated list of topics to cover",
    )
    chunks: str = Field(
        min_length=1,
        description="Formatted knowledge chunks from source material",
    )

    @classmethod
    def from_context(
        cls,
        context: AssessmentContext,
        structured_count: int,
        non_structured_count: int,
        difficulty_level: str | None,
    ) -> AssessmentGeneratorInputSchema:
        """Build input schema from AssessmentContext."""
        chunks_formatted = "\n\n".join(
            f"[Chunk {i}] {chunk}" for i, chunk in enumerate(context.chunks, start=1)
        )
        return cls(
            structured_count=structured_count,
            non_structured_count=non_structured_count,
            difficulty=difficulty_level or "medium",
            topics=", ".join(context.topic_ids),
            chunks=chunks_formatted,
        )


class MCQExplanationInputSchema(BaseModel):
    """Input variables for the MCQ Explanation Generator prompt."""

    model_config = ConfigDict(strict=True)

    question: str = Field(
        min_length=1,
        description="The MCQ question stem",
    )
    options: str = Field(
        min_length=1,
        description="JSON string with A, B, C, D options",
    )
    correct_answer: str = Field(
        pattern="^[ABCD]$",
        description="The correct option letter",
    )
    target_audience: str = Field(
        min_length=1,
        description="Description of target learners (e.g., 'Chinese L1 students')",
    )


class MCQAnswerInputSchema(BaseModel):
    """Input variables for the MCQ Answer Generator prompt."""

    model_config = ConfigDict(strict=True)

    question_stem: str = Field(
        min_length=1,
        description="The incomplete MCQ question stem",
    )
    grammar_target: str = Field(
        min_length=1,
        description="The grammar point being tested (e.g., 'article usage')",
    )
    difficulty: str = Field(
        pattern="^(easy|medium|hard)$",
        description="Target difficulty level",
    )
    l1_background: str = Field(
        min_length=1,
        description="Target learner L1 (e.g., 'Chinese', 'Vietnamese', 'Mixed')",
    )


# =============================================================================
# Legacy Prompt Builders
# =============================================================================


def build_system_prompt(question_type: QuestionType) -> str:
    """Return the system prompt for the requested question type."""
    _BASE_PROMPT = (
        "You are a rigorous assessment designer. "
        "Generate grounded {question_type} questions only from the supplied context. "
    )
    _STRUCTURED_SUFFIX = "Keep metadata concise. Never invent references."
    _OPEN_ENDED_SUFFIX = "Provide direct model answers and concise references."

    if question_type is QuestionType.STRUCTURED:
        return _BASE_PROMPT.format(question_type="structured") + _STRUCTURED_SUFFIX
    return _BASE_PROMPT.format(question_type="open-ended") + _OPEN_ENDED_SUFFIX


def build_user_prompt(
    *,
    context: AssessmentContext,
    count: int,
    difficulty_level: str | None,
    question_type: QuestionType,
) -> str:
    """Build a deterministic user prompt for the Strands agent."""
    mode = "structured" if question_type is QuestionType.STRUCTURED else "open-ended"
    chunks = "\n\n".join(
        f"[Chunk {index}] {chunk}"
        for index, chunk in enumerate(context.chunks, start=1)
    )
    topic_list = ", ".join(context.topic_ids)
    return (
        f"Assessment: {context.title}\n"
        f"Assessment ID: {context.assessment_id}\n"
        f"Topics: {topic_list}\n"
        f"Difficulty: {difficulty_level or 'medium'}\n"
        f"Mode: {mode}\n"
        f"Question count: {count}\n\n"
        "Use only the grounded material below.\n"
        "Return exactly the requested number of questions.\n"
        "Each item must include question_text, answer_text, explanation, references, topic_id, and metadata.\n\n"
        f"{chunks}"
    )
