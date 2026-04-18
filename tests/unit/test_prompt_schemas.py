"""Unit tests for prompt builder schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from qna_generation_agent.application.dto import AssessmentContext
from qna_generation_agent.domain.enums import DifficultyLevel
from qna_generation_agent.infrastructure.llm.prompt_builder import (
    AssessmentGeneratorInputSchema,
    AssessmentGeneratorOutputSchema,
    AssessmentQuestionSchema,
    MCQAnswerGeneratorOutputSchema,
    MCQAnswerInputSchema,
    MCQDistractorExplanationSchema,
    MCQExplanationInputSchema,
    MCQExplanationOutputSchema,
    MCQOptionsSchema,
    NonStructuredQuestionMetadataSchema,
    StructuredQuestionMetadataSchema,
)


@pytest.mark.unit
class TestMCQOptionsSchema:
    """Tests for MCQOptionsSchema."""

    def test_valid_options(self) -> None:
        opts = MCQOptionsSchema(
            A="Option A text",
            B="Option B text",
            C="Option C text",
            D="Option D text",
        )
        assert opts.A == "Option A text"
        assert opts.B == "Option B text"

    def test_empty_option_fails(self) -> None:
        with pytest.raises(ValidationError):
            MCQOptionsSchema(
                A="",
                B="Valid",
                C="Valid",
                D="Valid",
            )


@pytest.mark.unit
class TestStructuredQuestionMetadataSchema:
    """Tests for StructuredQuestionMetadataSchema."""

    def test_valid_metadata(self) -> None:
        opts = MCQOptionsSchema(
            A="Option A",
            B="Option B",
            C="Option C",
            D="Option D",
        )
        meta = StructuredQuestionMetadataSchema(
            options=opts,
            source_chunk_ids=["chunk-001", "chunk-002"],
            difficulty="medium",
            topic="Grammar",
        )
        assert meta.question_type == "structured"
        assert meta.difficulty == "medium"

    def test_invalid_difficulty_fails(self) -> None:
        opts = MCQOptionsSchema(A="A", B="B", C="C", D="D")
        with pytest.raises(ValidationError):
            StructuredQuestionMetadataSchema(
                options=opts,
                source_chunk_ids=["chunk-001"],
                difficulty="invalid",
                topic="Grammar",
            )


@pytest.mark.unit
class TestNonStructuredQuestionMetadataSchema:
    """Tests for NonStructuredQuestionMetadataSchema."""

    def test_valid_metadata(self) -> None:
        meta = NonStructuredQuestionMetadataSchema(
            source_chunk_ids=["chunk-001"],
            difficulty="hard",
            topic="Writing",
            rubric="5 points for content, 5 for grammar",
        )
        assert meta.question_type == "non_structured"
        assert meta.rubric == "5 points for content, 5 for grammar"


@pytest.mark.unit
class TestAssessmentQuestionSchema:
    """Tests for AssessmentQuestionSchema."""

    def test_valid_structured_question(self) -> None:
        opts = MCQOptionsSchema(A="A", B="B", C="C", D="D")
        meta = StructuredQuestionMetadataSchema(
            options=opts,
            source_chunk_ids=["chunk-001"],
            difficulty="easy",
            topic="Vocabulary",
        )
        question = AssessmentQuestionSchema(
            question_id="q-001",
            question_type="structured",
            content="What is the correct article?",
            structured_answer="B",
            metadata=meta.model_dump(),
        )
        assert question.question_id == "q-001"
        assert question.structured_answer == "B"

    def test_valid_non_structured_question(self) -> None:
        meta = NonStructuredQuestionMetadataSchema(
            source_chunk_ids=["chunk-001"],
            difficulty="hard",
            topic="Essay",
            rubric="Content: 10pts, Grammar: 10pts",
        )
        question = AssessmentQuestionSchema(
            question_id="q-002",
            question_type="non_structured",
            content="Explain the concept...",
            non_structured_model_answer="A comprehensive answer...",
            metadata=meta.model_dump(),
        )
        assert question.question_type == "non_structured"
        assert question.structured_answer is None

    def test_invalid_question_id_pattern(self) -> None:
        with pytest.raises(ValidationError):
            AssessmentQuestionSchema(
                question_id="invalid-id",
                question_type="structured",
                content="Some question?",
                metadata={"question_type": "structured"},
            )


@pytest.mark.unit
class TestAssessmentGeneratorOutputSchema:
    """Tests for AssessmentGeneratorOutputSchema."""

    def test_valid_output(self) -> None:
        opts = MCQOptionsSchema(A="A", B="B", C="C", D="D")
        meta = StructuredQuestionMetadataSchema(
            options=opts,
            source_chunk_ids=["chunk-001"],
            difficulty="medium",
            topic="Grammar",
        )
        question = AssessmentQuestionSchema(
            question_id="q-001",
            question_type="structured",
            content="What is the correct answer?",
            structured_answer="C",
            metadata=meta.model_dump(),
        )
        output = AssessmentGeneratorOutputSchema(questions=[question])
        assert len(output.questions) == 1
        assert output.questions[0].question_id == "q-001"

    def test_empty_questions_fails(self) -> None:
        with pytest.raises(ValidationError):
            AssessmentGeneratorOutputSchema(questions=[])

    def test_questions_validator_handles_single_dict(self) -> None:
        """Test that a single question dict is converted to a list."""
        single_question = {
            "question_id": "q-001",
            "question_type": "structured",
            "content": "What is the answer?",
            "structured_answer": "A",
            "metadata": {
                "options": {"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"},
                "source_chunk_ids": ["chunk-001"],
                "difficulty": "medium",
                "topic": "Grammar",
            },
        }
        output = AssessmentGeneratorOutputSchema(questions=single_question)  # type: ignore
        assert len(output.questions) == 1
        assert output.questions[0].question_id == "q-001"

    def test_questions_validator_handles_dict_with_numeric_keys(self) -> None:
        """Test that a dict with numeric keys is converted to a list."""
        questions_dict = {
            "0": {
                "question_id": "q-001",
                "question_type": "structured",
                "content": "Question 1?",
                "structured_answer": "A",
                "metadata": {
                    "options": {"A": "A", "B": "B", "C": "C", "D": "D"},
                    "source_chunk_ids": ["chunk-001"],
                    "difficulty": "medium",
                    "topic": "Grammar",
                },
            },
            "1": {
                "question_id": "q-002",
                "question_type": "structured",
                "content": "Question 2?",
                "structured_answer": "B",
                "metadata": {
                    "options": {"A": "A", "B": "B", "C": "C", "D": "D"},
                    "source_chunk_ids": ["chunk-002"],
                    "difficulty": "easy",
                    "topic": "Vocabulary",
                },
            },
        }
        output = AssessmentGeneratorOutputSchema(questions=questions_dict)  # type: ignore
        assert len(output.questions) == 2
        assert output.questions[0].question_id == "q-001"
        assert output.questions[1].question_id == "q-002"

    def test_questions_validator_handles_none(self) -> None:
        """Test that None value raises validation error (empty list after validation)."""
        with pytest.raises(ValidationError):
            AssessmentGeneratorOutputSchema(questions=None)  # type: ignore

    def test_questions_validator_handles_invalid_type(self) -> None:
        """Test that non-list, non-dict values raise validation error."""
        with pytest.raises(ValidationError):
            AssessmentGeneratorOutputSchema(questions="invalid string")  # type: ignore


@pytest.mark.unit
class TestAssessmentGeneratorInputSchema:
    """Tests for AssessmentGeneratorInputSchema."""

    def test_valid_input(self) -> None:
        inp = AssessmentGeneratorInputSchema(
            structured_count=5,
            non_structured_count=3,
            difficulty="medium",
            topics="Grammar, Vocabulary",
            chunks="[Chunk 1] Content...",
        )
        assert inp.structured_count == 5
        assert inp.non_structured_count == 3

    def test_negative_count_fails(self) -> None:
        with pytest.raises(ValidationError):
            AssessmentGeneratorInputSchema(
                structured_count=-1,
                non_structured_count=0,
                difficulty="easy",
                topics="Topic",
                chunks="Chunks...",
            )

    def test_from_context_factory(self) -> None:
        context = AssessmentContext(
            assessment_id="ass-001",
            title="Test Assessment",
            topic_ids=["topic-1", "topic-2"],
            chunks=["Chunk content 1", "Chunk content 2"],
        )
        inp = AssessmentGeneratorInputSchema.from_context(
            context=context,
            structured_count=4,
            non_structured_count=2,
            difficulty_level=DifficultyLevel.HARD,
        )
        assert inp.structured_count == 4
        assert inp.difficulty == "hard"
        assert "topic-1, topic-2" in inp.topics
        assert "[Chunk 1]" in inp.chunks
        assert "[Chunk 2]" in inp.chunks


@pytest.mark.unit
class TestMCQDistractorExplanationSchema:
    """Tests for MCQDistractorExplanationSchema."""

    def test_valid_explanation(self) -> None:
        expl = MCQDistractorExplanationSchema(
            option_letter="A",
            option_text="The first option",
            is_correct=True,
            explanation="This is correct because...",
            l1_interference_note=None,
        )
        assert expl.option_letter == "A"
        assert expl.is_correct is True

    def test_invalid_option_letter(self) -> None:
        with pytest.raises(ValidationError):
            MCQDistractorExplanationSchema(
                option_letter="E",
                option_text="Invalid",
                is_correct=False,
                explanation="Wrong letter",
            )


@pytest.mark.unit
class TestMCQExplanationOutputSchema:
    """Tests for MCQExplanationOutputSchema."""

    def test_valid_output(self) -> None:
        explanations = [
            MCQDistractorExplanationSchema(
                option_letter="A",
                option_text="Choice A",
                is_correct=True,
                explanation="Correct because...",
            ),
            MCQDistractorExplanationSchema(
                option_letter="B",
                option_text="Choice B",
                is_correct=False,
                explanation="Incorrect because...",
                l1_interference_note="Chinese L1 confusion",
            ),
            MCQDistractorExplanationSchema(
                option_letter="C",
                option_text="Choice C",
                is_correct=False,
                explanation="Wrong...",
            ),
            MCQDistractorExplanationSchema(
                option_letter="D",
                option_text="Choice D",
                is_correct=False,
                explanation="Wrong...",
            ),
        ]
        output = MCQExplanationOutputSchema(
            question_analysis="Tests article usage",
            option_explanations=explanations,
            teaching_tip="Focus on article rules",
            cefr_level="B1",
            related_grammar=["definite article", "indefinite article"],
            common_errors=["omitting articles", "using wrong article"],
        )
        assert len(output.option_explanations) == 4
        assert output.cefr_level == "B1"
        assert output.related_grammar == ["definite article", "indefinite article"]
        assert output.common_errors == ["omitting articles", "using wrong article"]

    def test_invalid_cefr_level(self) -> None:
        with pytest.raises(ValidationError):
            MCQExplanationOutputSchema(
                question_analysis="Analysis",
                option_explanations=[
                    MCQDistractorExplanationSchema(
                        option_letter="A",
                        option_text="A",
                        is_correct=True,
                        explanation="Correct",
                    ),
                    MCQDistractorExplanationSchema(
                        option_letter="B",
                        option_text="B",
                        is_correct=False,
                        explanation="No",
                    ),
                    MCQDistractorExplanationSchema(
                        option_letter="C",
                        option_text="C",
                        is_correct=False,
                        explanation="No",
                    ),
                    MCQDistractorExplanationSchema(
                        option_letter="D",
                        option_text="D",
                        is_correct=False,
                        explanation="No",
                    ),
                ],
                teaching_tip="Tip",
                cefr_level="D2",  # Invalid
            )


@pytest.mark.unit
class TestMCQExplanationInputSchema:
    """Tests for MCQExplanationInputSchema."""

    def test_valid_input(self) -> None:
        inp = MCQExplanationInputSchema(
            question_text="What is the answer?",
            topic="article usage",
            option_a="opt1",
            option_b="opt2",
            option_c="opt3",
            option_d="opt4",
            correct_answer="A",
            chunk_content="Chinese L1 students",
        )
        assert inp.correct_answer == "A"
        assert inp.option_a == "opt1"

    def test_invalid_correct_answer(self) -> None:
        with pytest.raises(ValidationError):
            MCQExplanationInputSchema(
                question_text="Q?",
                topic="grammar",
                option_a="a",
                option_b="b",
                option_c="c",
                option_d="d",
                correct_answer="E",  # Invalid
                chunk_content="Students",
            )


@pytest.mark.unit
class TestMCQAnswerInputSchema:
    """Tests for MCQAnswerInputSchema."""

    def test_valid_input(self) -> None:
        inp = MCQAnswerInputSchema(
            question_text="The cat sat on ___ mat.",
            topic="article usage",
            difficulty="medium",
            chunk_content="Chinese",
        )
        assert inp.chunk_content == "Chinese"
        assert inp.topic == "article usage"

    def test_invalid_difficulty(self) -> None:
        with pytest.raises(ValidationError):
            MCQAnswerInputSchema(
                question_text="Question?",
                topic="tense",
                difficulty="expert",  # Invalid
                chunk_content="Mixed",
            )


@pytest.mark.unit
class TestMCQAnswerGeneratorOutputSchema:
    """Tests for MCQAnswerGeneratorOutputSchema."""

    def test_valid_output(self) -> None:
        correct = MCQDistractorExplanationSchema(
            option_letter="A",
            option_text="the",
            is_correct=True,
            explanation="Correct article for specific reference",
        )
        distractors = [
            MCQDistractorExplanationSchema(
                option_letter="B",
                option_text="a",
                is_correct=False,
                explanation="Incorrect - not indefinite",
                l1_interference_note="Chinese L1: articles often omitted",
            ),
            MCQDistractorExplanationSchema(
                option_letter="C",
                option_text="an",
                is_correct=False,
                explanation="Wrong article for consonant sound",
            ),
            MCQDistractorExplanationSchema(
                option_letter="D",
                option_text="(no article)",
                is_correct=False,
                explanation="Article required here",
            ),
        ]
        output = MCQAnswerGeneratorOutputSchema(
            question_stem="The cat sat on ___ mat.",
            correct_answer=correct,
            distractors=distractors,
            grammar_point_tested="definite article usage",
            difficulty_justification="Medium: requires understanding of specific vs general",
            l1_considerations=[
                "Chinese L1 often omits articles",
                "No article in Chinese grammar",
            ],
        )
        assert len(output.distractors) == 3
        assert output.grammar_point_tested == "definite article usage"


@pytest.mark.unit
class TestOptionalMetadataFields:
    """Tests for optional metadata fields in output schemas."""

    def test_mcq_answer_with_optional_fields_absent(self) -> None:
        """MCQAnswerGeneratorOutputSchema accepts missing optional fields."""
        output = MCQAnswerGeneratorOutputSchema(
            question_stem="Test?",
            correct_answer=MCQDistractorExplanationSchema(
                option_letter="A",
                option_text="Correct",
                is_correct=True,
                explanation="Correct answer",
            ),
            distractors=[
                MCQDistractorExplanationSchema(
                    option_letter="B",
                    option_text="Wrong",
                    is_correct=False,
                    explanation="Wrong answer",
                ),
                MCQDistractorExplanationSchema(
                    option_letter="C",
                    option_text="Wrong",
                    is_correct=False,
                    explanation="Wrong answer",
                ),
                MCQDistractorExplanationSchema(
                    option_letter="D",
                    option_text="Wrong",
                    is_correct=False,
                    explanation="Wrong answer",
                ),
            ],
            grammar_point_tested="test",
            difficulty_justification="test",
        )
        assert output.grammar_points == []
        assert output.cefr_level is None
        assert output.l1_considerations == []

    def test_mcq_explanation_with_optional_fields_absent(self) -> None:
        """MCQExplanationOutputSchema accepts missing optional fields."""
        output = MCQExplanationOutputSchema(
            question_analysis="Analysis",
            option_explanations=[
                MCQDistractorExplanationSchema(
                    option_letter="A",
                    option_text="A",
                    is_correct=True,
                    explanation="Correct",
                ),
                MCQDistractorExplanationSchema(
                    option_letter="B",
                    option_text="B",
                    is_correct=False,
                    explanation="No",
                ),
                MCQDistractorExplanationSchema(
                    option_letter="C",
                    option_text="C",
                    is_correct=False,
                    explanation="No",
                ),
                MCQDistractorExplanationSchema(
                    option_letter="D",
                    option_text="D",
                    is_correct=False,
                    explanation="No",
                ),
            ],
            teaching_tip="Tip",
        )
        assert output.cefr_level is None
        assert output.related_grammar == []
        assert output.common_errors == []

    def test_mcq_explanation_with_cefr_level(self) -> None:
        """MCQExplanationOutputSchema accepts valid CEFR level."""
        output = MCQExplanationOutputSchema(
            question_analysis="Analysis",
            option_explanations=[
                MCQDistractorExplanationSchema(
                    option_letter="A",
                    option_text="A",
                    is_correct=True,
                    explanation="Correct",
                ),
                MCQDistractorExplanationSchema(
                    option_letter="B",
                    option_text="B",
                    is_correct=False,
                    explanation="No",
                ),
                MCQDistractorExplanationSchema(
                    option_letter="C",
                    option_text="C",
                    is_correct=False,
                    explanation="No",
                ),
                MCQDistractorExplanationSchema(
                    option_letter="D",
                    option_text="D",
                    is_correct=False,
                    explanation="No",
                ),
            ],
            teaching_tip="Tip",
            cefr_level="B1",
        )
        assert output.cefr_level == "B1"
