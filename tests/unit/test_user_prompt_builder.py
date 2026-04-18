"""Unit tests for UserPromptBuilder."""

from __future__ import annotations

import pytest

from qna_generation_agent.application.errors import ValidationError
from qna_generation_agent.infrastructure.llm.user_prompt_builder import (
    UserPromptBuilder,
)


@pytest.mark.unit
class TestUserPromptBuilder:
    """Tests for UserPromptBuilder."""

    @pytest.fixture
    def builder(self) -> UserPromptBuilder:
        return UserPromptBuilder()

    def test_build_assessment_generator_prompt(self, builder: UserPromptBuilder) -> None:
        prompt = builder.build_assessment_generator_prompt(
            structured_count=2,
            non_structured_count=1,
            difficulty="medium",
            topics="grammar, vocabulary",
            chunks="[Chunk 1] Content\n\n[Chunk 2] Content",
        )
        assert "2" in prompt
        assert "1" in prompt
        assert "medium" in prompt
        assert "grammar, vocabulary" in prompt
        assert "[Chunk 1]" in prompt

    def test_build_mcq_answer_generator_prompt(self, builder: UserPromptBuilder) -> None:
        prompt = builder.build_mcq_answer_generator_prompt(
            question_text="She ____ to school.",
            topic="present simple",
            difficulty="easy",
            chunk_content="Chinese L1",
        )
        assert "She ____ to school." in prompt
        assert "present simple" in prompt
        assert "easy" in prompt
        assert "Chinese L1" in prompt

    def test_build_mcq_explanation_generator_prompt(
        self, builder: UserPromptBuilder
    ) -> None:
        prompt = builder.build_mcq_explanation_generator_prompt(
            question_text="She ____ to school.",
            topic="present simple",
            correct_answer="B",
            option_a="walk",
            option_b="walks",
            option_c="walking",
            option_d="walked",
            chunk_content="Chinese L1",
        )
        assert "She ____ to school." in prompt
        assert "B" in prompt
        assert "walk" in prompt
        assert "walks" in prompt
        assert "walking" in prompt
        assert "walked" in prompt

    def test_invalid_difficulty(self, builder: UserPromptBuilder) -> None:
        with pytest.raises(ValidationError):
            builder.build_assessment_generator_prompt(
                structured_count=1,
                non_structured_count=0,
                difficulty="invalid",
                topics="grammar",
                chunks="chunk",
            )

    def test_negative_structured_count(self, builder: UserPromptBuilder) -> None:
        with pytest.raises(ValidationError):
            builder.build_assessment_generator_prompt(
                structured_count=-1,
                non_structured_count=0,
                difficulty="easy",
                topics="grammar",
                chunks="chunk",
            )

    def test_invalid_correct_answer_letter(self, builder: UserPromptBuilder) -> None:
        with pytest.raises(ValidationError):
            builder.build_mcq_explanation_generator_prompt(
                question_text="Q",
                topic="T",
                correct_answer="E",
                option_a="A",
                option_b="B",
                option_c="C",
                option_d="D",
                chunk_content="C",
            )

    def test_keyword_only_args(self, builder: UserPromptBuilder) -> None:
        with pytest.raises(TypeError):
            builder.build_assessment_generator_prompt(1, 0, "easy", "grammar", "chunk")  # type: ignore[misc]
