"""Unit tests for value objects."""

from __future__ import annotations

import pytest

from qna_generation_agent.domain.errors import ValidationError
from qna_generation_agent.domain.value_objects import AnswerId, ContentHash, QuestionId


class TestQuestionId:
    """Tests for QuestionId."""

    @pytest.mark.unit
    def test_valid_id(self) -> None:
        """Test that valid ID is accepted."""
        qid = QuestionId("q_test_123")
        assert qid.value == "q_test_123"

    @pytest.mark.unit
    def test_empty_id_raises_error(self) -> None:
        """Test that empty ID raises error."""
        with pytest.raises(ValidationError):
            QuestionId("")

    @pytest.mark.unit
    def test_whitespace_id_raises_error(self) -> None:
        """Test that whitespace-only ID raises error."""
        with pytest.raises(ValidationError):
            QuestionId("   ")

    @pytest.mark.unit
    def test_generate_creates_valid_id(self) -> None:
        """Test that generate creates valid ID."""
        qid = QuestionId.generate()
        assert qid.value.startswith("q_")
        assert len(qid.value) > 2

    @pytest.mark.unit
    def test_generate_with_custom_prefix(self) -> None:
        """Test that generate accepts custom prefix."""
        qid = QuestionId.generate(prefix="question")
        assert qid.value.startswith("question_")


class TestAnswerId:
    """Tests for AnswerId."""

    @pytest.mark.unit
    def test_valid_id(self) -> None:
        """Test that valid ID is accepted."""
        aid = AnswerId("a_test_123")
        assert aid.value == "a_test_123"

    @pytest.mark.unit
    def test_empty_id_raises_error(self) -> None:
        """Test that empty ID raises error."""
        with pytest.raises(ValidationError):
            AnswerId("")

    @pytest.mark.unit
    def test_whitespace_id_raises_error(self) -> None:
        """Test that whitespace-only ID raises error."""
        with pytest.raises(ValidationError):
            AnswerId("   ")

    @pytest.mark.unit
    def test_generate_creates_valid_id(self) -> None:
        """Test that generate creates valid ID."""
        aid = AnswerId.generate()
        assert aid.value.startswith("a_")
        assert len(aid.value) > 2

    @pytest.mark.unit
    def test_generate_with_custom_prefix(self) -> None:
        """Test that generate accepts custom prefix."""
        aid = AnswerId.generate(prefix="answer")
        assert aid.value.startswith("answer_")


class TestContentHash:
    """Tests for ContentHash."""

    @pytest.mark.unit
    def test_valid_sha256(self) -> None:
        """Test that valid SHA256 hash is accepted."""
        hash_value = "a" * 64  # SHA256 is 64 hex chars
        ch = ContentHash(algorithm="sha256", value=hash_value)
        assert ch.algorithm == "sha256"
        assert ch.value == hash_value

    @pytest.mark.unit
    def test_valid_md5(self) -> None:
        """Test that valid MD5 hash is accepted."""
        hash_value = "a" * 32  # MD5 is 32 hex chars
        ch = ContentHash(algorithm="md5", value=hash_value)
        assert ch.algorithm == "md5"

    @pytest.mark.unit
    def test_valid_blake2b(self) -> None:
        """Test that valid BLAKE2b hash is accepted."""
        hash_value = "a" * 64
        ch = ContentHash(algorithm="blake2b", value=hash_value)
        assert ch.algorithm == "blake2b"

    @pytest.mark.unit
    def test_empty_value_raises_error(self) -> None:
        """Test that empty value raises error."""
        with pytest.raises(ValidationError):
            ContentHash(algorithm="sha256", value="")

    @pytest.mark.unit
    def test_whitespace_value_raises_error(self) -> None:
        """Test that whitespace-only value raises error."""
        with pytest.raises(ValidationError):
            ContentHash(algorithm="sha256", value="   ")

    @pytest.mark.unit
    def test_unsupported_algorithm_raises_error(self) -> None:
        """Test that unsupported algorithm raises error."""
        with pytest.raises(ValidationError):
            ContentHash(algorithm="sha1", value="abc123")

    @pytest.mark.unit
    def test_from_content_creates_hash(self) -> None:
        """Test that from_content creates hash."""
        content = "test content"
        ch = ContentHash.from_content(content, algorithm="sha256")
        assert ch.algorithm == "sha256"
        assert len(ch.value) == 64  # SHA256 hex length

    @pytest.mark.unit
    def test_from_content_with_md5(self) -> None:
        """Test that from_content creates MD5 hash."""
        content = "test content"
        ch = ContentHash.from_content(content, algorithm="md5")
        assert ch.algorithm == "md5"
        assert len(ch.value) == 32  # MD5 hex length

    @pytest.mark.unit
    def test_from_content_with_blake2b(self) -> None:
        """Test that from_content creates BLAKE2b hash."""
        content = "test content"
        ch = ContentHash.from_content(content, algorithm="blake2b")
        assert ch.algorithm == "blake2b"

    @pytest.mark.unit
    def test_from_content_unsupported_algorithm(self) -> None:
        """Test that unsupported algorithm raises error."""
        with pytest.raises(ValidationError):
            ContentHash.from_content("content", algorithm="sha1")

    @pytest.mark.unit
    def test_equality(self) -> None:
        """Test that equality works correctly."""
        ch1 = ContentHash(algorithm="sha256", value="abc")
        ch2 = ContentHash(algorithm="sha256", value="abc")
        ch3 = ContentHash(algorithm="sha256", value="def")
        ch4 = ContentHash(algorithm="md5", value="abc")

        assert ch1 == ch2
        assert ch1 != ch3
        assert ch1 != ch4
        assert ch1 != "not a ContentHash"

    @pytest.mark.unit
    def test_hash(self) -> None:
        """Test that hash works correctly."""
        ch1 = ContentHash(algorithm="sha256", value="abc")
        ch2 = ContentHash(algorithm="sha256", value="abc")

        assert hash(ch1) == hash(ch2)

    @pytest.mark.unit
    def test_can_be_used_as_dict_key(self) -> None:
        """Test that ContentHash can be used as dict key."""
        ch1 = ContentHash(algorithm="sha256", value="abc")
        ch2 = ContentHash(algorithm="sha256", value="def")

        d = {ch1: "value1", ch2: "value2"}

        assert d[ch1] == "value1"
        assert d[ch2] == "value2"
