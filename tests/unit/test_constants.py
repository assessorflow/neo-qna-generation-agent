"""Unit tests for constants module."""

from __future__ import annotations

import pytest

from qna_generation_agent.app import constants


@pytest.mark.unit
class TestConstantsValues:
    """Tests for constant values."""

    def test_grpc_timeout_positive(self) -> None:
        """gRPC timeout should be positive."""
        assert constants.GRPC_DEFAULT_TIMEOUT_SECONDS > 0

    def test_llm_timeout_positive(self) -> None:
        """LLM timeout should be positive."""
        assert constants.LLM_DEFAULT_TIMEOUT_SECONDS > 0

    def test_retry_values_consistent(self) -> None:
        """Retry configuration should be consistent."""
        assert constants.MAX_RETRY_ATTEMPTS >= 1
        assert constants.RETRY_BASE_DELAY_SECONDS > 0
        assert constants.RETRY_MAX_DELAY_SECONDS > constants.RETRY_BASE_DELAY_SECONDS
        assert constants.RETRY_EXPONENTIAL_BASE > 1.0

    def test_domain_constraints_reasonable(self) -> None:
        """Domain constraints should be reasonable values."""
        assert constants.MAX_QUESTION_TEXT_LENGTH > 0
        assert constants.MAX_ANSWER_TEXT_LENGTH > 0
        assert constants.MAX_EXPLANATION_LENGTH > 0
        assert constants.MAX_REFERENCES_COUNT > 0
        assert constants.MAX_METADATA_ITEMS > 0

    def test_service_name_defined(self) -> None:
        """Service name should be defined and non-empty."""
        assert constants.SERVICE_NAME
        assert isinstance(constants.SERVICE_NAME, str)


@pytest.mark.unit
class TestGrpcRetryableCodes:
    """Tests for gRPC retryable status codes."""

    def test_retryable_codes_not_empty(self) -> None:
        """Should have at least some retryable codes defined."""
        assert len(constants.GRPC_RETRYABLE_STATUS_CODES) > 0

    def test_common_retryable_codes_present(self) -> None:
        """Should include common transient error codes."""
        assert "UNAVAILABLE" in constants.GRPC_RETRYABLE_STATUS_CODES
        assert "DEADLINE_EXCEEDED" in constants.GRPC_RETRYABLE_STATUS_CODES
