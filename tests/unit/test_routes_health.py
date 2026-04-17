"""Unit tests for HTTP health routes."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from qna_generation_agent.interfaces.http.routes_health import (
    _run_connectivity_checks,
)


class TestRunConnectivityChecks:
    """Tests for _run_connectivity_checks."""

    @pytest.mark.unit
    async def test_skips_disabled_services(self) -> None:
        """Test that disabled services are skipped."""
        container = MagicMock()
        container.llm_provider = None
        container.knowledge_client = None
        container.submission_client = None
        container.prompt_provider = None

        results = await _run_connectivity_checks(container)

        # All optional services should return True (not required)
        assert results.get("llm") is True
        assert results.get("knowledge") is True
        assert results.get("submission") is True
        assert results.get("prompt_provider") is True

    @pytest.mark.unit
    async def test_handles_timeout(self) -> None:
        """Test that timeout is handled gracefully."""
        container = MagicMock()
        container.llm_provider = MagicMock()
        container.llm_provider.health_check = AsyncMock(side_effect=TimeoutError())
        container.knowledge_client = None
        container.submission_client = None
        container.prompt_provider = None

        results = await _run_connectivity_checks(container)

        assert "llm" in results
