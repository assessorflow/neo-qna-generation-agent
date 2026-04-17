"""Unit tests for app/__init__ and package."""

from __future__ import annotations

import pytest

import qna_generation_agent


class TestPackageVersion:
    """Tests for package version."""

    @pytest.mark.unit
    def test_has_version(self) -> None:
        """Test that package has a version."""
        assert hasattr(qna_generation_agent, "__version__")
        assert qna_generation_agent.__version__ == "0.1.0"
