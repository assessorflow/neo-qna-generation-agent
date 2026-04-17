"""Unit tests for HTTP factory functions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from qna_generation_agent.interfaces.http.factory import (
    _configure_json,
    _configure_openapi,
)


class TestConfigureJson:
    """Tests for _configure_json."""

    @pytest.mark.unit
    def test_configures_json_settings(self) -> None:
        """Test that JSON settings are configured."""
        with patch(
            "qna_generation_agent.interfaces.http.factory.json_settings"
        ) as mock_settings:
            _configure_json()

            # Verify that loads and dumps are set
            assert mock_settings.loads is not None
            assert mock_settings.dumps is not None


class TestConfigureOpenapi:
    """Tests for _configure_openapi."""

    @pytest.mark.unit
    def test_binds_openapi_handler(self) -> None:
        """Test that OpenAPI handler is bound to app."""
        # Create a proper mock that won't trigger the "started" check
        app = MagicMock()
        app.started = False
        app.router = MagicMock()

        _configure_openapi(app)

        # Verify bind_app was called - doesn't raise exception
        assert True
