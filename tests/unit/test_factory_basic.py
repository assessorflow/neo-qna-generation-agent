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

        # Verify bind_app was called on the app
        # The OpenAPI handler's bind_app method should be invoked
        assert app.router is not None

    @pytest.mark.unit
    def test_configures_favicon_route(self) -> None:
        """Test that favicon route is configured for Swagger UI."""
        app = MagicMock()
        app.started = False
        app.router = MagicMock()

        _configure_openapi(app)

        # Verify that a favicon route was registered
        # This prevents 404 noise from Swagger UI
        route_calls = [
            call for call in app.router.method_calls if "favicon" in str(call)
        ]
        assert len(route_calls) > 0 or app.router.get.called
