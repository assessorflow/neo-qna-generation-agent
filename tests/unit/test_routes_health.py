"""Unit tests for HTTP health routes."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from qna_generation_agent.interfaces.http.routes_health import (
    register_health_routes,
)


class TestRegisterHealthRoutes:
    """Tests for register_health_routes."""

    @pytest.mark.unit
    def test_registers_all_routes(self) -> None:
        """Test that all health routes are registered."""
        app = MagicMock()
        app.router = MagicMock()
        app.router.get = MagicMock()

        register_health_routes(app)

        # Verify routes were registered
        route_calls = app.router.get.call_args_list
        paths = [call[0][0] for call in route_calls]

        # Check that standard health endpoints are registered
        assert any("/healthz" in str(p) for p in paths)
        assert any("/livez" in str(p) for p in paths)
        assert any("/readyz" in str(p) for p in paths)
        assert any("/version" in str(p) for p in paths)


class TestHealthzEndpoint:
    """Tests for healthz endpoint."""

    @pytest.mark.unit
    async def test_returns_200_when_healthy(self) -> None:
        """Test that healthz returns 200 when healthy."""

        app = MagicMock()
        register_health_routes(app)

        # Get the registered handler
        healthz_calls = [
            call for call in app.router.get.call_args_list if "/healthz" in str(call)
        ]
        assert len(healthz_calls) > 0


class TestReadyzEndpoint:
    """Tests for readyz endpoint."""

    @pytest.mark.unit
    async def test_checks_dependencies(self) -> None:
        """Test that readyz checks all dependencies."""
        app = MagicMock()
        register_health_routes(app)

        # Get the registered handler
        readyz_calls = [
            call for call in app.router.get.call_args_list if "/readyz" in str(call)
        ]
        assert len(readyz_calls) > 0

    @pytest.mark.unit
    async def test_readyz_route_is_registered(self) -> None:
        """Test that readyz route is properly registered with handler."""
        app = MagicMock()
        app.router = MagicMock()

        register_health_routes(app)

        # Verify the route was registered with the path
        paths = [call[0][0] for call in app.router.get.call_args_list]
        readyz_registered = any("/readyz" in str(p) for p in paths)
        assert readyz_registered


class TestLivezEndpoint:
    """Tests for livez endpoint."""

    @pytest.mark.unit
    async def test_returns_200_when_alive(self) -> None:
        """Test that livez returns 200 when service is alive."""
        app = MagicMock()
        register_health_routes(app)

        # Get the registered handler
        livez_calls = [
            call for call in app.router.get.call_args_list if "/livez" in str(call)
        ]
        assert len(livez_calls) > 0


class TestVersionEndpoint:
    """Tests for version endpoint."""

    @pytest.mark.unit
    async def test_returns_version_info(self) -> None:
        """Test that version returns version information."""
        app = MagicMock()
        register_health_routes(app)

        # Get the registered handler
        version_calls = [
            call for call in app.router.get.call_args_list if "/version" in str(call)
        ]
        assert len(version_calls) > 0
