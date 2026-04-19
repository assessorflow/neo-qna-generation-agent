"""Unit tests for HTTP health routes.

These tests invoke the actual route handler functions directly rather than
merely asserting that routes were registered on the router.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from blacksheep import Application
from blacksheep.server.routing import Router

from qna_generation_agent import __version__
from qna_generation_agent.app.settings import RuntimeEnvironment, Settings
from qna_generation_agent.interfaces.http.routes_health import (
    _is_ready_from_checks,
    register_health_routes,
)


def _get_handler(app: Application, pattern: bytes) -> object:
    """Extract route handler by pattern from registered routes."""
    for _method, route in app.router.registered_routes:
        if route.pattern == pattern:
            return route.handler
    raise ValueError(f"Route {pattern!r} not found")


def _make_settings(
    *,
    environment: RuntimeEnvironment = RuntimeEnvironment.LOCAL,
    worker_ready: bool = False,
    langfuse_enabled: bool = False,
) -> Settings:
    """Create a minimal settings object for tests."""
    kwargs: dict[str, object] = {
        "environment": environment,
        "host": "127.0.0.1",
        "port": 8000,
        "workers": 1,
        "llm_api_key": "test-key",
        "model_id": "gpt-4o-mini",
        "llm_base_url": "https://api.openai.com/v1",
        "submission_service_url": "localhost:50052",
        "knowledge_service_url": "localhost:9030",
    }
    if worker_ready:
        kwargs.update(
            pubsub_project_id="test-project",
            pubsub_subscription_trigger="test-sub",
            pubsub_topic_complete="test-topic",
        )
    if langfuse_enabled:
        kwargs.update(
            langfuse_public_key="pk",
            langfuse_secret_key="sk",
        )
    return Settings.model_construct(**kwargs)


def _make_container(
    settings: Settings | None = None, **overrides: object
) -> MagicMock:
    """Create a mock container with sensible defaults."""
    container = MagicMock()
    container.settings = settings or _make_settings()
    container.subscriber = None
    container.llm_provider = None
    container.knowledge_client = None
    container.submission_client = None
    container.telemetry = None
    container.prompt_provider = None
    for key, value in overrides.items():
        setattr(container, key, value)
    return container


class TestHealthzEndpoint:
    """Tests for /healthz."""

    @pytest.mark.unit
    async def test_returns_200_when_healthy(self) -> None:
        """healthz returns 200 with status=healthy when runtime is healthy."""
        app = Application(router=Router())
        register_health_routes(app)
        handler = _get_handler(app, b"/healthz")

        state = MagicMock()
        state.is_healthy.return_value = True

        with patch(
            "qna_generation_agent.interfaces.http.routes_health.get_runtime_state",
            return_value=state,
        ):
            response = await handler(_make_container())

        assert response.status == 200
        payload = await response.json()
        assert payload["status"] == "healthy"

    @pytest.mark.unit
    async def test_returns_503_when_unhealthy(self) -> None:
        """healthz returns 503 with status=unhealthy when runtime is not healthy."""
        app = Application(router=Router())
        register_health_routes(app)
        handler = _get_handler(app, b"/healthz")

        state = MagicMock()
        state.is_healthy.return_value = False

        with patch(
            "qna_generation_agent.interfaces.http.routes_health.get_runtime_state",
            return_value=state,
        ):
            response = await handler(_make_container())

        assert response.status == 503
        payload = await response.json()
        assert payload["status"] == "unhealthy"

    @pytest.mark.unit
    async def test_fallback_healthy_when_all_deps_present(self) -> None:
        """healthz fallback returns 200 when container has all essential deps."""
        app = Application(router=Router())
        register_health_routes(app)
        handler = _get_handler(app, b"/healthz")

        container = _make_container(
            llm_provider=MagicMock(),
            submission_client=MagicMock(),
            knowledge_client=MagicMock(),
        )

        with patch(
            "qna_generation_agent.interfaces.http.routes_health.get_runtime_state",
            side_effect=RuntimeError("not initialized"),
        ):
            response = await handler(container)

        assert response.status == 200
        payload = await response.json()
        assert payload["status"] == "healthy"

    @pytest.mark.unit
    async def test_fallback_initializing_when_deps_missing(self) -> None:
        """healthz fallback returns 503 when container lacks essential deps."""
        app = Application(router=Router())
        register_health_routes(app)
        handler = _get_handler(app, b"/healthz")

        container = _make_container(llm_provider=MagicMock())

        with patch(
            "qna_generation_agent.interfaces.http.routes_health.get_runtime_state",
            side_effect=RuntimeError("not initialized"),
        ):
            response = await handler(container)

        assert response.status == 503
        payload = await response.json()
        assert payload["status"] == "initializing"


class TestLivezEndpoint:
    """Tests for /livez."""

    @pytest.mark.unit
    async def test_returns_200_when_alive(self) -> None:
        """livez returns 200 with alive=true when runtime is alive."""
        app = Application(router=Router())
        register_health_routes(app)
        handler = _get_handler(app, b"/livez")

        state = MagicMock()
        state.is_live.return_value = True

        with patch(
            "qna_generation_agent.interfaces.http.routes_health.get_runtime_state",
            return_value=state,
        ):
            response = await handler(_make_container())

        assert response.status == 200
        payload = await response.json()
        assert payload["alive"] is True

    @pytest.mark.unit
    async def test_returns_503_when_not_alive(self) -> None:
        """livez returns 503 with alive=false when runtime is not alive."""
        app = Application(router=Router())
        register_health_routes(app)
        handler = _get_handler(app, b"/livez")

        state = MagicMock()
        state.is_live.return_value = False

        with patch(
            "qna_generation_agent.interfaces.http.routes_health.get_runtime_state",
            return_value=state,
        ):
            response = await handler(_make_container())

        assert response.status == 503
        payload = await response.json()
        assert payload["alive"] is False

    @pytest.mark.unit
    async def test_fallback_alive_when_lifespan_not_initialized(self) -> None:
        """livez fallback returns 200 when lifespan has not been entered."""
        app = Application(router=Router())
        register_health_routes(app)
        handler = _get_handler(app, b"/livez")

        with patch(
            "qna_generation_agent.interfaces.http.routes_health.get_runtime_state",
            side_effect=RuntimeError("not initialized"),
        ):
            response = await handler(_make_container())

        assert response.status == 200
        payload = await response.json()
        assert payload["alive"] is True


class TestReadyzEndpoint:
    """Tests for /readyz."""

    @pytest.mark.unit
    async def test_returns_200_when_ready(self) -> None:
        """readyz returns 200 with ready=true when runtime is ready."""
        app = Application(router=Router())
        register_health_routes(app)
        handler = _get_handler(app, b"/readyz")

        state = MagicMock()
        state.lifecycle.value = "running"
        state.subscriber_running = True
        state.llm_healthy = True
        state.knowledge_service_healthy = True
        state.submission_service_healthy = True
        state.telemetry_healthy = True
        state.prompt_provider_healthy = True
        state.is_ready.return_value = True

        with patch(
            "qna_generation_agent.interfaces.http.routes_health.get_runtime_state",
            return_value=state,
        ):
            response = await handler(_make_container())

        assert response.status == 200
        payload = await response.json()
        assert payload["ready"] is True
        assert "checks" in payload

    @pytest.mark.unit
    async def test_returns_503_when_not_ready(self) -> None:
        """readyz returns 503 with ready=false and detailed checks dict."""
        app = Application(router=Router())
        register_health_routes(app)
        handler = _get_handler(app, b"/readyz")

        state = MagicMock()
        state.lifecycle.value = "running"
        state.subscriber_running = False
        state.llm_healthy = False
        state.knowledge_service_healthy = True
        state.submission_service_healthy = True
        state.telemetry_healthy = True
        state.prompt_provider_healthy = True
        state.is_ready.return_value = False

        with patch(
            "qna_generation_agent.interfaces.http.routes_health.get_runtime_state",
            return_value=state,
        ):
            response = await handler(_make_container())

        assert response.status == 503
        payload = await response.json()
        assert payload["ready"] is False
        checks = payload["checks"]
        assert checks["startup_complete"] is True
        assert checks["subscriber_running"] is False
        assert checks["llm_healthy"] is False

    @pytest.mark.unit
    async def test_checks_include_telemetry_configured(self) -> None:
        """readyz checks dict includes telemetry_configured flag."""
        app = Application(router=Router())
        register_health_routes(app)
        handler = _get_handler(app, b"/readyz")

        state = MagicMock()
        state.lifecycle.value = "running"
        state.subscriber_running = True
        state.llm_healthy = True
        state.knowledge_service_healthy = True
        state.submission_service_healthy = True
        state.telemetry_healthy = True
        state.prompt_provider_healthy = True
        state.is_ready.return_value = True

        with patch(
            "qna_generation_agent.interfaces.http.routes_health.get_runtime_state",
            return_value=state,
        ):
            response = await handler(_make_container())

        payload = await response.json()
        assert "telemetry_configured" in payload["checks"]

    @pytest.mark.unit
    async def test_fallback_uses_container_health_checks(self) -> None:
        """readyz fallback runs fallback health checks when lifespan not initialized."""
        app = Application(router=Router())
        register_health_routes(app)
        handler = _get_handler(app, b"/readyz")

        llm = MagicMock()
        llm.health_check = AsyncMock(return_value=True)
        knowledge = MagicMock()
        knowledge.health_check = AsyncMock(return_value=True)
        submission = MagicMock()
        submission.health_check = AsyncMock(return_value=True)

        container = _make_container(
            settings=_make_settings(worker_ready=True),
            llm_provider=llm,
            knowledge_client=knowledge,
            submission_client=submission,
            subscriber=MagicMock(),
        )

        with patch(
            "qna_generation_agent.interfaces.http.routes_health.get_runtime_state",
            side_effect=RuntimeError("not initialized"),
        ):
            response = await handler(container)

        payload = await response.json()
        checks = payload["checks"]
        assert checks["llm_healthy"] is True
        assert checks["knowledge_service_healthy"] is True
        assert checks["submission_service_healthy"] is True
        assert checks["subscriber_running"] is True
        assert checks["startup_complete"] is True


class TestVersionEndpoint:
    """Tests for /version."""

    @pytest.mark.unit
    async def test_returns_version_and_environment(self) -> None:
        """version returns version string and environment."""
        app = Application(router=Router())
        register_health_routes(app)
        handler = _get_handler(app, b"/version")

        settings = _make_settings(environment=RuntimeEnvironment.DEV)
        container = _make_container(settings=settings)

        response = await handler(container)

        assert response.status == 200
        payload = await response.json()
        assert payload["version"] == __version__
        assert payload["environment"] == "dev"


class TestIsReadyFromChecks:
    """Direct tests for _is_ready_from_checks logic."""

    @pytest.mark.unit
    def test_all_pass_returns_true(self) -> None:
        """All checks pass → ready."""
        settings = _make_settings(worker_ready=True)
        checks: dict[str, bool] = {
            "startup_complete": True,
            "subscriber_running": True,
            "llm_healthy": True,
            "knowledge_service_healthy": True,
            "submission_service_healthy": True,
        }
        assert _is_ready_from_checks(settings, checks) is True

    @pytest.mark.unit
    def test_missing_startup_complete_returns_false(self) -> None:
        """Missing startup_complete → not ready."""
        settings = _make_settings(worker_ready=True)
        checks: dict[str, bool] = {
            "startup_complete": False,
            "subscriber_running": True,
            "llm_healthy": True,
            "knowledge_service_healthy": True,
            "submission_service_healthy": True,
        }
        assert _is_ready_from_checks(settings, checks) is False

    @pytest.mark.unit
    def test_worker_ready_requires_subscriber(self) -> None:
        """worker_ready=True requires subscriber_running."""
        settings = _make_settings(worker_ready=True)
        checks: dict[str, bool] = {
            "startup_complete": True,
            "subscriber_running": False,
            "llm_healthy": True,
            "knowledge_service_healthy": True,
            "submission_service_healthy": True,
        }
        assert _is_ready_from_checks(settings, checks) is False

    @pytest.mark.unit
    def test_llm_unhealthy_returns_false(self) -> None:
        """LLM unhealthy → not ready regardless of other checks."""
        settings = _make_settings(worker_ready=False)
        checks: dict[str, bool] = {
            "startup_complete": True,
            "subscriber_running": True,
            "llm_healthy": False,
            "knowledge_service_healthy": True,
            "submission_service_healthy": True,
        }
        assert _is_ready_from_checks(settings, checks) is False

    @pytest.mark.unit
    def test_worker_ready_requires_knowledge_service(self) -> None:
        """worker_ready=True requires knowledge_service_healthy."""
        settings = _make_settings(worker_ready=True)
        checks: dict[str, bool] = {
            "startup_complete": True,
            "subscriber_running": True,
            "llm_healthy": True,
            "knowledge_service_healthy": False,
            "submission_service_healthy": True,
        }
        assert _is_ready_from_checks(settings, checks) is False

    @pytest.mark.unit
    def test_worker_ready_requires_submission_service(self) -> None:
        """worker_ready=True requires submission_service_healthy."""
        settings = _make_settings(worker_ready=True)
        checks: dict[str, bool] = {
            "startup_complete": True,
            "subscriber_running": True,
            "llm_healthy": True,
            "knowledge_service_healthy": True,
            "submission_service_healthy": False,
        }
        assert _is_ready_from_checks(settings, checks) is False

    @pytest.mark.unit
    def test_non_worker_mode_ignores_subscriber_and_grpc(self) -> None:
        """Non-worker mode ignores subscriber and gRPC checks."""
        settings = _make_settings(worker_ready=False)
        checks: dict[str, bool] = {
            "startup_complete": True,
            "subscriber_running": False,
            "llm_healthy": True,
            "knowledge_service_healthy": False,
            "submission_service_healthy": False,
        }
        assert _is_ready_from_checks(settings, checks) is True
