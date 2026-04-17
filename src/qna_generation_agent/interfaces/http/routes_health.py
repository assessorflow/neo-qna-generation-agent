"""Health and version routes with OpenAPI metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING

from blacksheep import Application, Response
from blacksheep.server.responses import json

from qna_generation_agent import __version__
from qna_generation_agent.app.bootstrap import ApplicationContainer
from qna_generation_agent.app.lifespan import get_runtime_state
from qna_generation_agent.app.logging import get_logger
from qna_generation_agent.interfaces.http.schemas import (
    HealthResponse,
    LiveResponse,
    ReadyResponse,
    VersionResponse,
)

if TYPE_CHECKING:
    from qna_generation_agent.app.settings import Settings

logger = get_logger(__name__)


async def _run_fallback_health_checks(
    container: ApplicationContainer,
) -> dict[str, bool]:
    """Run health checks directly on container when runtime state is not available.

    This is used in test mode when the ASGI lifespan context is not entered.
    """
    checks: dict[str, bool] = {
        "startup_complete": True,  # Container is initialized
        "subscriber_running": container.subscriber is not None,
        "llm_healthy": False,
        "knowledge_service_healthy": False,
        "submission_service_healthy": False,
        "telemetry_healthy": True,  # Optional
        "prompt_provider_healthy": True,  # Optional
    }

    # Check LLM
    if container.llm_provider is not None:
        try:
            if hasattr(container.llm_provider, "health_check"):
                checks["llm_healthy"] = await container.llm_provider.health_check()
        except Exception as error:
            logger.warning("fallback_llm_health_check_failed", error=str(error))
            checks["llm_healthy"] = False

    # Check Knowledge Service
    if container.knowledge_client is not None:
        try:
            if hasattr(container.knowledge_client, "health_check"):
                checks[
                    "knowledge_service_healthy"
                ] = await container.knowledge_client.health_check()
        except Exception as error:
            logger.warning("fallback_knowledge_health_check_failed", error=str(error))
            checks["knowledge_service_healthy"] = False

    # Check Submission Service
    if container.submission_client is not None:
        try:
            if hasattr(container.submission_client, "health_check"):
                checks[
                    "submission_service_healthy"
                ] = await container.submission_client.health_check()
        except Exception as error:
            logger.warning("fallback_submission_health_check_failed", error=str(error))
            checks["submission_service_healthy"] = False

    # Check Telemetry
    if container.telemetry is not None:
        try:
            if hasattr(container.telemetry, "health_check"):
                checks["telemetry_healthy"] = await container.telemetry.health_check()
        except Exception as error:
            logger.warning("fallback_telemetry_health_check_failed", error=str(error))
            checks["telemetry_healthy"] = False

    return checks


def _is_ready_from_checks(settings: Settings, checks: dict[str, bool]) -> bool:
    """Determine readiness from health check results."""
    # Startup must be complete
    if not checks.get("startup_complete", False):
        return False

    # In worker mode, subscriber must be running
    if settings.worker_ready and not checks.get("subscriber_running", False):
        return False

    # LLM is always required
    if not checks.get("llm_healthy", False):
        return False

    # gRPC services are required in worker mode
    if settings.worker_ready:
        if not checks.get("knowledge_service_healthy", False):
            return False
        if not checks.get("submission_service_healthy", False):
            return False

    return True


def register_health_routes(app: Application) -> None:
    """Register the operational HTTP routes."""

    @app.router.get("/healthz")
    async def healthz(container: ApplicationContainer) -> Response:
        """Health check endpoint.

        Returns the current health status of the service based on lifecycle state.
        Returns HTTP 200 when healthy, HTTP 503 when not healthy or still starting.
        """
        try:
            state = get_runtime_state()
            is_healthy = state.is_healthy()
            return json(
                HealthResponse(status="healthy" if is_healthy else "unhealthy"),
                status=200 if is_healthy else 503,
            )
        except RuntimeError:
            # Lifespan not initialized yet - use fallback
            # In this mode, we check if container has essential components
            is_healthy = (
                container.llm_provider is not None
                and container.submission_client is not None
                and container.knowledge_client is not None
            )
            return json(
                HealthResponse(status="healthy" if is_healthy else "initializing"),
                status=200 if is_healthy else 503,
            )

    @app.router.get("/livez")
    async def livez(container: ApplicationContainer) -> Response:
        """Liveness probe endpoint.

        Kubernetes liveness probe. Returns alive status based on lifecycle state.
        Returns HTTP 200 when alive, HTTP 503 when shutting down or failed.
        """
        try:
            state = get_runtime_state()
            is_live = state.is_live()
            return json(
                LiveResponse(alive=is_live),
                status=200 if is_live else 503,
            )
        except RuntimeError:
            # Lifespan not initialized yet - still considered alive
            return json(
                LiveResponse(alive=True),
                status=200,
            )

    @app.router.get("/readyz")
    async def readyz(container: ApplicationContainer) -> Response:
        """Readiness probe endpoint.

        Kubernetes readiness probe. Checks if the service is ready to accept traffic
        by verifying connectivity to LLM provider, Knowledge Service,
        Submission Service, and telemetry via runtime state. Returns HTTP 200 when ready,
        HTTP 503 when not ready.
        """
        settings = container.settings

        try:
            state = get_runtime_state()

            # Build detailed checks from runtime state
            checks = {
                "startup_complete": state.lifecycle.value == "running",
                "subscriber_running": state.subscriber_running,
                "llm_healthy": state.llm_healthy,
                "knowledge_service_healthy": state.knowledge_service_healthy,
                "submission_service_healthy": state.submission_service_healthy,
                "telemetry_healthy": state.telemetry_healthy,
                "prompt_provider_healthy": state.prompt_provider_healthy,
                # Config presence checks for context
                "telemetry_configured": (
                    not settings.langfuse_enabled or container.telemetry is not None
                ),
            }

            is_ready = state.is_ready(settings)
        except RuntimeError:
            # Lifespan not initialized - use fallback health checks
            checks = await _run_fallback_health_checks(container)
            checks["telemetry_configured"] = (
                not settings.langfuse_enabled or container.telemetry is not None
            )
            is_ready = _is_ready_from_checks(settings, checks)

        return json(
            ReadyResponse(ready=is_ready, checks=checks),
            status=200 if is_ready else 503,
        )

    @app.router.get("/version")
    async def version_endpoint(container: ApplicationContainer) -> Response:
        """Version endpoint.

        Returns the application version and current environment.
        """
        return json(
            VersionResponse(
                version=__version__,
                environment=container.settings.environment.value,
            ).model_dump(mode="json")
        )
