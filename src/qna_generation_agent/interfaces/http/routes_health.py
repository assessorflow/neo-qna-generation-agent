"""Health and version routes with OpenAPI metadata."""

from __future__ import annotations

import asyncio

from blacksheep import Application, Response
from blacksheep.server.responses import json

from qna_generation_agent import __version__
from qna_generation_agent.app.bootstrap import ApplicationContainer
from qna_generation_agent.app.logging import get_logger
from qna_generation_agent.interfaces.http.schemas import (
    HealthResponse,
    LiveResponse,
    ReadyResponse,
    VersionResponse,
)

logger = get_logger(__name__)


def register_health_routes(app: Application) -> None:
    """Register the operational HTTP routes."""

    @app.router.get("/healthz")
    async def healthz() -> Response:
        """Health check endpoint.

        Returns the current health status of the service.
        """
        return json(HealthResponse().model_dump(mode="json"))

    @app.router.get("/livez")
    async def livez() -> Response:
        """Liveness probe endpoint.

        Kubernetes liveness probe. Returns alive status.
        """
        return json(LiveResponse().model_dump(mode="json"))

    @app.router.get("/readyz")
    async def readyz(container: ApplicationContainer) -> Response:
        """Readiness probe endpoint.

        Kubernetes readiness probe. Checks if the service is ready to accept traffic
        by verifying connectivity to LLM provider, Knowledge Service,
        Submission Service, and telemetry. Returns HTTP 200 when ready,
        HTTP 503 when not ready.
        """
        # Run connectivity checks concurrently with a 10-second overall timeout
        check_results = await _run_connectivity_checks(container)

        checks = {
            "settings_loaded": True,
            "telemetry_configured": (not container.settings.langfuse_enabled)
            or (container.telemetry is not None),
            "llm_available": check_results.get("llm", False),
            "subscriber_running": (not container.settings.worker_ready)
            or (container.subscriber is not None),
            "knowledge_service_available": check_results.get("knowledge", False),
            "submission_service_available": check_results.get("submission", False),
            # Prompt provider is optional - service functions without it
            "prompt_provider_available": check_results.get("prompt_provider", True),
        }
        is_ready = all(checks.values())
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


async def _run_connectivity_checks(
    container: ApplicationContainer,
) -> dict[str, bool]:
    """Run all connectivity checks concurrently.

    Returns a dictionary with check results for each dependency.
    """
    checks: dict[str, asyncio.Task[bool]] = {}

    # LLM connectivity check (only for Strands provider with health_check method)
    if container.llm_provider is not None and hasattr(
        container.llm_provider, "health_check"
    ):
        checks["llm"] = asyncio.create_task(
            container.llm_provider.health_check(),
            name="llm_health_check",
        )

    # Knowledge Service connectivity check (only if client has health_check method)
    if container.knowledge_client is not None and hasattr(
        container.knowledge_client, "health_check"
    ):
        checks["knowledge"] = asyncio.create_task(
            container.knowledge_client.health_check(),
            name="knowledge_health_check",
        )

    # Submission Service connectivity check (only if client has health_check method)
    if container.submission_client is not None and hasattr(
        container.submission_client, "health_check"
    ):
        checks["submission"] = asyncio.create_task(
            container.submission_client.health_check(),
            name="submission_health_check",
        )

    # Prompt provider health check (optional - service works without it)
    if container.prompt_provider is not None and hasattr(
        container.prompt_provider, "health_check"
    ):
        checks["prompt_provider"] = asyncio.create_task(
            container.prompt_provider.health_check(),
            name="prompt_provider_health_check",
        )

    # Set defaults for optional/unconfigured checks
    # If a client doesn't have health_check, we assume it's a mock/stub for testing
    llm_has_health = container.llm_provider is not None and hasattr(
        container.llm_provider, "health_check"
    )
    knowledge_has_health = container.knowledge_client is not None and hasattr(
        container.knowledge_client, "health_check"
    )
    submission_has_health = container.submission_client is not None and hasattr(
        container.submission_client, "health_check"
    )

    results: dict[str, bool] = {
        "llm": True
        if not llm_has_health
        else False,  # True if no health check available
        "knowledge": True if not knowledge_has_health else False,
        "submission": True if not submission_has_health else False,
        "prompt_provider": True,  # Optional - always true if not explicitly checked
    }

    # Gather all results with a 10-second timeout
    try:
        async with asyncio.timeout(10.0):
            for name, task in checks.items():
                try:
                    results[name] = await task
                except (TimeoutError, ConnectionError, OSError) as error:
                    logger.warning("health_check_failed", check=name, error=str(error))
                    results[name] = False
    except TimeoutError:
        logger.error("health_checks_timeout")
        # Cancel pending tasks
        for task in checks.values():
            if not task.done():
                task.cancel()
        # Mark unfinished checks as False
        for name in checks:
            if name not in results or results[name] is None:
                results[name] = False

    return results
