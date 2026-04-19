"""Application lifespan management and runtime state tracking.

This module owns startup, shutdown, subscriber state, and readiness state.
It provides a single source of truth for the application's lifecycle state.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from qna_generation_agent.app.logging import get_logger

if TYPE_CHECKING:
    from qna_generation_agent.app.bootstrap import ApplicationContainer
    from qna_generation_agent.app.settings import Settings

logger = get_logger(__name__)


class LifecycleState(StrEnum):
    """Application lifecycle states."""

    INITIALIZING = "initializing"
    STARTUP_COMPLETE = "startup_complete"
    STARTUP_FAILED = "startup_failed"
    RUNNING = "running"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN_COMPLETE = "shutdown_complete"


@dataclass(slots=True)
class RuntimeState:
    """Thread-safe runtime state for health and readiness checks."""

    lifecycle: LifecycleState = field(default=LifecycleState.INITIALIZING)
    subscriber_running: bool = field(default=False)
    startup_error: str | None = field(default=None)

    # Track specific dependency health
    llm_healthy: bool = field(default=False)
    knowledge_service_healthy: bool = field(default=False)
    submission_service_healthy: bool = field(default=False)
    telemetry_healthy: bool = field(default=False)
    prompt_provider_healthy: bool = field(default=False)

    def is_ready(self, settings: Settings) -> bool:
        """Determine if the application is ready to accept traffic.

        Readiness requires:
        - Startup completed successfully
        - Subscriber is running (if worker mode is enabled)
        - All required dependencies are healthy
        """
        if self.lifecycle != LifecycleState.RUNNING:
            return False

        # In worker mode, subscriber must be running
        if settings.worker_ready and not self.subscriber_running:
            return False

        # LLM is always required
        if not self.llm_healthy:
            return False

        # gRPC services are required in worker mode
        if settings.worker_ready:
            if not self.knowledge_service_healthy:
                return False
            if not self.submission_service_healthy:
                return False

        # Telemetry is optional but if configured should be healthy
        # (telemetry_healthy tracks connectivity, not just config)

        return True

    def is_live(self) -> bool:
        """Determine if the application is alive.

        Liveness is a weaker check - the process is alive unless
        it's explicitly shutting down or failed to start.
        """
        return self.lifecycle in {
            LifecycleState.STARTUP_COMPLETE,
            LifecycleState.RUNNING,
            LifecycleState.SHUTTING_DOWN,
        }

    def is_healthy(self) -> bool:
        """Determine if the application is healthy.

        Health requires the application to be running and
        critical dependencies to be available.
        """
        return self.lifecycle == LifecycleState.RUNNING and self.llm_healthy


# Module-level singleton for runtime state
_runtime_state: RuntimeState | None = None
_container: ApplicationContainer | None = None


def get_runtime_state() -> RuntimeState:
    """Get the current runtime state.

    Raises:
        RuntimeError: If lifespan has not been initialized.
    """
    if _runtime_state is None:
        raise RuntimeError("Runtime state not initialized - lifespan not entered")
    return _runtime_state


def get_container() -> ApplicationContainer:
    """Get the initialized application container.

    Raises:
        RuntimeError: If lifespan has not been initialized.
    """
    if _container is None:
        raise RuntimeError("Container not initialized - lifespan not entered")
    return _container


def reset_runtime_state() -> None:
    """Reset the runtime state and container singletons.

    This is primarily for testing to ensure clean state between tests.
    """
    global _runtime_state, _container
    _runtime_state = None
    _container = None


async def _run_health_checks(container: ApplicationContainer) -> RuntimeState:
    """Run health checks on all dependencies and update state."""
    state = get_runtime_state()

    if container.llm_provider is not None:
        try:
            if hasattr(container.llm_provider, "health_check"):
                state.llm_healthy = await container.llm_provider.health_check()
            else:
                # No health check available - assume unhealthy for required deps
                state.llm_healthy = False
                logger.warning("llm_provider_no_health_check")
        except Exception as error:
            logger.warning("llm_health_check_failed", error=str(error))
            state.llm_healthy = False

    if container.knowledge_client is not None:
        try:
            if hasattr(container.knowledge_client, "health_check"):
                state.knowledge_service_healthy = (
                    await container.knowledge_client.health_check()
                )
            else:
                state.knowledge_service_healthy = False
                logger.warning("knowledge_client_no_health_check")
        except Exception as error:
            logger.warning("knowledge_health_check_failed", error=str(error))
            state.knowledge_service_healthy = False

    if container.submission_client is not None:
        try:
            if hasattr(container.submission_client, "health_check"):
                state.submission_service_healthy = (
                    await container.submission_client.health_check()
                )
            else:
                state.submission_service_healthy = False
                logger.warning("submission_client_no_health_check")
        except Exception as error:
            logger.warning("submission_health_check_failed", error=str(error))
            state.submission_service_healthy = False

    if container.telemetry is not None:
        try:
            if hasattr(container.telemetry, "health_check"):
                state.telemetry_healthy = await container.telemetry.health_check()
            else:
                # Telemetry without health_check is OK (e.g., no-op implementations)
                state.telemetry_healthy = True
        except Exception as error:
            logger.warning("telemetry_health_check_failed", error=str(error))
            state.telemetry_healthy = False
    else:
        state.telemetry_healthy = True

    if container.prompt_provider is not None:
        try:
            if hasattr(container.prompt_provider, "health_check"):
                state.prompt_provider_healthy = (
                    await container.prompt_provider.health_check()
                )
            else:
                state.prompt_provider_healthy = True
        except Exception as error:
            logger.warning("prompt_provider_health_check_failed", error=str(error))
            state.prompt_provider_healthy = False
    else:
        state.prompt_provider_healthy = True

    return state


@asynccontextmanager
async def lifespan(container: ApplicationContainer) -> AsyncIterator[RuntimeState]:
    """ASGI lifespan context manager.

    Manages the full application lifecycle:
    1. Initialize runtime state
    2. Start subscriber and run dependency health checks
    3. Yield for application runtime
    4. Graceful shutdown on exit

    Usage:
        async with lifespan(container):
            # Application is running
            pass
    """
    global _runtime_state, _container

    _runtime_state = RuntimeState(lifecycle=LifecycleState.INITIALIZING)
    _container = container

    try:
        logger.info(
            "lifespan_startup_begin",
            subscriber=bool(container.subscriber),
            telemetry=bool(container.telemetry),
        )

        await container.startup()

        if container.subscriber is not None:
            _runtime_state.subscriber_running = True

        await _run_health_checks(container)

        _runtime_state.lifecycle = LifecycleState.RUNNING
        logger.info("lifespan_startup_complete")

        yield _runtime_state

    except Exception as error:
        _runtime_state.lifecycle = LifecycleState.STARTUP_FAILED
        _runtime_state.startup_error = str(error)
        logger.exception("lifespan_startup_failed", error=str(error))
        raise

    finally:
        # Shutdown phase
        if _runtime_state.lifecycle not in {
            LifecycleState.SHUTTING_DOWN,
            LifecycleState.SHUTDOWN_COMPLETE,
        }:
            _runtime_state.lifecycle = LifecycleState.SHUTTING_DOWN
            logger.info("lifespan_shutdown_begin")

            _runtime_state.subscriber_running = False

            try:
                await container.shutdown()
                _runtime_state.lifecycle = LifecycleState.SHUTDOWN_COMPLETE
                logger.info("lifespan_shutdown_complete")
            except Exception:
                logger.exception("lifespan_shutdown_error")
                raise


__all__ = [
    "LifecycleState",
    "RuntimeState",
    "get_container",
    "get_runtime_state",
    "lifespan",
]
