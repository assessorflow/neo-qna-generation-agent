"""Unit tests for application lifespan and runtime state management."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qna_generation_agent.app.lifespan import (
    LifecycleState,
    RuntimeState,
    get_container,
    get_runtime_state,
    lifespan,
    reset_runtime_state,
)
from qna_generation_agent.app.settings import LogLevel, RuntimeEnvironment, Settings


@pytest.fixture(autouse=True)
def reset_state() -> None:
    """Reset singletons before and after each test."""
    reset_runtime_state()
    yield
    reset_runtime_state()


def _settings(**overrides: Any) -> Settings:
    """Build a Settings instance with sensible test defaults."""
    base: dict[str, Any] = {
        "environment": RuntimeEnvironment.LOCAL,
        "host": "127.0.0.1",
        "port": 8080,
        "workers": 1,
        "log_level": LogLevel.DEBUG,
        "pubsub_project_id": None,
        "pubsub_subscription_trigger": None,
        "pubsub_topic_complete": None,
        "llm_api_key": "test-key",
        "model_id": "gpt-4o-mini",
        "llm_base_url": None,
        "submission_service_url": "localhost:50052",
        "knowledge_service_url": "localhost:9030",
        "release": "test",
    }
    base.update(overrides)
    return Settings.model_construct(**base)


@pytest.mark.unit
class TestRuntimeStateReady:
    """Tests for RuntimeState.is_ready()."""

    def test_runtime_state_is_ready_when_running(self) -> None:
        """RUNNING lifecycle with healthy LLM should be ready."""
        state = RuntimeState(
            lifecycle=LifecycleState.RUNNING,
            llm_healthy=True,
        )
        settings = _settings()
        assert state.is_ready(settings) is True

    def test_runtime_state_not_ready_when_not_running(self) -> None:
        """Non-RUNNING lifecycle should never be ready."""
        settings = _settings()
        for lifecycle in (
            LifecycleState.INITIALIZING,
            LifecycleState.STARTUP_COMPLETE,
            LifecycleState.STARTUP_FAILED,
            LifecycleState.SHUTTING_DOWN,
            LifecycleState.SHUTDOWN_COMPLETE,
        ):
            state = RuntimeState(
                lifecycle=lifecycle,
                llm_healthy=True,
            )
            assert state.is_ready(settings) is False

    def test_runtime_state_not_ready_when_llm_unhealthy(self) -> None:
        """RUNNING lifecycle with unhealthy LLM should not be ready."""
        state = RuntimeState(
            lifecycle=LifecycleState.RUNNING,
            llm_healthy=False,
        )
        settings = _settings()
        assert state.is_ready(settings) is False

    def test_runtime_state_not_ready_worker_mode_subscriber_not_running(self) -> None:
        """Worker mode with subscriber not running should not be ready."""
        state = RuntimeState(
            lifecycle=LifecycleState.RUNNING,
            llm_healthy=True,
            subscriber_running=False,
            knowledge_service_healthy=True,
            submission_service_healthy=True,
        )
        settings = _settings(
            pubsub_project_id="test-project",
            pubsub_subscription_trigger="test-sub",
            pubsub_topic_complete="test-topic",
        )
        assert settings.worker_ready is True
        assert state.is_ready(settings) is False

    def test_runtime_state_not_ready_worker_mode_knowledge_unhealthy(self) -> None:
        """Worker mode with unhealthy knowledge service should not be ready."""
        state = RuntimeState(
            lifecycle=LifecycleState.RUNNING,
            llm_healthy=True,
            subscriber_running=True,
            knowledge_service_healthy=False,
            submission_service_healthy=True,
        )
        settings = _settings(
            pubsub_project_id="test-project",
            pubsub_subscription_trigger="test-sub",
            pubsub_topic_complete="test-topic",
        )
        assert settings.worker_ready is True
        assert state.is_ready(settings) is False

    def test_runtime_state_not_ready_worker_mode_submission_unhealthy(self) -> None:
        """Worker mode with unhealthy submission service should not be ready."""
        state = RuntimeState(
            lifecycle=LifecycleState.RUNNING,
            llm_healthy=True,
            subscriber_running=True,
            knowledge_service_healthy=True,
            submission_service_healthy=False,
        )
        settings = _settings(
            pubsub_project_id="test-project",
            pubsub_subscription_trigger="test-sub",
            pubsub_topic_complete="test-topic",
        )
        assert settings.worker_ready is True
        assert state.is_ready(settings) is False


@pytest.mark.unit
class TestRuntimeStateLive:
    """Tests for RuntimeState.is_live()."""

    def test_runtime_state_is_live_during_startup_and_running(self) -> None:
        """STARTUP_COMPLETE and RUNNING are considered live."""
        for lifecycle in (LifecycleState.STARTUP_COMPLETE, LifecycleState.RUNNING):
            state = RuntimeState(lifecycle=lifecycle)
            assert state.is_live() is True

    def test_runtime_state_is_live_during_shutdown(self) -> None:
        """SHUTTING_DOWN is considered live."""
        state = RuntimeState(lifecycle=LifecycleState.SHUTTING_DOWN)
        assert state.is_live() is True

    def test_runtime_state_not_live_when_shutdown_complete(self) -> None:
        """SHUTDOWN_COMPLETE is not live."""
        state = RuntimeState(lifecycle=LifecycleState.SHUTDOWN_COMPLETE)
        assert state.is_live() is False

    def test_runtime_state_not_live_when_initializing(self) -> None:
        """INITIALIZING is not live."""
        state = RuntimeState(lifecycle=LifecycleState.INITIALIZING)
        assert state.is_live() is False

    def test_runtime_state_not_live_when_startup_failed(self) -> None:
        """STARTUP_FAILED is not live."""
        state = RuntimeState(lifecycle=LifecycleState.STARTUP_FAILED)
        assert state.is_live() is False


@pytest.mark.unit
class TestRuntimeStateHealthy:
    """Tests for RuntimeState.is_healthy()."""

    def test_runtime_state_is_healthy_when_running_and_llm_healthy(self) -> None:
        """RUNNING with healthy LLM should be healthy."""
        state = RuntimeState(
            lifecycle=LifecycleState.RUNNING,
            llm_healthy=True,
        )
        assert state.is_healthy() is True

    def test_runtime_state_not_healthy_when_not_running(self) -> None:
        """Non-RUNNING lifecycle should not be healthy."""
        for lifecycle in (
            LifecycleState.INITIALIZING,
            LifecycleState.STARTUP_COMPLETE,
            LifecycleState.STARTUP_FAILED,
            LifecycleState.SHUTTING_DOWN,
            LifecycleState.SHUTDOWN_COMPLETE,
        ):
            state = RuntimeState(lifecycle=lifecycle, llm_healthy=True)
            assert state.is_healthy() is False

    def test_runtime_state_not_healthy_when_llm_unhealthy(self) -> None:
        """RUNNING with unhealthy LLM should not be healthy."""
        state = RuntimeState(
            lifecycle=LifecycleState.RUNNING,
            llm_healthy=False,
        )
        assert state.is_healthy() is False


@pytest.mark.unit
class TestRuntimeStateSingletons:
    """Tests for get_runtime_state, get_container, and reset_runtime_state."""

    def test_get_runtime_state_raises_when_not_initialized(self) -> None:
        """get_runtime_state should raise when _runtime_state is None."""
        with pytest.raises(RuntimeError, match="Runtime state not initialized"):
            get_runtime_state()

    def test_get_container_raises_when_not_initialized(self) -> None:
        """get_container should raise when _container is None."""
        with pytest.raises(RuntimeError, match="Container not initialized"):
            get_container()

    def test_reset_runtime_state_clears_singletons(self) -> None:
        """reset_runtime_state should set both singletons to None."""
        # Manually set the singletons to verify reset works
        from qna_generation_agent.app import lifespan as lifespan_module

        lifespan_module._runtime_state = RuntimeState()
        lifespan_module._container = MagicMock()

        reset_runtime_state()

        assert lifespan_module._runtime_state is None
        assert lifespan_module._container is None


@pytest.mark.unit
class TestLifespan:
    """Tests for the lifespan context manager."""

    def _mock_container(self, **overrides: Any) -> MagicMock:
        """Create a mocked ApplicationContainer with async lifecycle methods."""
        container = MagicMock()
        container.subscriber = None
        container.llm_provider = None
        container.knowledge_client = None
        container.submission_client = None
        container.telemetry = None
        container.prompt_provider = None
        container.startup = AsyncMock()
        container.shutdown = AsyncMock()
        for key, value in overrides.items():
            setattr(container, key, value)
        return container

    async def test_lifespan_transitions_through_states(self) -> None:
        """Verify lifespan transitions through INITIALIZING, RUNNING, and shutdown."""
        container = self._mock_container()

        with patch(
            "qna_generation_agent.app.lifespan._run_health_checks",
            new_callable=AsyncMock,
        ):
            async with lifespan(container) as state:
                # During runtime, state should be RUNNING
                assert state.lifecycle == LifecycleState.RUNNING
                assert get_runtime_state().lifecycle == LifecycleState.RUNNING

        # After exiting, shutdown should have been called
        container.shutdown.assert_awaited_once()
        # Final state should be SHUTDOWN_COMPLETE
        assert get_runtime_state().lifecycle == LifecycleState.SHUTDOWN_COMPLETE

    async def test_lifespan_sets_subscriber_running(self) -> None:
        """When container has a subscriber, subscriber_running should be True."""
        subscriber = MagicMock()
        container = self._mock_container(subscriber=subscriber)

        with patch(
            "qna_generation_agent.app.lifespan._run_health_checks",
            new_callable=AsyncMock,
        ):
            async with lifespan(container) as state:
                assert state.subscriber_running is True

    async def test_lifespan_handles_startup_failure(self) -> None:
        """When container.startup() raises, lifecycle becomes STARTUP_FAILED."""
        container = self._mock_container()
        container.startup.side_effect = RuntimeError("Startup failed")

        with pytest.raises(RuntimeError, match="Startup failed"):
            async with lifespan(container):
                pass  # pragma: no cover

        # startup_error should be set and preserved through shutdown
        final_state = get_runtime_state()
        assert final_state.startup_error == "Startup failed"
        # Shutdown should still have been attempted
        container.shutdown.assert_awaited_once()

    async def test_lifespan_runs_health_checks(self) -> None:
        """Verify _run_health_checks is called during startup."""
        container = self._mock_container()

        with patch(
            "qna_generation_agent.app.lifespan._run_health_checks",
            new_callable=AsyncMock,
        ) as mock_health_checks:
            async with lifespan(container):
                pass  # pragma: no cover

            mock_health_checks.assert_awaited_once_with(container)

    async def test_lifespan_sets_container_singleton(self) -> None:
        """Verify lifespan sets the _container singleton."""
        container = self._mock_container()

        with patch(
            "qna_generation_agent.app.lifespan._run_health_checks",
            new_callable=AsyncMock,
        ):
            async with lifespan(container):
                assert get_container() is container

    async def test_lifespan_sets_runtime_state_singleton(self) -> None:
        """Verify lifespan sets the _runtime_state singleton."""
        container = self._mock_container()

        with patch(
            "qna_generation_agent.app.lifespan._run_health_checks",
            new_callable=AsyncMock,
        ):
            async with lifespan(container) as state:
                assert get_runtime_state() is state

    async def test_lifespan_resets_subscriber_running_on_shutdown(self) -> None:
        """subscriber_running should be set to False during shutdown."""
        subscriber = MagicMock()
        container = self._mock_container(subscriber=subscriber)

        with patch(
            "qna_generation_agent.app.lifespan._run_health_checks",
            new_callable=AsyncMock,
        ):
            async with lifespan(container) as state:
                assert state.subscriber_running is True

        assert get_runtime_state().subscriber_running is False

    async def test_lifespan_logs_startup_and_shutdown(self) -> None:
        """Verify lifespan logs startup and shutdown events."""
        container = self._mock_container()

        with patch(
            "qna_generation_agent.app.lifespan._run_health_checks",
            new_callable=AsyncMock,
        ):
            with patch("qna_generation_agent.app.lifespan.logger") as mock_logger:
                async with lifespan(container):
                    pass  # pragma: no cover

        # Should log startup begin, startup complete, shutdown begin, shutdown complete
        mock_logger.info.assert_any_call(
            "lifespan_startup_begin",
            subscriber=False,
            telemetry=False,
        )
        mock_logger.info.assert_any_call("lifespan_startup_complete")
        mock_logger.info.assert_any_call("lifespan_shutdown_begin")
        mock_logger.info.assert_any_call("lifespan_shutdown_complete")

    async def test_lifespan_logs_startup_failure(self) -> None:
        """Verify lifespan logs startup failure."""
        container = self._mock_container()
        container.startup.side_effect = ValueError("Boom")

        with pytest.raises(ValueError, match="Boom"):
            async with lifespan(container):
                pass  # pragma: no cover

    async def test_lifespan_logs_shutdown_error(self) -> None:
        """Verify lifespan logs shutdown errors."""
        container = self._mock_container()
        container.shutdown.side_effect = RuntimeError("Shutdown error")

        with patch(
            "qna_generation_agent.app.lifespan._run_health_checks",
            new_callable=AsyncMock,
        ):
            with pytest.raises(RuntimeError, match="Shutdown error"):
                async with lifespan(container):
                    pass  # pragma: no cover


@pytest.mark.unit
class TestRunHealthChecks:
    """Tests for _run_health_checks."""

    async def test_run_health_checks_llm_healthy(self) -> None:
        """LLM health check should update llm_healthy when provider returns True."""
        container = MagicMock()
        container.llm_provider = MagicMock()
        container.llm_provider.health_check = AsyncMock(return_value=True)
        container.knowledge_client = None
        container.submission_client = None
        container.telemetry = None
        container.prompt_provider = None

        # Prime the singleton so _run_health_checks can access it
        from qna_generation_agent.app import lifespan as lifespan_module

        lifespan_module._runtime_state = RuntimeState()

        from qna_generation_agent.app.lifespan import _run_health_checks

        state = await _run_health_checks(container)
        assert state.llm_healthy is True

    async def test_run_health_checks_llm_unhealthy(self) -> None:
        """LLM health check failure should set llm_healthy to False."""
        container = MagicMock()
        container.llm_provider = MagicMock()
        container.llm_provider.health_check = AsyncMock(return_value=False)
        container.knowledge_client = None
        container.submission_client = None
        container.telemetry = None
        container.prompt_provider = None

        from qna_generation_agent.app import lifespan as lifespan_module

        lifespan_module._runtime_state = RuntimeState()

        from qna_generation_agent.app.lifespan import _run_health_checks

        state = await _run_health_checks(container)
        assert state.llm_healthy is False

    async def test_run_health_checks_llm_exception(self) -> None:
        """LLM health check exception should set llm_healthy to False."""
        container = MagicMock()
        container.llm_provider = MagicMock()
        container.llm_provider.health_check = AsyncMock(
            side_effect=ConnectionError("LLM down")
        )
        container.knowledge_client = None
        container.submission_client = None
        container.telemetry = None
        container.prompt_provider = None

        from qna_generation_agent.app import lifespan as lifespan_module

        lifespan_module._runtime_state = RuntimeState()

        from qna_generation_agent.app.lifespan import _run_health_checks

        state = await _run_health_checks(container)
        assert state.llm_healthy is False

    async def test_run_health_checks_llm_no_health_check(self) -> None:
        """LLM provider without health_check method should set llm_healthy to False."""
        container = MagicMock()
        container.llm_provider = MagicMock(spec=[])
        container.knowledge_client = None
        container.submission_client = None
        container.telemetry = None
        container.prompt_provider = None

        from qna_generation_agent.app import lifespan as lifespan_module

        lifespan_module._runtime_state = RuntimeState()

        from qna_generation_agent.app.lifespan import _run_health_checks

        state = await _run_health_checks(container)
        assert state.llm_healthy is False

    async def test_run_health_checks_knowledge_service(self) -> None:
        """Knowledge client health check should update knowledge_service_healthy."""
        container = MagicMock()
        container.llm_provider = None
        container.knowledge_client = MagicMock()
        container.knowledge_client.health_check = AsyncMock(return_value=True)
        container.submission_client = None
        container.telemetry = None
        container.prompt_provider = None

        from qna_generation_agent.app import lifespan as lifespan_module

        lifespan_module._runtime_state = RuntimeState()

        from qna_generation_agent.app.lifespan import _run_health_checks

        state = await _run_health_checks(container)
        assert state.knowledge_service_healthy is True

    async def test_run_health_checks_knowledge_no_method(self) -> None:
        """Knowledge client without health_check should set knowledge_service_healthy to False."""
        container = MagicMock()
        container.llm_provider = None
        container.knowledge_client = MagicMock(spec=[])
        container.submission_client = None
        container.telemetry = None
        container.prompt_provider = None

        from qna_generation_agent.app import lifespan as lifespan_module

        lifespan_module._runtime_state = RuntimeState()

        from qna_generation_agent.app.lifespan import _run_health_checks

        state = await _run_health_checks(container)
        assert state.knowledge_service_healthy is False

    async def test_run_health_checks_submission_service(self) -> None:
        """Submission client health check should update submission_service_healthy."""
        container = MagicMock()
        container.llm_provider = None
        container.knowledge_client = None
        container.submission_client = MagicMock()
        container.submission_client.health_check = AsyncMock(return_value=True)
        container.telemetry = None
        container.prompt_provider = None

        from qna_generation_agent.app import lifespan as lifespan_module

        lifespan_module._runtime_state = RuntimeState()

        from qna_generation_agent.app.lifespan import _run_health_checks

        state = await _run_health_checks(container)
        assert state.submission_service_healthy is True

    async def test_run_health_checks_submission_exception(self) -> None:
        """Submission client health check exception should set submission_service_healthy to False."""
        container = MagicMock()
        container.llm_provider = None
        container.knowledge_client = None
        container.submission_client = MagicMock()
        container.submission_client.health_check = AsyncMock(
            side_effect=ConnectionError("Submission down")
        )
        container.telemetry = None
        container.prompt_provider = None

        from qna_generation_agent.app import lifespan as lifespan_module

        lifespan_module._runtime_state = RuntimeState()

        from qna_generation_agent.app.lifespan import _run_health_checks

        state = await _run_health_checks(container)
        assert state.submission_service_healthy is False

    async def test_run_health_checks_telemetry_healthy(self) -> None:
        """Telemetry with health_check returning True should set telemetry_healthy."""
        container = MagicMock()
        container.llm_provider = None
        container.knowledge_client = None
        container.submission_client = None
        container.telemetry = MagicMock()
        container.telemetry.health_check = AsyncMock(return_value=True)
        container.prompt_provider = None

        from qna_generation_agent.app import lifespan as lifespan_module

        lifespan_module._runtime_state = RuntimeState()

        from qna_generation_agent.app.lifespan import _run_health_checks

        state = await _run_health_checks(container)
        assert state.telemetry_healthy is True

    async def test_run_health_checks_telemetry_none(self) -> None:
        """No telemetry should set telemetry_healthy to True."""
        container = MagicMock()
        container.llm_provider = None
        container.knowledge_client = None
        container.submission_client = None
        container.telemetry = None
        container.prompt_provider = None

        from qna_generation_agent.app import lifespan as lifespan_module

        lifespan_module._runtime_state = RuntimeState()

        from qna_generation_agent.app.lifespan import _run_health_checks

        state = await _run_health_checks(container)
        assert state.telemetry_healthy is True

    async def test_run_health_checks_telemetry_no_method(self) -> None:
        """Telemetry without health_check should set telemetry_healthy to True."""
        container = MagicMock()
        container.llm_provider = None
        container.knowledge_client = None
        container.submission_client = None
        container.telemetry = MagicMock(spec=[])
        container.prompt_provider = None

        from qna_generation_agent.app import lifespan as lifespan_module

        lifespan_module._runtime_state = RuntimeState()

        from qna_generation_agent.app.lifespan import _run_health_checks

        state = await _run_health_checks(container)
        assert state.telemetry_healthy is True

    async def test_run_health_checks_telemetry_exception(self) -> None:
        """Telemetry health check exception should set telemetry_healthy to False."""
        container = MagicMock()
        container.llm_provider = None
        container.knowledge_client = None
        container.submission_client = None
        container.telemetry = MagicMock()
        container.telemetry.health_check = AsyncMock(
            side_effect=ConnectionError("Telemetry down")
        )
        container.prompt_provider = None

        from qna_generation_agent.app import lifespan as lifespan_module

        lifespan_module._runtime_state = RuntimeState()

        from qna_generation_agent.app.lifespan import _run_health_checks

        state = await _run_health_checks(container)
        assert state.telemetry_healthy is False

    async def test_run_health_checks_prompt_provider_healthy(self) -> None:
        """Prompt provider with health_check should update prompt_provider_healthy."""
        container = MagicMock()
        container.llm_provider = None
        container.knowledge_client = None
        container.submission_client = None
        container.telemetry = None
        container.prompt_provider = MagicMock()
        container.prompt_provider.health_check = AsyncMock(return_value=True)

        from qna_generation_agent.app import lifespan as lifespan_module

        lifespan_module._runtime_state = RuntimeState()

        from qna_generation_agent.app.lifespan import _run_health_checks

        state = await _run_health_checks(container)
        assert state.prompt_provider_healthy is True

    async def test_run_health_checks_prompt_provider_none(self) -> None:
        """No prompt provider should set prompt_provider_healthy to True."""
        container = MagicMock()
        container.llm_provider = None
        container.knowledge_client = None
        container.submission_client = None
        container.telemetry = None
        container.prompt_provider = None

        from qna_generation_agent.app import lifespan as lifespan_module

        lifespan_module._runtime_state = RuntimeState()

        from qna_generation_agent.app.lifespan import _run_health_checks

        state = await _run_health_checks(container)
        assert state.prompt_provider_healthy is True

    async def test_run_health_checks_prompt_provider_no_method(self) -> None:
        """Prompt provider without health_check should set prompt_provider_healthy to True."""
        container = MagicMock()
        container.llm_provider = None
        container.knowledge_client = None
        container.submission_client = None
        container.telemetry = None
        container.prompt_provider = MagicMock(spec=[])

        from qna_generation_agent.app import lifespan as lifespan_module

        lifespan_module._runtime_state = RuntimeState()

        from qna_generation_agent.app.lifespan import _run_health_checks

        state = await _run_health_checks(container)
        assert state.prompt_provider_healthy is True

    async def test_run_health_checks_prompt_provider_exception(self) -> None:
        """Prompt provider health check exception should set prompt_provider_healthy to False."""
        container = MagicMock()
        container.llm_provider = None
        container.knowledge_client = None
        container.submission_client = None
        container.telemetry = None
        container.prompt_provider = MagicMock()
        container.prompt_provider.health_check = AsyncMock(
            side_effect=ConnectionError("Prompt provider down")
        )

        from qna_generation_agent.app import lifespan as lifespan_module

        lifespan_module._runtime_state = RuntimeState()

        from qna_generation_agent.app.lifespan import _run_health_checks

        state = await _run_health_checks(container)
        assert state.prompt_provider_healthy is False
