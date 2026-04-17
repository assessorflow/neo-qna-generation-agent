"""Pub/Sub publisher adapter using async patterns with PublisherClient.

This module uses google-cloud-pubsub>=2.37's PublisherClient with
asyncio-compatible futures for non-blocking message publishing.
"""

from __future__ import annotations

import asyncio

from google.api_core.exceptions import PermissionDenied
from google.cloud.pubsub_v1 import PublisherClient

from qna_generation_agent.app.json import dumps
from qna_generation_agent.app.logging import get_logger
from qna_generation_agent.application.errors import (
    StoragePermanentError,
    StorageTransientError,
)
from qna_generation_agent.application.ports.publisher import (
    DecisionAuditEvent,
    EventPublisher,
    TokenUsageEvent,
)
from qna_generation_agent.domain.events import QnAGenerationCompleted
from qna_generation_agent.infrastructure.messaging.envelope_models import (
    CompletionEnvelope,
    DecisionAuditEnvelope,
    DecisionAuditInputSummary,
    DecisionAuditOutputSummary,
    DecisionAuditPayload,
    TokenUsageEnvelope,
    TokenUsagePayload,
)

logger = get_logger(__name__)


class NullEventPublisher(EventPublisher):
    """Null-object publisher used when outbound Pub/Sub is disabled."""

    async def publish_completion(self, event: QnAGenerationCompleted) -> None:
        return None

    async def publish_decision_audit(self, event: DecisionAuditEvent) -> None:
        return None

    async def publish_token_usage(self, event: TokenUsageEvent) -> None:
        return None

    async def close(self) -> None:
        return None


class PubSubCompletionPublisher(EventPublisher):
    """Publisher adapter using PublisherClient with asyncio futures.

    Uses google-cloud-pubsub>=2.37's PublisherClient with async
    patterns via asyncio.to_thread() for non-blocking operations.
    """

    def __init__(
        self,
        *,
        project_id: str,
        topic_id: str,
        decision_audit_topic_id: str | None = None,
        token_usage_topic_id: str | None = None,
    ) -> None:
        self._project_id = project_id
        self._topic_id = topic_id
        self._decision_audit_topic_id = decision_audit_topic_id
        self._token_usage_topic_id = token_usage_topic_id
        self._client: PublisherClient | None = None

    async def _get_client(self) -> PublisherClient:
        """Lazy initialization of the publisher client."""
        if self._client is None:
            self._client = await asyncio.to_thread(PublisherClient)
        return self._client

    def _get_topic_path(self, client: PublisherClient, topic_id: str) -> str:
        """Get the full topic path."""
        return str(client.topic_path(self._project_id, topic_id))

    async def publish_completion(self, event: QnAGenerationCompleted) -> None:
        """Publish a completion event to Pub/Sub."""
        envelope = CompletionEnvelope.from_domain_event(event)
        attributes = {
            "event_type": envelope.event_type,
            "workflow_id": envelope.workflow_id,
            "correlation_id": envelope.correlation_id,
        }
        try:
            client = await self._get_client()
            topic_path = self._get_topic_path(client, self._topic_id)

            # Publish in thread and await the future
            future = await asyncio.to_thread(
                client.publish,
                topic_path,
                dumps(envelope.model_dump(mode="json")),
                **attributes,
            )
            await asyncio.to_thread(future.result, 30.0)
        except (PermissionDenied, PermissionError) as error:
            raise StoragePermanentError(
                "Pub/Sub publish was rejected due to permissions",
                topic=self._topic_id,
            ) from error
        except StoragePermanentError:
            raise
        except Exception as error:
            raise StorageTransientError(
                "Pub/Sub publish failed",
                topic=self._topic_id,
                error=str(error),
            ) from error

    async def publish_decision_audit(self, event: DecisionAuditEvent) -> None:
        """Publish decision audit event per spec."""
        if not self._decision_audit_topic_id:
            return

        payload = DecisionAuditPayload(
            workflow_id=event.workflow_id,
            input_summary=DecisionAuditInputSummary(
                question_set_id=event.input_summary.get("question_set_id", ""),
                iteration=event.input_summary.get("iteration", 1),
            ),
            output_summary=DecisionAuditOutputSummary(
                result=event.output_summary.get("result", "pass"),
                issues_found=event.output_summary.get("issues_found", 0),
            ),
            reasoning_steps=event.reasoning_steps,
            confidence_score=event.confidence_score,
            prompt_version=event.prompt_version,
            model_id=event.model_id,
            grounding_sources=event.grounding_sources,
        )
        envelope = DecisionAuditEnvelope(payload=payload)
        attributes = {
            "event_type": envelope.event_type,
            "workflow_id": event.workflow_id,
        }
        try:
            client = await self._get_client()
            topic_path = self._get_topic_path(client, self._decision_audit_topic_id)
            future = await asyncio.to_thread(
                client.publish,
                topic_path,
                dumps(envelope.model_dump(mode="json")),
                **attributes,
            )
            await asyncio.to_thread(future.result, 30.0)
        except Exception:
            logger.exception("decision_audit_publish_failed")

    async def publish_token_usage(self, event: TokenUsageEvent) -> None:
        """Publish token usage audit event per spec."""
        if not self._token_usage_topic_id:
            return

        payload = TokenUsagePayload(
            workflow_id=event.workflow_id,
            model_id=event.model_id,
            prompt_tokens=event.prompt_tokens,
            completion_tokens=event.completion_tokens,
            total_tokens=event.total_tokens,
            estimated_cost_usd=event.estimated_cost_usd,
            prompt_version=event.prompt_version,
        )
        envelope = TokenUsageEnvelope(payload=payload)
        attributes = {
            "event_type": envelope.event_type,
            "workflow_id": event.workflow_id,
        }
        try:
            client = await self._get_client()
            topic_path = self._get_topic_path(client, self._token_usage_topic_id)
            future = await asyncio.to_thread(
                client.publish,
                topic_path,
                dumps(envelope.model_dump(mode="json")),
                **attributes,
            )
            await asyncio.to_thread(future.result, 30.0)
        except Exception:
            logger.exception("token_usage_publish_failed")

    async def close(self) -> None:
        """Close the publisher client."""
        if self._client is not None:
            await asyncio.to_thread(self._client.transport.close)
            self._client = None
