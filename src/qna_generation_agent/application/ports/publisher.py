"""Outbound event publisher port."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from qna_generation_agent.domain.events import QnAGenerationCompleted


@dataclass(frozen=True, slots=True)
class DecisionAuditEvent:
    """Decision audit event per spec."""

    workflow_id: str
    input_summary: dict[str, Any]
    output_summary: dict[str, Any]
    reasoning_steps: list[str]
    confidence_score: float
    prompt_version: str
    model_id: str
    grounding_sources: list[str]


@dataclass(frozen=True, slots=True)
class TokenUsageEvent:
    """Token usage audit event per spec."""

    workflow_id: str
    model_id: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    prompt_version: str


class EventPublisher(ABC):
    """Publisher port for outbound domain events."""

    @abstractmethod
    async def publish_completion(self, event: QnAGenerationCompleted) -> None:
        """Publish the completion event."""

    @abstractmethod
    async def publish_decision_audit(self, event: DecisionAuditEvent) -> None:
        """Publish decision audit event per spec."""

    @abstractmethod
    async def publish_token_usage(self, event: TokenUsageEvent) -> None:
        """Publish token usage audit event per spec."""

    @abstractmethod
    async def close(self) -> None:
        """Dispose any network clients held by the publisher."""
