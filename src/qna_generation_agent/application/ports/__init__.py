"""Ports (interfaces) for the application layer.

Ports define the contracts that infrastructure adapters must implement.
"""

from __future__ import annotations

from qna_generation_agent.application.ports.idempotency import IdempotencyStore
from qna_generation_agent.application.ports.llm import LLMProvider
from qna_generation_agent.application.ports.prompt_provider import (
    Prompt,
    PromptProvider,
)
from qna_generation_agent.application.ports.publisher import EventPublisher
from qna_generation_agent.application.ports.repository import (
    AssessmentContextRepository,
    QuestionSetRepository,
)
from qna_generation_agent.application.ports.telemetry import TelemetryPort

__all__ = [
    "AssessmentContextRepository",
    "EventPublisher",
    "IdempotencyStore",
    "LLMProvider",
    "Prompt",
    "PromptProvider",
    "QuestionSetRepository",
    "TelemetryPort",
]
