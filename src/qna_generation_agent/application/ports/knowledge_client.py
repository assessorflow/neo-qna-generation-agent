"""Knowledge Service client port."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Chunk:
    """Knowledge chunk returned by Knowledge Service."""

    chunk_id: str
    workflow_id: str
    content: str
    source_type: str
    metadata: dict[str, str]
    score: float


@dataclass(frozen=True, slots=True)
class SimilaritySearchCommand:
    """Command to search for similar chunks."""

    query: str
    workflow_id: str
    kb_type: str  # "document", "enriched", "policy"
    top_k: int = 5
    assessment_id: str = ""  # Optional per proto3


@dataclass(frozen=True, slots=True)
class Topic:
    """Topic returned by Knowledge Service."""

    topic_id: str
    name: str
    subtopics: list[Topic]


@dataclass(frozen=True, slots=True)
class GetTopicsCommand:
    """Command to get topics for a workflow."""

    workflow_id: str


@dataclass(frozen=True, slots=True)
class GetChunksByIdsCommand:
    """Command to get chunks by their IDs."""

    chunk_ids: list[str]


class KnowledgeClient(ABC):
    """Port for Knowledge Service gRPC client."""

    @abstractmethod
    async def similarity_search(
        self,
        command: SimilaritySearchCommand,
    ) -> list[Chunk]:
        """Search for similar chunks in the Knowledge Service.

        Args:
            command: The search command with query, workflow_id, kb_type, top_k.

        Returns:
            List of matching chunks ordered by relevance.

        Raises:
            StorageTransientError: If the request fails temporarily (retryable).
            StoragePermanentError: If the request fails permanently.
        """

    @abstractmethod
    async def get_chunks_by_ids(
        self,
        command: GetChunksByIdsCommand,
    ) -> list[Chunk]:
        """Get chunks by their IDs from the Knowledge Service.

        Args:
            command: The command with chunk_ids to retrieve.

        Returns:
            List of chunks matching the provided IDs.

        Raises:
            StorageTransientError: If the request fails temporarily (retryable).
            StoragePermanentError: If the request fails permanently.
        """

    @abstractmethod
    async def get_topics(
        self,
        command: GetTopicsCommand,
    ) -> list[Topic]:
        """Get topics for a workflow from the Knowledge Service.

        Args:
            command: The command with workflow_id.

        Returns:
            List of topics with nested subtopics.

        Raises:
            StorageTransientError: If the request fails temporarily (retryable).
            StoragePermanentError: If the request fails permanently.
        """

    @abstractmethod
    async def close(self) -> None:
        """Close the client connection."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Check connectivity to the Knowledge Service.

        Returns:
            True if the service is healthy, False otherwise.
        """
