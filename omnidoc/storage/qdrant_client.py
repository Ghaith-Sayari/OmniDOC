"""
QdrantStore — thin wrapper around the official qdrant-client SDK.

Handles collection creation (with HNSW + cosine similarity), vector
upsert, dense similarity search, and payload-filtered queries.
"""

from __future__ import annotations

from typing import Any, List

from omnidoc.utils import get_logger

logger = get_logger(__name__)

DEFAULT_VECTOR_SIZE = 1024  # bge-large-en-v1.5 output dimension
DEFAULT_COLLECTION = "omnidoc_vectors"


class QdrantStore:
    """
    Encapsulates all Qdrant operations required by the OmniDoc pipelines.

    Parameters
    ----------
    host:
        Qdrant server hostname (default ``"localhost"``).
    port:
        Qdrant gRPC/REST port (default ``6333``).
    collection_name:
        Name of the target collection (created if absent).
    vector_size:
        Embedding dimension — must match the chosen embedding model.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = DEFAULT_COLLECTION,
        vector_size: int = DEFAULT_VECTOR_SIZE,
    ) -> None:
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._client = self._connect()
        self._ensure_collection()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert(
        self,
        ids: List[str],
        vectors: List[List[float]],
        payloads: List[dict[str, Any]],
    ) -> None:
        """Upsert a batch of vectors with their payloads into Qdrant."""
        from qdrant_client.models import PointStruct  # type: ignore

        points = [
            PointStruct(id=self._hash_id(id_), vector=vec, payload=payload)
            for id_, vec, payload in zip(ids, vectors, payloads)
        ]
        self._client.upsert(collection_name=self.collection_name, points=points)
        logger.debug("Upserted %d points to '%s'", len(points), self.collection_name)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
        score_threshold: float = 0.0,
    ) -> List[dict[str, Any]]:
        """
        Dense vector similarity search.

        Returns a list of dicts with keys:
        ``id``, ``score``, ``text``, ``metadata``, ``parent_node_id``.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue  # type: ignore

        qdrant_filter = None
        if filters:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ]
            qdrant_filter = Filter(must=conditions)

        results = self._client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return [
            {
                "id": str(hit.id),
                "score": hit.score,
                "text": hit.payload.get("text", ""),
                "metadata": hit.payload.get("metadata", {}),
                "parent_node_id": hit.payload.get("parent_node_id"),
            }
            for hit in results
        ]

    def delete_collection(self) -> None:
        """Drop the entire collection (use with caution)."""
        self._client.delete_collection(self.collection_name)
        logger.warning("Deleted collection '%s'", self.collection_name)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _connect(self):
        try:
            from qdrant_client import QdrantClient  # type: ignore

            client = QdrantClient(host=self.host, port=self.port)
            logger.info("Connected to Qdrant @ %s:%d", self.host, self.port)
            return client
        except ImportError:
            raise ImportError("Install qdrant-client: pip install qdrant-client")

    def _ensure_collection(self) -> None:
        from qdrant_client.models import VectorParams, Distance  # type: ignore

        existing = [c.name for c in self._client.get_collections().collections]
        if self.collection_name not in existing:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection '%s'", self.collection_name)

    @staticmethod
    def _hash_id(node_id: str) -> int:
        """Convert a UUID string to a stable integer Qdrant point ID."""
        import hashlib

        return int(hashlib.md5(node_id.encode()).hexdigest(), 16) % (2**63)
