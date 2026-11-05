"""
DenseRetriever — queries Qdrant with an embedded query vector.
"""

from __future__ import annotations

from typing import Any, List

from omnidoc.ingestion.embedder import EmbeddingEngine
from omnidoc.storage import QdrantStore
from omnidoc.utils import get_logger

logger = get_logger(__name__)


class DenseRetriever:
    """
    Embeds the user query and runs ANN search against Qdrant.

    Parameters
    ----------
    embedder:
        Shared :class:`EmbeddingEngine` instance.
    qdrant_store:
        Shared :class:`QdrantStore` instance.
    top_k:
        Number of candidates to return before fusion / re-ranking.
    """

    def __init__(
        self,
        embedder: EmbeddingEngine,
        qdrant_store: QdrantStore,
        top_k: int = 100,
    ) -> None:
        self.embedder = embedder
        self.qdrant = qdrant_store
        self.top_k = top_k

    def retrieve(self, query: str, filters: dict[str, Any] | None = None) -> List[dict[str, Any]]:
        """
        Embed *query* and return the top-k dense candidates from Qdrant.

        Each result dict contains: ``id``, ``score``, ``text``,
        ``metadata``, ``parent_node_id``.
        """
        logger.debug("Dense retrieval | query='%.60s…'  top_k=%d", query, self.top_k)
        vector = self.embedder.embed_query(query)
        results = self.qdrant.search(query_vector=vector, top_k=self.top_k, filters=filters)
        logger.debug("Dense retrieval returned %d results", len(results))
        return results
