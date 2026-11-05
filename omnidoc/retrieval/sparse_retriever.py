"""
SparseRetriever — keyword search via the BM25 inverted index.
"""

from __future__ import annotations

from typing import Any, List

from omnidoc.storage import BM25Index
from omnidoc.utils import get_logger

logger = get_logger(__name__)


class SparseRetriever:
    """
    Runs BM25 keyword lookup against the in-memory inverted index.

    Parameters
    ----------
    bm25_index:
        Shared :class:`BM25Index` instance.
    top_k:
        Number of sparse candidates to return before fusion.
    """

    def __init__(self, bm25_index: BM25Index, top_k: int = 100) -> None:
        self.bm25 = bm25_index
        self.top_k = top_k

    def retrieve(self, query: str) -> List[dict[str, Any]]:
        """
        Perform BM25 keyword search and return ranked candidates.

        Each result dict contains: ``id`` and ``score``.
        Note: text/metadata must be fetched from the vector store
        payload during the fusion step.
        """
        logger.debug("Sparse retrieval | query='%.60s…'  top_k=%d", query, self.top_k)
        results = self.bm25.search(query=query, top_k=self.top_k)
        logger.debug("Sparse retrieval returned %d results", len(results))
        return results
