"""
VectorIndexer — commits hierarchical nodes to Qdrant (dense) and
BM25 (sparse) stores, maintaining parent-child relationships.
"""

from __future__ import annotations

from typing import Any

from llama_index.core.schema import BaseNode

from omnidoc.ingestion.chunker import ChunkResult
from omnidoc.ingestion.embedder import EmbeddingEngine
from omnidoc.storage import QdrantStore, BM25Index
from omnidoc.utils import get_logger

logger = get_logger(__name__)


class VectorIndexer:
    """
    Orchestrates the two-phase indexing step:

    1. **Dense indexing** — embeds leaf nodes and upserts vectors +
       full payload (text, metadata, parent-node ID) into Qdrant.
    2. **Sparse indexing** — adds leaf node text to BM25 corpus and
       persists the inverted index to disk.

    Parameters
    ----------
    embedder:
        Configured :class:`EmbeddingEngine` instance.
    qdrant_store:
        Configured :class:`QdrantStore` instance.
    bm25_index:
        Configured :class:`BM25Index` instance.
    """

    def __init__(
        self,
        embedder: EmbeddingEngine,
        qdrant_store: QdrantStore,
        bm25_index: BM25Index,
    ) -> None:
        self.embedder = embedder
        self.qdrant = qdrant_store
        self.bm25 = bm25_index

    def index(self, chunk_result: ChunkResult) -> dict[str, Any]:
        """
        Embed and upsert all leaf nodes from *chunk_result*.

        Returns
        -------
        dict
            Summary with ``indexed_count`` and ``collection_name``.
        """
        leaves = chunk_result.leaf_nodes
        logger.info("Indexing %d leaf nodes…", len(leaves))

        texts = [node.get_content() for node in leaves]
        vectors = self.embedder.embed_texts(texts)

        payloads = self._build_payloads(leaves)
        node_ids = [node.node_id for node in leaves]

        self.qdrant.upsert(ids=node_ids, vectors=vectors, payloads=payloads)
        self.bm25.add_corpus(texts, ids=node_ids)
        self.bm25.persist()

        logger.info("Indexing complete | count=%d", len(leaves))
        return {"indexed_count": len(leaves), "collection": self.qdrant.collection_name}

    @staticmethod
    def _build_payloads(nodes: list[BaseNode]) -> list[dict[str, Any]]:
        payloads = []
        for node in nodes:
            parent_id = node.parent_node.node_id if node.parent_node else None
            payloads.append(
                {
                    "text": node.get_content(),
                    "metadata": node.metadata,
                    "parent_node_id": parent_id,
                    "node_id": node.node_id,
                }
            )
        return payloads
