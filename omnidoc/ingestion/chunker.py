"""
HierarchicalChunker — layout-aware hierarchical semantic chunking.

Implements a three-level hierarchy:
  Level 0 (parent)  : 1024-token blocks  — broad context windows
  Level 1 (section) :  512-token blocks  — paragraph-level context
  Level 2 (child)   :  128-token blocks  — sentence-level embeddings

Child nodes are embedded; retrieval returns parent nodes to the LLM,
preserving surrounding context (small-to-big retrieval pattern).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llama_index.core import Document
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.schema import BaseNode

from omnidoc.utils import get_logger

logger = get_logger(__name__)

DEFAULT_CHUNK_SIZES = [1024, 512, 128]


@dataclass
class ChunkResult:
    """Container returned by :class:`HierarchicalChunker`."""

    all_nodes: list[BaseNode] = field(default_factory=list)
    leaf_nodes: list[BaseNode] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class HierarchicalChunker:
    """
    Wraps LlamaIndex's :class:`HierarchicalNodeParser` to produce
    a three-level node tree and extract leaf nodes for embedding.

    Parameters
    ----------
    chunk_sizes:
        Ordered list of token sizes [parent, section, child].
        Defaults to ``[1024, 512, 128]``.
    """

    def __init__(self, chunk_sizes: list[int] | None = None) -> None:
        self.chunk_sizes = chunk_sizes or DEFAULT_CHUNK_SIZES
        self._parser = HierarchicalNodeParser.from_defaults(chunk_sizes=self.chunk_sizes)
        logger.info("HierarchicalChunker initialised | chunk_sizes=%s", self.chunk_sizes)

    def chunk(self, documents: list[Document]) -> ChunkResult:
        """
        Parse *documents* into a hierarchical node tree.

        Parameters
        ----------
        documents:
            List of LlamaIndex :class:`Document` objects (output of
            :meth:`DocumentParser.to_llama_document`).

        Returns
        -------
        ChunkResult
            Contains ``all_nodes`` (full tree) and ``leaf_nodes``
            (128-token children ready for embedding).
        """
        logger.info("Chunking %d document(s)…", len(documents))
        all_nodes = self._parser.get_nodes_from_documents(documents)
        leaves = get_leaf_nodes(all_nodes)

        logger.info(
            "Chunking complete | total_nodes=%d  leaf_nodes=%d",
            len(all_nodes),
            len(leaves),
        )
        return ChunkResult(
            all_nodes=all_nodes,
            leaf_nodes=leaves,
            metadata={
                "chunk_sizes": self.chunk_sizes,
                "total_nodes": len(all_nodes),
                "leaf_count": len(leaves),
            },
        )
