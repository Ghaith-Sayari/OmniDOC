"""Unit tests for HierarchicalChunker."""

import pytest
from llama_index.core import Document
from omnidoc.ingestion.chunker import HierarchicalChunker, ChunkResult


class TestHierarchicalChunker:
    def test_default_chunk_sizes(self):
        chunker = HierarchicalChunker()
        assert chunker.chunk_sizes == [1024, 512, 128]

    def test_custom_chunk_sizes(self):
        chunker = HierarchicalChunker(chunk_sizes=[512, 256, 64])
        assert chunker.chunk_sizes == [512, 256, 64]

    def test_chunk_returns_chunk_result(self):
        chunker = HierarchicalChunker()
        docs = [Document(text="This is a test document. " * 100)]
        result = chunker.chunk(docs)
        assert isinstance(result, ChunkResult)
        assert len(result.all_nodes) > 0
        assert len(result.leaf_nodes) > 0
        assert len(result.leaf_nodes) <= len(result.all_nodes)

    def test_leaf_nodes_are_subset(self):
        chunker = HierarchicalChunker()
        docs = [Document(text="Sample text. " * 200)]
        result = chunker.chunk(docs)
        all_ids = {n.node_id for n in result.all_nodes}
        leaf_ids = {n.node_id for n in result.leaf_nodes}
        assert leaf_ids.issubset(all_ids)

    def test_metadata_contains_counts(self):
        chunker = HierarchicalChunker()
        docs = [Document(text="Text. " * 50)]
        result = chunker.chunk(docs)
        assert "total_nodes" in result.metadata
        assert "leaf_count" in result.metadata
        assert result.metadata["leaf_count"] == len(result.leaf_nodes)
