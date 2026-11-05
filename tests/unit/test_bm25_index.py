"""Unit tests for BM25Index."""

import pytest
from pathlib import Path
from omnidoc.storage.bm25_index import BM25Index


class TestBM25Index:
    def test_add_and_search(self, tmp_path):
        index = BM25Index(
            index_path=tmp_path / "bm25.pkl",
            corpus_path=tmp_path / "corpus.json",
        )
        texts = ["The quick brown fox", "jumps over the lazy dog", "enterprise document retrieval"]
        ids = ["doc1", "doc2", "doc3"]
        index.add_corpus(texts, ids)

        results = index.search("quick fox", top_k=3)
        assert len(results) > 0
        assert results[0]["id"] == "doc1"

    def test_persist_and_reload(self, tmp_path):
        index_path = tmp_path / "bm25.pkl"
        corpus_path = tmp_path / "corpus.json"

        index = BM25Index(index_path=index_path, corpus_path=corpus_path)
        index.add_corpus(["Hello world", "RAG systems"], ["id1", "id2"])
        index.persist()

        # Reload
        reloaded = BM25Index(index_path=index_path, corpus_path=corpus_path)
        results = reloaded.search("hello", top_k=1)
        assert len(results) == 1
        assert results[0]["id"] == "id1"

    def test_empty_index_returns_empty(self, tmp_path):
        index = BM25Index(
            index_path=tmp_path / "bm25.pkl",
            corpus_path=tmp_path / "corpus.json",
        )
        results = index.search("anything")
        assert results == []

    def test_no_match_returns_empty(self, tmp_path):
        index = BM25Index(
            index_path=tmp_path / "bm25.pkl",
            corpus_path=tmp_path / "corpus.json",
        )
        index.add_corpus(["alpha beta gamma"], ["id1"])
        results = index.search("xyz123notexist")
        assert results == []
