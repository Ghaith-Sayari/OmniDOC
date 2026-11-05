"""Unit tests for RRFFusion."""

import pytest
from omnidoc.retrieval.fusion import RRFFusion


class TestRRFFusion:
    def setup_method(self):
        self.fusion = RRFFusion(k=60)

    def test_empty_inputs(self):
        result = self.fusion.fuse([], [])
        assert result == []

    def test_dense_only(self):
        dense = [{"id": "a", "score": 0.9, "text": "doc a"}, {"id": "b", "score": 0.7, "text": "doc b"}]
        result = self.fusion.fuse(dense, [])
        assert len(result) == 2
        assert result[0]["id"] == "a"  # higher rank = higher RRF score

    def test_sparse_only(self):
        sparse = [{"id": "x", "score": 5.0}, {"id": "y", "score": 3.0}]
        result = self.fusion.fuse([], sparse)
        assert len(result) == 2

    def test_overlap_boosts_score(self):
        dense = [{"id": "shared", "score": 0.8, "text": "shared doc"}]
        sparse = [{"id": "shared", "score": 4.0}]
        result_overlap = self.fusion.fuse(dense, sparse)
        result_dense_only = self.fusion.fuse(dense, [])
        # Shared ID should have higher RRF score when it appears in both lists
        assert result_overlap[0]["rrf_score"] > result_dense_only[0]["rrf_score"]

    def test_all_results_have_rrf_score(self):
        dense = [{"id": f"d{i}", "score": float(i), "text": f"doc {i}"} for i in range(5)]
        sparse = [{"id": f"s{i}", "score": float(i)} for i in range(5)]
        result = self.fusion.fuse(dense, sparse)
        assert all("rrf_score" in r for r in result)

    def test_weights_affect_ranking(self):
        fusion_dense_heavy = RRFFusion(k=60, dense_weight=2.0, sparse_weight=0.5)
        dense = [{"id": "d1", "score": 0.9, "text": "dense doc"}]
        sparse = [{"id": "s1", "score": 9.0}]
        result = fusion_dense_heavy.fuse(dense, sparse)
        # d1 gets 2x weight; ensure it outranks s1 (which gets 0.5x)
        ids = [r["id"] for r in result]
        assert ids[0] == "d1"
