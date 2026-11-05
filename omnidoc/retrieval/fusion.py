"""
RRFFusion — Reciprocal Rank Fusion to merge dense and sparse result lists.

Formula:  RRF(d) = Σ  1 / (k + rank_i(d))
where k=60 is a smoothing constant recommended by Cormack et al. (2009).
"""

from __future__ import annotations

from typing import Any, Dict, List


class RRFFusion:
    """
    Combines dense (Qdrant) and sparse (BM25) result lists using RRF.

    Parameters
    ----------
    k:
        Smoothing constant (default 60).
    dense_weight / sparse_weight:
        Multiplicative weights applied before fusion (default 1.0 each).
        Increase ``dense_weight`` for semantic-heavy corpora; increase
        ``sparse_weight`` for keyword-rich technical documentation.
    """

    def __init__(
        self,
        k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> None:
        self.k = k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def fuse(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Merge two ranked lists and return a unified ranked list.

        Parameters
        ----------
        dense_results:
            Output of :meth:`DenseRetriever.retrieve` — each item must
            have an ``"id"`` key.
        sparse_results:
            Output of :meth:`SparseRetriever.retrieve` — each item must
            have an ``"id"`` key.

        Returns
        -------
        list[dict]
            Sorted by descending RRF score.  Items originating from
            dense results carry ``text`` and ``metadata`` fields.
        """
        scores: Dict[str, float] = {}
        id_to_data: Dict[str, Dict[str, Any]] = {}

        for rank, item in enumerate(dense_results, start=1):
            id_ = item["id"]
            scores[id_] = scores.get(id_, 0.0) + self.dense_weight / (self.k + rank)
            id_to_data[id_] = item  # carry payload from dense result

        for rank, item in enumerate(sparse_results, start=1):
            id_ = item["id"]
            scores[id_] = scores.get(id_, 0.0) + self.sparse_weight / (self.k + rank)
            if id_ not in id_to_data:
                id_to_data[id_] = item  # sparse-only hit (no dense payload yet)

        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [{"rrf_score": score, **id_to_data[id_]} for id_, score in fused]
