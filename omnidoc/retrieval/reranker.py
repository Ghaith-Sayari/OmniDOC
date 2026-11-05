"""
CrossEncoderReranker — filters fused candidates with a cross-encoder model.

Supported backends:
  * Cohere Rerank API  (cloud, highest accuracy)
  * BGE Reranker       (local, BAAI/bge-reranker-large)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List

from omnidoc.utils import get_logger

logger = get_logger(__name__)


class RerankerBackend(str, Enum):
    COHERE = "cohere"
    BGE = "bge"


class CrossEncoderReranker:
    """
    Re-scores a list of retrieved candidates against the original query
    using a cross-encoder model, then returns the top-N results.

    Parameters
    ----------
    backend:
        ``"cohere"`` or ``"bge"``.
    model_name:
        Override for the default model of the chosen backend.
    top_n:
        Final number of results returned to the generation layer
        (default 5 — matching the architecture spec).
    """

    _DEFAULTS = {
        RerankerBackend.COHERE: "rerank-english-v3.0",
        RerankerBackend.BGE: "BAAI/bge-reranker-large",
    }

    def __init__(
        self,
        backend: RerankerBackend | str = RerankerBackend.COHERE,
        model_name: str | None = None,
        top_n: int = 5,
    ) -> None:
        self.backend = RerankerBackend(backend)
        self.model_name = model_name or self._DEFAULTS[self.backend]
        self.top_n = top_n
        logger.info(
            "CrossEncoderReranker ready | backend=%s  model=%s  top_n=%d",
            self.backend,
            self.model_name,
            self.top_n,
        )

    def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Re-rank *candidates* against *query* and return top-N results.

        Parameters
        ----------
        query:
            Original user query string.
        candidates:
            Fused result list from :class:`RRFFusion`.  Each item must
            contain a ``"text"`` key.

        Returns
        -------
        list[dict]
            Up to ``top_n`` dicts with an added ``"rerank_score"`` key.
        """
        if not candidates:
            return []

        if self.backend == RerankerBackend.COHERE:
            return self._rerank_cohere(query, candidates)
        return self._rerank_bge(query, candidates)

    # ------------------------------------------------------------------

    def _rerank_cohere(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            import cohere  # type: ignore

            co = cohere.Client()
            docs = [c.get("text", "") for c in candidates]
            response = co.rerank(
                model=self.model_name,
                query=query,
                documents=docs,
                top_n=self.top_n,
                return_documents=False,
            )
            reranked = []
            for result in response.results:
                item = dict(candidates[result.index])
                item["rerank_score"] = result.relevance_score
                reranked.append(item)
            return reranked
        except ImportError:
            raise ImportError("Install cohere: pip install cohere")

    def _rerank_bge(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            model = CrossEncoder(self.model_name)
            pairs = [(query, c.get("text", "")) for c in candidates]
            scores = model.predict(pairs)
            ranked = sorted(
                zip(scores, candidates),
                key=lambda x: x[0],
                reverse=True,
            )[: self.top_n]
            return [{**item, "rerank_score": float(score)} for score, item in ranked]
        except ImportError:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers")
