"""
BM25Index — sparse keyword index backed by rank_bm25.

Provides fast keyword search complementing Qdrant's dense vectors.
The index is serialised to disk so it can be reloaded across restarts.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List

from omnidoc.utils import get_logger

logger = get_logger(__name__)

DEFAULT_INDEX_PATH = Path("data/processed/bm25_index.pkl")
DEFAULT_CORPUS_PATH = Path("data/processed/bm25_corpus.json")


class BM25Index:
    """
    Maintains a BM25 corpus and exposes keyword search returning
    ranked ``(id, score)`` tuples.

    Parameters
    ----------
    index_path:
        Path where the serialised BM25 object is stored.
    corpus_path:
        Path where the raw tokenised corpus is stored (for reloading).
    """

    def __init__(
        self,
        index_path: Path | str = DEFAULT_INDEX_PATH,
        corpus_path: Path | str = DEFAULT_CORPUS_PATH,
    ) -> None:
        self.index_path = Path(index_path)
        self.corpus_path = Path(corpus_path)
        self._index = None
        self._ids: List[str] = []
        self._corpus: List[List[str]] = []

        if self.index_path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_corpus(self, texts: List[str], ids: List[str]) -> None:
        """Tokenise *texts* and append to the internal corpus."""
        tokenised = [self._tokenise(t) for t in texts]
        self._corpus.extend(tokenised)
        self._ids.extend(ids)
        self._rebuild()

    def search(self, query: str, top_k: int = 20) -> List[dict]:
        """
        Return up to *top_k* results ranked by BM25 score.

        Each result is a dict with keys ``id`` and ``score``.
        """
        if self._index is None:
            logger.warning("BM25 index is empty — returning no results.")
            return []

        tokens = self._tokenise(query)
        scores = self._index.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"id": self._ids[i], "score": float(s)} for i, s in ranked if s > 0]

    def persist(self) -> None:
        """Serialise the BM25 object and corpus to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.corpus_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.index_path, "wb") as f:
            pickle.dump(self._index, f)

        with open(self.corpus_path, "w", encoding="utf-8") as f:
            json.dump({"ids": self._ids, "corpus": self._corpus}, f)

        logger.info("BM25 index persisted to %s", self.index_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _rebuild(self) -> None:
        try:
            from rank_bm25 import BM25Okapi  # type: ignore

            self._index = BM25Okapi(self._corpus)
        except ImportError:
            raise ImportError("Install rank-bm25: pip install rank-bm25")

    def _load(self) -> None:
        try:
            with open(self.index_path, "rb") as f:
                self._index = pickle.load(f)
            with open(self.corpus_path, encoding="utf-8") as f:
                data = json.load(f)
                self._ids = data["ids"]
                self._corpus = data["corpus"]
            logger.info("BM25 index loaded from %s (%d docs)", self.index_path, len(self._ids))
        except Exception as exc:
            logger.warning("Could not load BM25 index: %s", exc)

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        import re

        return re.findall(r"\b\w+\b", text.lower())
