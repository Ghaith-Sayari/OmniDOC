"""
EmbeddingEngine — wraps open-source (BAAI/bge-large-en-v1.5) and
cloud (OpenAI text-embedding-3-small) embedding backends.
"""

from __future__ import annotations

from enum import Enum
from typing import List

from omnidoc.utils import get_logger

logger = get_logger(__name__)


class EmbeddingBackend(str, Enum):
    BGE = "bge"
    OPENAI = "openai"


class EmbeddingEngine:
    """
    Thin wrapper that selects and initialises the correct embedding
    backend based on configuration.

    Parameters
    ----------
    backend:
        ``"bge"`` — BAAI/bge-large-en-v1.5 via HuggingFace (local).
        ``"openai"`` — text-embedding-3-small via OpenAI API.
    model_name:
        Override the default model name for the selected backend.
    batch_size:
        Number of texts embedded per API/inference call.
    """

    _DEFAULTS = {
        EmbeddingBackend.BGE: "BAAI/bge-large-en-v1.5",
        EmbeddingBackend.OPENAI: "text-embedding-3-small",
    }

    def __init__(
        self,
        backend: EmbeddingBackend | str = EmbeddingBackend.BGE,
        model_name: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.backend = EmbeddingBackend(backend)
        self.model_name = model_name or self._DEFAULTS[self.backend]
        self.batch_size = batch_size
        self._model = self._load_model()
        logger.info("EmbeddingEngine ready | backend=%s  model=%s", self.backend, self.model_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Return a list of embedding vectors for *texts*."""
        if self.backend == EmbeddingBackend.BGE:
            return self._embed_bge(texts)
        return self._embed_openai(texts)

    def embed_query(self, query: str) -> List[float]:
        """Return a single embedding vector for a user *query*."""
        return self.embed_texts([query])[0]

    def get_llama_embed_model(self):
        """
        Return a LlamaIndex-compatible embed model object for use in
        :class:`llama_index.core.Settings`.
        """
        if self.backend == EmbeddingBackend.BGE:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore

            return HuggingFaceEmbedding(model_name=self.model_name)
        else:
            from llama_index.embeddings.openai import OpenAIEmbedding  # type: ignore

            return OpenAIEmbedding(model=self.model_name)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        """Eagerly validate that the required library is importable."""
        if self.backend == EmbeddingBackend.BGE:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                return SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError("Install sentence-transformers: pip install sentence-transformers")
        else:
            try:
                import openai  # type: ignore  # noqa: F401

                return None  # OpenAI calls are stateless
            except ImportError:
                raise ImportError("Install openai: pip install openai")

    def _embed_bge(self, texts: List[str]) -> List[List[float]]:
        vectors = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            embeddings = self._model.encode(batch, normalize_embeddings=True)
            vectors.extend(embeddings.tolist())
        return vectors

    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        import openai  # type: ignore

        client = openai.OpenAI()
        vectors = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = client.embeddings.create(input=batch, model=self.model_name)
            vectors.extend([item.embedding for item in response.data])
        return vectors
