"""
Query API route.

POST /query  — full RAG pipeline: retrieve → fuse → rerank → generate.
"""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from omnidoc.config import Settings
from omnidoc.generation import AnswerGenerator, CitationVerifier
from omnidoc.ingestion.embedder import EmbeddingEngine
from omnidoc.retrieval import DenseRetriever, SparseRetriever, RRFFusion, CrossEncoderReranker
from omnidoc.storage import QdrantStore, BM25Index
from omnidoc.utils import get_logger

router = APIRouter()
logger = get_logger(__name__)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="User question.")
    filters: Dict[str, Any] | None = Field(default=None, description="Optional Qdrant payload filters.")
    top_k: int = Field(default=100, description="Dense + sparse candidates before fusion.")
    top_n: int = Field(default=5, description="Final candidates after re-ranking.")


class QueryResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
    citation_verification: Dict[str, Any]
    model: str
    usage: Dict[str, int]


def get_settings() -> Settings:
    return Settings()


@router.post("", response_model=QueryResponse)
async def query(request: QueryRequest, settings: Settings = Depends(get_settings)) -> QueryResponse:
    """
    Execute the full RAG pipeline and return a grounded answer.

    Pipeline:
      1. Dense retrieval (Qdrant ANN)
      2. Sparse retrieval (BM25)
      3. RRF fusion
      4. Cross-encoder re-ranking → top-N
      5. LLM generation with citation guardrails
      6. Citation verification
    """
    embedder = EmbeddingEngine(backend=settings.embedding_backend, model_name=settings.embedding_model)
    qdrant = QdrantStore(host=settings.qdrant_host, port=settings.qdrant_port, collection_name=settings.qdrant_collection)
    bm25 = BM25Index(index_path=settings.bm25_index_path, corpus_path=settings.bm25_corpus_path)

    dense = DenseRetriever(embedder=embedder, qdrant_store=qdrant, top_k=request.top_k)
    sparse = SparseRetriever(bm25_index=bm25, top_k=request.top_k)
    fusion = RRFFusion(k=settings.rrf_k, dense_weight=settings.rrf_dense_weight, sparse_weight=settings.rrf_sparse_weight)
    reranker = CrossEncoderReranker(backend=settings.reranker_backend, model_name=settings.reranker_model, top_n=request.top_n)
    generator = AnswerGenerator(backend=settings.llm_backend, model_name=settings.llm_model, temperature=settings.llm_temperature)
    verifier = CitationVerifier(strict=False)

    dense_results = dense.retrieve(request.question, filters=request.filters)
    sparse_results = sparse.retrieve(request.question)
    fused = fusion.fuse(dense_results, sparse_results)
    reranked = reranker.rerank(request.question, fused)

    result = generator.generate(query=request.question, context_chunks=reranked)
    verification = verifier.verify(result.answer, reranked)

    return QueryResponse(
        answer=result.answer,
        citations=reranked,
        citation_verification=verification,
        model=result.model,
        usage=result.usage,
    )
