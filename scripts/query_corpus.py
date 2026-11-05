"""
One-shot query script — useful for quick corpus sanity checks.

Usage:
    python scripts/query_corpus.py "What is the refund policy?"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from omnidoc.config import Settings
from omnidoc.generation import AnswerGenerator, CitationVerifier
from omnidoc.ingestion.embedder import EmbeddingEngine
from omnidoc.retrieval import DenseRetriever, SparseRetriever, RRFFusion, CrossEncoderReranker
from omnidoc.storage import QdrantStore, BM25Index
from omnidoc.utils import get_logger

logger = get_logger("query_corpus")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/query_corpus.py <question>")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    settings = Settings()

    embedder = EmbeddingEngine(backend=settings.embedding_backend)
    qdrant = QdrantStore(host=settings.qdrant_host, port=settings.qdrant_port, collection_name=settings.qdrant_collection)
    bm25 = BM25Index(index_path=settings.bm25_index_path, corpus_path=settings.bm25_corpus_path)

    dense = DenseRetriever(embedder=embedder, qdrant_store=qdrant, top_k=settings.retrieval_top_k)
    sparse = SparseRetriever(bm25_index=bm25, top_k=settings.retrieval_top_k)
    fusion = RRFFusion(k=settings.rrf_k)
    reranker = CrossEncoderReranker(backend=settings.reranker_backend, top_n=settings.reranker_top_n)
    generator = AnswerGenerator(backend=settings.llm_backend, temperature=settings.llm_temperature)
    verifier = CitationVerifier()

    logger.info("Query: %s", question)
    dense_res = dense.retrieve(question)
    sparse_res = sparse.retrieve(question)
    fused = fusion.fuse(dense_res, sparse_res)
    reranked = reranker.rerank(question, fused)
    result = generator.generate(query=question, context_chunks=reranked)
    report = verifier.verify(result.answer, reranked)

    print("\n" + "=" * 70)
    print(result.answer)
    print("=" * 70)
    print(f"\n✓ Citations valid: {report['valid']}")
    if report["fabricated_ids"]:
        print(f"⚠ Fabricated: {report['fabricated_ids']}")


if __name__ == "__main__":
    main()
