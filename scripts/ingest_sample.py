"""
Ingestion pipeline smoke test script.

Usage:
    python scripts/ingest_sample.py --path data/raw/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from omnidoc.config import Settings
from omnidoc.ingestion import DocumentParser, HierarchicalChunker, EmbeddingEngine, VectorIndexer
from omnidoc.storage import QdrantStore, BM25Index, MetadataStore
from omnidoc.utils import list_documents, get_logger

logger = get_logger("ingest_sample")


def main() -> None:
    parser = argparse.ArgumentParser(description="OmniDoc sample ingestion script.")
    parser.add_argument("--path", default="data/raw", help="Path to ingest (file or directory).")
    parser.add_argument("--recursive", action="store_true", default=True)
    args = parser.parse_args()

    settings = Settings()
    target = Path(args.path)

    files = [target] if target.is_file() else list(list_documents(target, recursive=args.recursive))
    if not files:
        logger.warning("No supported documents found under %s", target)
        sys.exit(1)

    logger.info("Found %d document(s) to ingest", len(files))

    doc_parser = DocumentParser()
    chunker = HierarchicalChunker(chunk_sizes=settings.chunk_sizes)
    embedder = EmbeddingEngine(backend=settings.embedding_backend, batch_size=settings.embedding_batch_size)
    qdrant = QdrantStore(
        host=settings.qdrant_host, port=settings.qdrant_port,
        collection_name=settings.qdrant_collection, vector_size=settings.qdrant_vector_size,
    )
    bm25 = BM25Index(index_path=settings.bm25_index_path, corpus_path=settings.bm25_corpus_path)
    indexer = VectorIndexer(embedder=embedder, qdrant_store=qdrant, bm25_index=bm25)
    meta_store = MetadataStore(settings.metadata_db_path)

    for i, file in enumerate(files, 1):
        logger.info("[%d/%d] Ingesting %s", i, len(files), file.name)
        try:
            parsed = doc_parser.parse(file)
            doc = doc_parser.to_llama_document(parsed)
            chunk_result = chunker.chunk([doc])
            result = indexer.index(chunk_result)
            meta_store.upsert(str(file), parsed.metadata | {"source": str(file), "format": parsed.format})
            logger.info("  → Indexed %d chunks", result["indexed_count"])
        except Exception as exc:
            logger.exception("  ✗ Failed: %s", exc)

    meta_store.close()
    logger.info("All done.")


if __name__ == "__main__":
    main()
