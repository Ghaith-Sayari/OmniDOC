"""
Ingestion API routes.

POST /ingest/file   — upload a single document file for processing.
POST /ingest/batch  — enqueue a directory path for batch ingestion.
GET  /ingest/status — list all indexed documents from MetadataStore.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel

from omnidoc.config import Settings
from omnidoc.ingestion import DocumentParser, HierarchicalChunker, EmbeddingEngine, VectorIndexer
from omnidoc.storage import QdrantStore, BM25Index, MetadataStore
from omnidoc.utils import get_logger

router = APIRouter()
logger = get_logger(__name__)


class BatchIngestRequest(BaseModel):
    directory: str
    recursive: bool = True


def get_settings() -> Settings:
    return Settings()


@router.post("/file")
async def ingest_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
) -> dict:
    """
    Accept a document upload, save it to ``data/raw/``, and queue
    the ingestion pipeline as a background task.
    """
    suffix = Path(file.filename or "upload").suffix.lower()
    doc_id = str(uuid.uuid4())
    dest = Path("data/raw") / f"{doc_id}{suffix}"
    dest.parent.mkdir(parents=True, exist_ok=True)

    contents = await file.read()
    dest.write_bytes(contents)
    logger.info("Uploaded %s → %s", file.filename, dest)

    background_tasks.add_task(_run_ingestion, str(dest), doc_id, settings)
    return {"doc_id": doc_id, "status": "queued", "file": file.filename}


@router.post("/batch")
async def ingest_batch(
    request: BatchIngestRequest,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
) -> dict:
    """Enqueue all supported documents found under *request.directory*."""
    from omnidoc.utils import list_documents

    directory = Path(request.directory)
    if not directory.exists():
        raise HTTPException(status_code=404, detail=f"Directory not found: {directory}")

    files = list(list_documents(directory, recursive=request.recursive))
    if not files:
        raise HTTPException(status_code=422, detail="No supported documents found.")

    queued = []
    for path in files:
        doc_id = str(uuid.uuid4())
        background_tasks.add_task(_run_ingestion, str(path), doc_id, settings)
        queued.append({"doc_id": doc_id, "file": path.name})

    return {"queued": len(queued), "documents": queued}


@router.get("/status")
async def ingest_status(settings: Settings = Depends(get_settings)) -> dict:
    """Return the list of all indexed documents from the metadata store."""
    store = MetadataStore(settings.metadata_db_path)
    records = store.list_all()
    store.close()
    return {"total": len(records), "documents": records}


# ------------------------------------------------------------------
# Background task
# ------------------------------------------------------------------

def _run_ingestion(file_path: str, doc_id: str, settings: Settings) -> None:
    """Full ingestion pipeline executed in a background thread."""
    try:
        parser = DocumentParser()
        parsed = parser.parse(file_path)
        document = parser.to_llama_document(parsed)

        chunker = HierarchicalChunker(chunk_sizes=settings.chunk_sizes)
        chunk_result = chunker.chunk([document])

        embedder = EmbeddingEngine(
            backend=settings.embedding_backend,
            model_name=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
        )
        qdrant = QdrantStore(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=settings.qdrant_collection,
            vector_size=settings.qdrant_vector_size,
        )
        bm25 = BM25Index(
            index_path=settings.bm25_index_path,
            corpus_path=settings.bm25_corpus_path,
        )
        indexer = VectorIndexer(embedder=embedder, qdrant_store=qdrant, bm25_index=bm25)
        indexer.index(chunk_result)

        meta_store = MetadataStore(settings.metadata_db_path)
        meta_store.upsert(doc_id, {**parsed.metadata, "source": file_path, "format": parsed.format, "page_count": parsed.page_count})
        meta_store.close()

        logger.info("Ingestion complete | doc_id=%s", doc_id)
    except Exception as exc:
        logger.exception("Ingestion failed for %s: %s", file_path, exc)
