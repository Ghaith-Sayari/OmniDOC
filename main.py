"""
OmniDoc RAG — CLI entrypoint.

Commands
--------
  serve     Start the FastAPI server.
  ingest    Ingest a file or directory from the command line.
  query     Run a one-shot query against the indexed corpus.
  evaluate  Run Ragas evaluation on a JSON sample file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
import uvicorn

app = typer.Typer(name="omnidoc", help="OmniDoc RAG command-line interface.")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host."),
    port: int = typer.Option(8000, help="Bind port."),
    workers: int = typer.Option(1, help="Uvicorn worker count."),
    reload: bool = typer.Option(False, help="Enable hot-reload (dev only)."),
) -> None:
    """Start the OmniDoc FastAPI server."""
    typer.echo(f"Starting OmniDoc RAG API on {host}:{port} …")
    uvicorn.run(
        "omnidoc.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level="info",
    )


@app.command()
def ingest(
    path: str = typer.Argument(..., help="File or directory path to ingest."),
    recursive: bool = typer.Option(True, help="Scan directory recursively."),
) -> None:
    """Ingest a document or directory into the OmniDoc corpus."""
    from omnidoc.config import Settings
    from omnidoc.ingestion import DocumentParser, HierarchicalChunker, EmbeddingEngine, VectorIndexer
    from omnidoc.storage import QdrantStore, BM25Index, MetadataStore
    from omnidoc.utils import list_documents

    settings = Settings()
    target = Path(path)

    files = [target] if target.is_file() else list(list_documents(target, recursive=recursive))
    if not files:
        typer.echo("No supported documents found.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Found {len(files)} document(s). Starting ingestion…")

    parser = DocumentParser()
    chunker = HierarchicalChunker(chunk_sizes=settings.chunk_sizes)
    embedder = EmbeddingEngine(backend=settings.embedding_backend, batch_size=settings.embedding_batch_size)
    qdrant = QdrantStore(host=settings.qdrant_host, port=settings.qdrant_port, collection_name=settings.qdrant_collection)
    bm25 = BM25Index(index_path=settings.bm25_index_path, corpus_path=settings.bm25_corpus_path)
    indexer = VectorIndexer(embedder=embedder, qdrant_store=qdrant, bm25_index=bm25)
    meta_store = MetadataStore(settings.metadata_db_path)

    for i, file in enumerate(files, 1):
        typer.echo(f"[{i}/{len(files)}] Ingesting {file.name} …")
        try:
            parsed = parser.parse(file)
            doc = parser.to_llama_document(parsed)
            chunk_result = chunker.chunk([doc])
            result = indexer.index(chunk_result)
            meta_store.upsert(str(file), parsed.metadata | {"source": str(file), "format": parsed.format})
            typer.echo(f"  ✓ Indexed {result['indexed_count']} chunks")
        except Exception as exc:
            typer.echo(f"  ✗ Failed: {exc}", err=True)

    meta_store.close()
    typer.echo("Ingestion complete.")


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask the corpus."),
    top_k: int = typer.Option(100, help="Dense + sparse candidates before fusion."),
    top_n: int = typer.Option(5, help="Final candidates after re-ranking."),
) -> None:
    """Ask a question and get a grounded answer from the indexed corpus."""
    from omnidoc.config import Settings
    from omnidoc.generation import AnswerGenerator, CitationVerifier
    from omnidoc.ingestion.embedder import EmbeddingEngine
    from omnidoc.retrieval import DenseRetriever, SparseRetriever, RRFFusion, CrossEncoderReranker
    from omnidoc.storage import QdrantStore, BM25Index

    settings = Settings()
    embedder = EmbeddingEngine(backend=settings.embedding_backend)
    qdrant = QdrantStore(host=settings.qdrant_host, port=settings.qdrant_port, collection_name=settings.qdrant_collection)
    bm25 = BM25Index(index_path=settings.bm25_index_path, corpus_path=settings.bm25_corpus_path)

    dense = DenseRetriever(embedder=embedder, qdrant_store=qdrant, top_k=top_k)
    sparse = SparseRetriever(bm25_index=bm25, top_k=top_k)
    fusion = RRFFusion(k=settings.rrf_k)
    reranker = CrossEncoderReranker(backend=settings.reranker_backend, top_n=top_n)
    generator = AnswerGenerator(backend=settings.llm_backend, temperature=settings.llm_temperature)
    verifier = CitationVerifier()

    typer.echo("\nRetrieving…")
    dense_res = dense.retrieve(question)
    sparse_res = sparse.retrieve(question)
    fused = fusion.fuse(dense_res, sparse_res)
    reranked = reranker.rerank(question, fused)

    typer.echo("Generating answer…\n")
    result = generator.generate(query=question, context_chunks=reranked)
    verification = verifier.verify(result.answer, reranked)

    typer.echo("=" * 60)
    typer.echo(result.answer)
    typer.echo("=" * 60)
    if verification["fabricated_ids"]:
        typer.echo(f"⚠ Fabricated citations: {verification['fabricated_ids']}", err=True)
    else:
        typer.echo("✓ All citations verified.")


@app.command()
def evaluate(
    samples_file: str = typer.Argument(..., help="JSON file with Q&A samples."),
    export: bool = typer.Option(True, help="Export results to CSV/JSON."),
) -> None:
    """Run Ragas triad evaluation on a JSON sample file."""
    from omnidoc.config import Settings
    from omnidoc.evaluation import RagasEvaluator, EvaluationReporter

    settings = Settings()
    path = Path(samples_file)
    if not path.exists():
        typer.echo(f"File not found: {path}", err=True)
        raise typer.Exit(1)

    with open(path, encoding="utf-8") as f:
        samples = json.load(f)

    evaluator = RagasEvaluator(
        thresholds={
            "context_relevance": settings.eval_threshold_context_relevance,
            "groundedness": settings.eval_threshold_groundedness,
            "answer_relevance": settings.eval_threshold_answer_relevance,
        }
    )
    reporter = EvaluationReporter(output_dir=settings.eval_output_dir)
    records = evaluator.evaluate(samples)
    reporter.print_summary(records)

    if export:
        reporter.to_json(records)
        reporter.to_csv(records)


if __name__ == "__main__":
    app()
