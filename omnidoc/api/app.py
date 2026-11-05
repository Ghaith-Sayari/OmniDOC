"""
FastAPI application factory and router registration.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from omnidoc.api.routes import ingest, query, evaluate, health


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: initialise shared resources
        yield
        # Shutdown: flush connections, persist indices

    app = FastAPI(
        title="OmniDoc RAG API",
        description="Enterprise Multi-Format Retrieval-Augmented Generation",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(health.router, tags=["Health"])
    app.include_router(ingest.router, prefix="/ingest", tags=["Ingestion"])
    app.include_router(query.router, prefix="/query", tags=["Query"])
    app.include_router(evaluate.router, prefix="/evaluate", tags=["Evaluation"])

    return app
