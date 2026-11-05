"""FastAPI application exposing ingestion and query endpoints."""

from .app import create_app

__all__ = ["create_app"]
