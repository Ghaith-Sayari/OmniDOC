"""Integration tests for the FastAPI endpoints."""

import pytest
from httpx import AsyncClient, ASGITransport
from omnidoc.api.app import create_app


@pytest.fixture
def app():
    return create_app()


@pytest.mark.asyncio
async def test_health_endpoint(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "omnidoc-rag"


@pytest.mark.asyncio
async def test_ingest_status_empty(app):
    """GET /ingest/status should return an empty list on fresh start."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/ingest/status")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "documents" in data


@pytest.mark.asyncio
async def test_ingest_batch_nonexistent_dir(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/ingest/batch",
            json={"directory": "/nonexistent/directory", "recursive": True},
        )
    assert response.status_code == 404
