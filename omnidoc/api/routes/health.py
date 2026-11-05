"""Health check endpoint."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check() -> dict:
    """Return service liveness status."""
    return {"status": "ok", "service": "omnidoc-rag"}
