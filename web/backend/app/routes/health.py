"""Health check endpoint — always returns 200, no model required."""

from fastapi import APIRouter

from app.model_service import ModelService

router = APIRouter(tags=["health"])
_service = ModelService()


@router.get("/health")
async def health() -> dict:
    """Liveness check. Returns model_loaded=true once load() has been called."""
    return {"status": "ok", "model_loaded": _service.is_loaded}
