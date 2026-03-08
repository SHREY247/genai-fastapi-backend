"""
api/routes/health.py
--------------------
A lightweight health-check endpoint.

Every production API should expose at least one health route so
that monitoring tools, load balancers, and developers can verify
the server is alive.
"""

from fastapi import APIRouter

from app.models.response_models import HealthResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns the current server status.",
)
def health_check() -> HealthResponse:
    """Return a simple status object confirming the server is up."""
    return HealthResponse(
        status="ok",
        message="GenAI FastAPI backend is running.",
    )
