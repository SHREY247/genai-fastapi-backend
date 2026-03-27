"""
main.py
-------
Application entry-point.

Creates the FastAPI application instance and wires up all routers.

Run with:
    uvicorn app.main:app --reload
"""

from fastapi import FastAPI
from app.api.routes import health, chat
from app.core.logging import setup_logging

# Setup application-wide logging
setup_logging()

# ---- Create the FastAPI app ----
app = FastAPI(
    title="GenAI FastAPI Backend",
    description=(
        "Applied GenAI Engineering course backend. "
        "Session 4: Provider-agnostic LLM gateway. "
        "Session 7: RAG foundations. "
        "Session 8: Multi-source interview prep RAG."
    ),
    version="1.2.0"
)

# ---- Root route ----
@app.get("/", tags=["Root"])
def root():
    """Quick check that the server is alive."""
    return {
        "message": "GenAI FastAPI Backend is running",
        "status": "online",
        "latest_session": 8
    }

# ---- Register routers ----
app.include_router(health.router, tags=["Health"])
app.include_router(chat.router, tags=["Chat"])
