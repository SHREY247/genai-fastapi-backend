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
    title="GenAI LLM Gateway Backend",
    description="A production-style FastAPI backend with a provider-agnostic LLM gateway supporting Groq, OpenAI, and Anthropic.",
    version="1.1.0"
)

# ---- Root route ----
@app.get("/", tags=["Root"])
def root():
    """Quick check that the server is alive."""
    return {
        "message": "GenAI LLM Gateway Backend is running",
        "status": "online",
        "session": 4
    }

# ---- Register routers ----
app.include_router(health.router, tags=["Health"])
app.include_router(chat.router, tags=["Chat"])
