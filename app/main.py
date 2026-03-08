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

# ---- Create the FastAPI app ----
app = FastAPI(
    title="GenAI FastAPI Backend",
    description="A simple FastAPI service that wraps an LLM API (Groq) into backend endpoints for AI applications.",
    version="1.0.0"
)

# ---- Root route ----
# A quick sanity-check for students — just open http://localhost:8000/
@app.get("/", tags=["Root"])
def root():
    """Quick check that the server is alive."""
    return {"message": "GenAI FastAPI Backend is running"}


# ---- Register routers ----
app.include_router(health.router, tags=["Health"])
app.include_router(chat.router, tags=["Chat"])
