"""
models/response_models.py
-------------------------
Pydantic models that describe the shape of API responses.

Defining explicit response schemas:
  • makes the Swagger docs self-documenting
  • lets frontend teams code against a contract
  • keeps our API responses consistent
"""

from pydantic import BaseModel, Field


class ChatResponse(BaseModel):
    """Response returned by the /ai/chat endpoint."""

    response: str = Field(
        ...,
        description="The assistant's reply from the LLM.",
    )


class HealthResponse(BaseModel):
    """Response returned by the /health endpoint."""

    status: str = Field(
        ...,
        description="Current server status.",
        examples=["ok"],
    )
    message: str = Field(
        ...,
        description="Human-readable status message.",
        examples=["GenAI FastAPI backend is running."],
    )
