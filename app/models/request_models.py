"""
models/request_models.py
------------------------
Pydantic models that describe the shape of incoming API requests.

Using Pydantic gives us automatic validation, type coercion,
and auto-generated Swagger/OpenAPI documentation for free.
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Body payload for the /ai/chat endpoint."""

    provider: str = Field(
        default="groq",
        description="The LLM provider to use (groq, openai, anthropic).",
        examples=["groq", "openai", "anthropic"],
    )
    prompt: str = Field(
        ...,
        min_length=1,
        description="The user prompt to send to the LLM.",
        examples=["Explain what an API is in simple terms."],
    )
