"""
api/routes/chat.py
------------------
Chat endpoint that accepts a user prompt, forwards it to the
LLM service, and returns the model's response.
"""

from fastapi import APIRouter, HTTPException
from app.models.request_models import ChatRequest
from app.models.response_models import ChatResponse
from app.services.llm_service import ask_llm

router = APIRouter()

@router.post(
    "/ai/chat",
    response_model=ChatResponse,
    summary="Chat with the LLM Gateway",
    description="Send a prompt and a provider selection to receive an AI-generated response.",
)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Thin route handler:
      1. Receive the validated ChatRequest from the client.
      2. Delegate to the llm_service with provider and prompt.
      3. Wrap the result in a ChatResponse.
    """
    try:
        reply = ask_llm(provider=request.provider, prompt=request.prompt)
        return ChatResponse(response=reply)
    except HTTPException:
        # Re-raise known HTTP errors from the service layer as-is
        raise
    except Exception as exc:
        # Catch any unexpected errors and return a clean 500
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(exc)}",
        )
