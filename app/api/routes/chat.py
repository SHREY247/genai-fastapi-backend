"""
api/routes/chat.py
------------------
Chat endpoint that accepts a user prompt, forwards it to the
LLM service, and returns the model's response.

This is the main business endpoint of the API.
Route handlers should stay thin — validate input, call the
service layer, and return a typed response.
"""

from fastapi import APIRouter, HTTPException

from app.models.request_models import ChatRequest
from app.models.response_models import ChatResponse
from app.services.llm_service import ask_llm

router = APIRouter()


@router.post(
    "/ai/chat",
    response_model=ChatResponse,
    summary="Chat with the LLM",
    description="Send a prompt and receive an AI-generated response.",
)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Thin route handler:
      1. Receive the validated ChatRequest from the client.
      2. Delegate to the llm_service.
      3. Wrap the result in a ChatResponse.
    """
    try:
        reply = ask_llm(prompt=request.prompt)
        return ChatResponse(response=reply)
    except HTTPException:
        # Re-raise known HTTP errors from the service layer as-is
        raise
    except Exception as exc:
        # Catch any unexpected errors and return a clean 500
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {exc}",
        )
