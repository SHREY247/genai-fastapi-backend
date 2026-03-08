"""
services/llm_service.py
-----------------------
Service layer that handles communication with the LLM API.

Why a separate service layer?
  • Keeps route handlers thin — they only deal with HTTP concerns.
  • Business logic (calling the LLM) is isolated and easy to test.
  • When we later add RAG, embeddings, or caching, this is the
    only file that needs to change.
"""

import requests
from fastapi import HTTPException

from app.core.config import GROQ_API_KEY, GROQ_MODEL, GROQ_BASE_URL


def ask_llm(prompt: str) -> str:
    """
    Send a user prompt to the Groq / OpenAI-compatible chat completion
    endpoint and return the assistant's reply.

    Parameters
    ----------
    prompt : str
        The user's message to send to the model.

    Returns
    -------
    str
        The text content of the assistant's response.

    Raises
    ------
    HTTPException
        400 — if the API key is not configured.
        502 — if the upstream LLM API returns an error.
    """

    # --- Guard: ensure the API key is available ---
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=400,
            detail=(
                "GROQ_API_KEY is not set. "
                "Please add it to your .env file and restart the server."
            ),
        )

    # --- Build the request ---
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "user", "content": prompt},
        ],
    }

    # --- Call the LLM API ---
    try:
        response = requests.post(
            GROQ_BASE_URL,
            headers=headers,
            json=payload,
            timeout=30,  # seconds — prevents hanging if the API is slow
        )
    except requests.Timeout:
        # Separate handler so students see a clear timeout message
        raise HTTPException(
            status_code=504,
            detail="LLM API request timed out. Try again or increase the timeout.",
        )
    except requests.ConnectionError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not connect to the LLM API. Check your network or GROQ_BASE_URL: {exc}",
        )
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"LLM API request failed: {exc}",
        )

    # --- Handle non-2xx responses ---
    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=(
                f"LLM API returned status {response.status_code}: "
                f"{response.text}"
            ),
        )

    # --- Extract the assistant message ---
    data = response.json()

    try:
        assistant_message: str = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Unexpected response structure from LLM API: {exc}",
        )

    return assistant_message
