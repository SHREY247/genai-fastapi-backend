"""
providers/groq_provider.py
--------------------------
Implementation of the Groq LLM provider.
"""

import requests
from fastapi import HTTPException
from app.providers.base import BaseLLMProvider
from app.core.config import GROQ_API_KEY, GROQ_MODEL
from app.core.logging import logger

class GroqProvider(BaseLLMProvider):
    def __init__(self):
        self.api_key = GROQ_API_KEY
        self.model = GROQ_MODEL
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"

    def generate(self, prompt: str) -> str:
        if not self.api_key:
            logger.error("GROQ_API_KEY is not configured")
            raise HTTPException(
                status_code=400,
                detail="GROQ_API_KEY is not configured. Please add it to your .env file."
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
        }

        logger.info(f"Calling Groq API with model: {self.model}")
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
        except requests.Timeout:
            logger.error("Groq API request timed out")
            raise HTTPException(status_code=504, detail="Groq API request timed out.")
        except requests.RequestException as e:
            logger.error(f"Groq API error: {str(e)}")
            raise HTTPException(status_code=502, detail=f"Groq API error: {str(e)}")

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            logger.error("Unexpected response structure from Groq API")
            raise HTTPException(status_code=502, detail="Unexpected response structure from Groq API.")
