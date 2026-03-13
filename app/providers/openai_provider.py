"""
providers/openai_provider.py
----------------------------
Implementation of the OpenAI LLM provider.
"""

import requests
from fastapi import HTTPException
from app.providers.base import BaseLLMProvider
from app.core.config import OPENAI_API_KEY, OPENAI_MODEL
from app.core.logging import logger

class OpenAIProvider(BaseLLMProvider):
    def __init__(self):
        self.api_key = OPENAI_API_KEY
        self.model = OPENAI_MODEL
        self.base_url = "https://api.openai.com/v1/chat/completions"

    def generate(self, prompt: str) -> str:
        if not self.api_key:
            logger.error("OPENAI_API_KEY is not configured")
            raise HTTPException(
                status_code=400,
                detail="OPENAI_API_KEY is not configured. Please add it to your .env file."
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

        logger.info(f"Calling OpenAI API with model: {self.model}")
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
        except requests.Timeout:
            logger.error("OpenAI API request timed out")
            raise HTTPException(status_code=504, detail="OpenAI API request timed out.")
        except requests.RequestException as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise HTTPException(status_code=502, detail=f"OpenAI API error: {str(e)}")

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            logger.error("Unexpected response structure from OpenAI API")
            raise HTTPException(status_code=502, detail="Unexpected response structure from OpenAI API.")
