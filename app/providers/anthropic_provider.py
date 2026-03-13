"""
providers/anthropic_provider.py
-------------------------------
Implementation of the Anthropic LLM provider.
"""

import requests
from fastapi import HTTPException
from app.providers.base import BaseLLMProvider
from app.core.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL
from app.core.logging import logger

class AnthropicProvider(BaseLLMProvider):
    def __init__(self):
        self.api_key = ANTHROPIC_API_KEY
        self.model = ANTHROPIC_MODEL
        self.base_url = "https://api.anthropic.com/v1/messages"

    def generate(self, prompt: str) -> str:
        if not self.api_key:
            logger.error("ANTHROPIC_API_KEY is not configured")
            raise HTTPException(
                status_code=400,
                detail="ANTHROPIC_API_KEY is not configured. Please add it to your .env file."
            )

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": prompt},
            ],
        }

        logger.info(f"Calling Anthropic API with model: {self.model}")
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
        except requests.Timeout:
            logger.error("Anthropic API request timed out")
            raise HTTPException(status_code=504, detail="Anthropic API request timed out.")
        except requests.RequestException as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise HTTPException(status_code=502, detail=f"Anthropic API error: {str(e)}")

        data = response.json()
        try:
            return data["content"][0]["text"]
        except (KeyError, IndexError):
            logger.error("Unexpected response structure from Anthropic API")
            raise HTTPException(status_code=502, detail="Unexpected response structure from Anthropic API.")
