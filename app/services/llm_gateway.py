"""
services/llm_gateway.py
-----------------------
The LLM Gateway acts as a central dispatcher.
It selects the appropriate provider based on the request.
"""

from fastapi import HTTPException
from app.providers.groq_provider import GroqProvider
from app.providers.openai_provider import OpenAIProvider
from app.providers.anthropic_provider import AnthropicProvider
from app.core.logging import logger

class LLMGateway:
    def __init__(self):
        # Initialise providers
        self.providers = {
            "groq": GroqProvider(),
            "openai": OpenAIProvider(),
            "anthropic": AnthropicProvider(),
        }

    def generate(self, provider_name: str, prompt: str) -> str:
        """
        Routes the generation request to the selected provider.
        """
        provider_name = provider_name.lower()
        
        if provider_name not in self.providers:
            logger.error(f"Unsupported provider requested: {provider_name}")
            raise HTTPException(
                status_code=400,
                detail=f"Provider '{provider_name}' is not supported. Choose from: {list(self.providers.keys())}"
            )

        logger.info(f"Gateway selecting provider: {provider_name}")
        provider = self.providers[provider_name]
        return provider.generate(prompt)

# Single instance to be used across the app
gateway = LLMGateway()
