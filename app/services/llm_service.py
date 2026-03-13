"""
services/llm_service.py
-----------------------
Service layer that handles communication with the LLM Gateway.

This version is now thin and provider-agnostic, delegating
the actual calling logic to the LLMGateway.
"""

from app.services.llm_gateway import gateway

def ask_llm(provider: str, prompt: str) -> str:
    """
    Passes the request to the LLMGateway.
    """
    return gateway.generate(provider, prompt)
