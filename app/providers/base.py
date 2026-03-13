"""
providers/base.py
-----------------
The abstract base class for all LLM providers.
Ensures every provider implements a common generate method.
"""

from abc import ABC, abstractmethod

class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Takes a prompt and returns the string response from the LLM.
        """
        pass
