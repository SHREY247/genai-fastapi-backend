"""
core/config.py
--------------
Centralised configuration — loads environment variables once
and exposes them as simple module-level constants.

Uses python-dotenv so students can store secrets in a .env file
without hard-coding them into source code.
"""

import os
from dotenv import load_dotenv

# Load variables from the .env file into the environment
load_dotenv()

# ---- Groq / OpenAI-compatible settings ----

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_BASE_URL: str = os.getenv(
    "GROQ_BASE_URL",
    "https://api.groq.com/openai/v1/chat/completions",
)
