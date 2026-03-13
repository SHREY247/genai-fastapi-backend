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

# Project Settings
PROJECT_NAME: str = "Applied GenAI Teaching Backend"
API_V1_STR: str = "/api/v1"

# ---- Groq Settings ----
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ---- OpenAI Settings ----
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ---- Anthropic Settings ----
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")
