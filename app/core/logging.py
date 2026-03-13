"""
core/logging.py
---------------
Simple logging setup for the application.
"""

import logging
import sys

# Configure the root logger
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

setup_logging()
logger = logging.getLogger("genai-gateway")
