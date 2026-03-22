"""
rag/embedding.py
----------------
Dense embedding generation using Hugging Face sentence-transformers.

Wraps the model in a clean class so students can see:
- model loading
- document embedding
- query embedding
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List


class HuggingFaceEmbedder:
    """
    Generates dense vector embeddings using a sentence-transformers model.

    Default model: all-MiniLM-L6-v2  (384-dimensional, fast, good quality)
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        print(f"[Embedding] Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"[Embedding] Model loaded  (dimension={self.dimension})")

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Embeds a list of document/chunk texts.

        Args:
            texts: list of strings to embed.

        Returns:
            numpy array of shape (len(texts), dimension).
        """
        print(f"[Embedding] Embedding {len(texts)} text(s)...")
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embeds a single query string.

        Returns:
            numpy array of shape (1, dimension).
        """
        embedding = self.model.encode([query], show_progress_bar=False)
        return np.array(embedding, dtype=np.float32)
