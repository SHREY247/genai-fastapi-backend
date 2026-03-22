"""
rag/retriever.py
----------------
Thin wrapper that ties together:
- query embedding
- FAISS search
- result formatting

Keeps the pipeline.py code clean and readable.
"""

from typing import List, Dict
from app.rag.embedding import HuggingFaceEmbedder
from app.rag.vector_store import FAISSVectorStore


class Retriever:
    """
    Combines the embedder and vector store for single-call retrieval.
    """

    def __init__(self, embedder: HuggingFaceEmbedder, store: FAISSVectorStore):
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, top_k: int = 3, k: int = None) -> List[Dict]:
        """
        Embeds the query and returns the top-k matching records.

        Args:
            query: the user's question.
            top_k: number of results to return.
            k:     alias for top_k (for convenience).

        Returns:
            List of dicts with: rank, source, chunk_id, text, score
        """
        num_results = k if k is not None else top_k
        query_embedding = self.embedder.embed_query(query)
        results = self.store.search(query_embedding, top_k=num_results)
        return results
