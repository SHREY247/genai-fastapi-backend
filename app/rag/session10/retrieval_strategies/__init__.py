"""
retrieval_strategies/
---------------------
Pluggable retrieval backends for Session 10.

Each retriever exposes the same interface:
    .retrieve(query: str, top_k: int) -> List[Dict]

This makes them interchangeable in the pipeline.
"""

from app.rag.session10.retrieval_strategies.vector import VectorRetriever
from app.rag.session10.retrieval_strategies.bm25 import BM25Retriever
from app.rag.session10.retrieval_strategies.hybrid import HybridRetriever

__all__ = ["VectorRetriever", "BM25Retriever", "HybridRetriever"]
