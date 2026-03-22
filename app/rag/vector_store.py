"""
rag/vector_store.py
-------------------
FAISS-based in-memory vector store.

Key responsibilities:
- Build an IndexFlatIP (inner-product) index from normalized embeddings
  to approximate cosine similarity.
- Store metadata (source, chunk_id, text) alongside vectors.
- Return top-k results with scores.
"""

import numpy as np
import faiss
from typing import List, Dict


class FAISSVectorStore:
    """
    A simple FAISS vector store for dense retrieval.

    Uses Inner Product search on L2-normalized vectors,
    which is equivalent to cosine similarity.
    """

    def __init__(self):
        self.index = None
        self.records: List[Dict[str, str]] = []

    def build_index(
        self,
        embeddings: np.ndarray,
        records: List[Dict[str, str]],
    ) -> None:
        """
        Builds the FAISS index from embeddings and stores the metadata records.

        Args:
            embeddings: numpy array of shape (n, dim).
            records:    list of dicts with at least 'source', 'chunk_id', 'text'.
        """
        if len(embeddings) != len(records):
            raise ValueError(
                f"Mismatch: {len(embeddings)} embeddings vs {len(records)} records"
            )

        # L2-normalize for cosine similarity via inner product
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        self.records = records

        print(
            f"[VectorStore] FAISS index built: "
            f"{self.index.ntotal} vectors, dim={dimension}"
        )

    @classmethod
    def from_embeddings(
        cls,
        records: List[Dict[str, str]],
        embeddings: np.ndarray,
    ) -> "FAISSVectorStore":
        """
        Convenience constructor: creates a store and builds the index in one call.

        Args:
            records:    list of dicts with at least 'source', 'chunk_id', 'text'.
            embeddings: numpy array of shape (n, dim).
        """
        store = cls()
        store.build_index(embeddings, records)
        return store

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
    ) -> List[Dict]:
        """
        Searches the FAISS index for the top-k most similar vectors.

        Args:
            query_embedding: numpy array of shape (1, dim).
            top_k:           number of results to return.

        Returns:
            List of dicts: {source, chunk_id, text, score}
        """
        if self.index is None:
            raise RuntimeError("Index has not been built yet. Call build_index first.")

        # Normalize query for cosine similarity
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx == -1:
                continue  # FAISS returns -1 for unfilled slots
            record = self.records[idx]
            results.append({
                "rank": rank + 1,
                "source": record["source"],
                "chunk_id": record["chunk_id"],
                "text": record["text"],
                "score": float(score),
            })

        return results
