"""
retrieval_strategies/hybrid.py
-------------------------------
Hybrid retriever that fuses Vector (semantic) and BM25 (keyword) signals.

Fusion strategy:
    1. Run both retrievers independently
    2. Normalize scores to [0, 1] using min-max (see scoring.py)
    3. Combine: final_score = alpha * vector_score + beta * bm25_score
    4. Deduplicate by chunk_id (a chunk found by both gets the combined score)
    5. Re-rank by final_score and return top_k

alpha and beta are configurable. Common presets:
    - Equal weight:       alpha=0.5, beta=0.5  (balanced)
    - Semantic-heavy:     alpha=0.7, beta=0.3  (when queries are paraphrased)
    - Keyword-heavy:      alpha=0.3, beta=0.7  (when queries have exact terms)

The hybrid retriever does NOT rebuild indexes — it composes two
pre-built retrievers and only orchestrates scoring at query time.
"""

from typing import List, Dict
from copy import deepcopy

from app.rag.session10.retrieval_strategies.vector import VectorRetriever
from app.rag.session10.retrieval_strategies.bm25 import BM25Retriever
from app.rag.session10.scoring import normalize_scores


class HybridRetriever:
    """
    Combines Vector and BM25 retrievers using weighted score fusion.

    Both sub-retrievers must be pre-built before passing them in.
    The hybrid retriever only handles scoring logic at query time.

    Usage:
        vec = VectorRetriever(data_dir="data/interview_prep")
        bm  = BM25Retriever(data_dir="data/interview_prep")
        hyb = HybridRetriever(vector_retriever=vec, bm25_retriever=bm)

        results = hyb.retrieve("Amazon leadership principles", top_k=5)
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        bm25_retriever: BM25Retriever,
        alpha: float = 0.5,
        beta: float = 0.5,
    ):
        """
        Composes two pre-built retrievers with configurable weights.

        Args:
            vector_retriever: pre-built VectorRetriever instance.
            bm25_retriever:   pre-built BM25Retriever instance.
            alpha:            weight for vector scores.
            beta:             weight for BM25 scores.
        """
        self.vector = vector_retriever
        self.bm25 = bm25_retriever
        self.alpha = alpha
        self.beta = beta

        print(
            f"[HybridRetriever] Configured with alpha={alpha}, beta={beta} "
            f"(vector={alpha:.0%}, bm25={beta:.0%})"
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Fuses vector and BM25 results using weighted normalized scores.

        Steps:
            1. Retrieve from both sub-retrievers (fetch more than top_k
               to improve fusion quality)
            2. Normalize scores independently
            3. Merge by chunk_id — if a chunk appears in both, combine scores
            4. Sort by final_score, return top_k

        Args:
            query:  the user's question.
            top_k:  number of final results to return.

        Returns:
            List of dicts with: rank, source, source_type, company, page,
            chunk_id, text, score, retriever, vec_norm, bm25_norm

        Note on sub-scores:
            vec_norm and bm25_norm are POST-NORMALIZATION values (0-1 range),
            not raw retriever scores. They show how each signal contributed
            to the final fused score. See scoring.py for normalization details.
        """
        # Fetch more candidates than needed — fusion works better with
        # a larger candidate pool
        fetch_k = top_k * 2

        # Step 1: Independent retrieval
        vector_results = self.vector.retrieve(query, top_k=fetch_k)
        bm25_results = self.bm25.retrieve(query, top_k=fetch_k)

        # Step 2: Normalize scores to [0, 1]
        vector_normed = normalize_scores(vector_results)
        bm25_normed = normalize_scores(bm25_results)

        # Step 3: Merge into a unified dict keyed by chunk_id
        merged = {}

        for r in vector_normed:
            key = r["chunk_id"]
            entry = deepcopy(r)
            entry["vec_norm"] = r["score"]
            entry["bm25_norm"] = 0.0
            entry["score"] = self.alpha * r["score"]
            entry["retriever"] = "hybrid"
            merged[key] = entry

        for r in bm25_normed:
            key = r["chunk_id"]
            if key in merged:
                # Chunk found by BOTH retrievers — combine scores
                merged[key]["bm25_norm"] = r["score"]
                merged[key]["score"] += self.beta * r["score"]
            else:
                # Only found by BM25
                entry = deepcopy(r)
                entry["vec_norm"] = 0.0
                entry["bm25_norm"] = r["score"]
                entry["score"] = self.beta * r["score"]
                entry["retriever"] = "hybrid"
                merged[key] = entry

        # Step 4: Sort by combined score, take top_k
        fused = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
        fused = fused[:top_k]

        # Re-assign ranks
        for i, r in enumerate(fused, start=1):
            r["rank"] = i

        return fused
