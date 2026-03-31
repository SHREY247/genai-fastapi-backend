"""
session10/scoring.py
--------------------
Centralized score normalization for combining retrieval signals.

When fusing results from different retrievers (vector cosine similarity
vs. BM25 term-frequency scores), raw scores live on incompatible scales.
Normalization maps them to [0, 1] so they can be weighted and added.

IMPORTANT — Demo-Friendly Disclaimer:
    Min-max normalization is used here because it is simple and easy to
    explain in a teaching context. It has known limitations:

    1. Sensitive to outliers — a single extreme score skews the range.
    2. Unstable on tiny result sets — if top_k=1, min == max and all
       scores collapse to 0.0 (handled below, but still degenerate).
    3. Not comparable across queries — the normalization is per-query,
       so a score of 0.8 on query A means something different than 0.8
       on query B.

    In production, consider:
    - Z-score normalization (mean/std-based)
    - Reciprocal Rank Fusion (RRF) — rank-based, score-agnostic
    - Learned score calibration
"""

from typing import List, Dict
from copy import deepcopy


def normalize_scores(results: List[Dict]) -> List[Dict]:
    """
    Applies min-max normalization to the 'score' field of each result.

    Maps scores into [0, 1] range:
        normalized = (score - min) / (max - min)

    Edge cases:
        - Empty list       → returns []
        - All same scores  → all normalized to 1.0 (avoids division by zero)
        - Single result    → score set to 1.0

    Args:
        results: list of dicts, each must have a 'score' key.

    Returns:
        NEW list of dicts with 'score' replaced by normalized value.
        Original 'score' is preserved as 'raw_score'.
    """
    if not results:
        return []

    # Deep copy to avoid mutating the caller's data
    normalized = deepcopy(results)

    scores = [r["score"] for r in normalized]
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score

    for r in normalized:
        # Preserve the original score for debugging / display
        r["raw_score"] = r["score"]

        if score_range == 0:
            # All scores identical (or single result) → assign 1.0
            r["score"] = 1.0
        else:
            r["score"] = (r["score"] - min_score) / score_range

    return normalized
