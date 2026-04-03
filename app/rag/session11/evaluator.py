"""
session11/evaluator.py
-----------------------
Lightweight evaluation utilities that work WITHOUT RAGAS.

Provides basic scoring functions that can be used as a fallback when
RAGAS is not installed, or as supplementary metrics alongside RAGAS.

Metrics implemented:
    1. Source coverage  — did the retrieved chunks come from expected sources?
    2. Answer presence  — does the answer contain key terms from ground truth?
    3. Context hit rate — what fraction of retrieved chunks are from expected sources?

These are intentionally simple. The point is to show students that
even basic metrics give useful signal, and RAGAS adds more sophisticated
LLM-based evaluation on top.

Usage:
    from app.rag.session11.evaluator import evaluate_single, evaluate_batch

    score = evaluate_single(result_dict, ground_truth, expected_sources)
"""

from typing import Dict, List, Optional
import re


def _normalize_text(text: str) -> str:
    """Lowercases and strips extra whitespace for comparison."""
    return re.sub(r"\s+", " ", text.lower().strip())


def score_source_coverage(
    context_metadata: List[Dict],
    expected_sources: List[str],
) -> float:
    """
    Measures what fraction of expected sources appear in retrieved chunks.

    A score of 1.0 means every expected source was retrieved.
    A score of 0.0 means none of the expected sources were found.

    Args:
        context_metadata: list of chunk metadata dicts (must have 'source' key).
        expected_sources: list of expected source filenames.

    Returns:
        Float between 0.0 and 1.0.
    """
    if not expected_sources:
        return 1.0  # no expectations, trivially satisfied

    retrieved_sources = {m.get("source", "") for m in context_metadata}

    hits = sum(
        1 for src in expected_sources
        if any(src in rs for rs in retrieved_sources)
    )

    return hits / len(expected_sources)


def score_answer_presence(
    answer: str,
    ground_truth: str,
    min_term_length: int = 4,
) -> float:
    """
    Measures what fraction of significant terms from the ground truth
    appear in the generated answer.

    This is a crude keyword-overlap metric — NOT a substitute for
    LLM-based evaluation. But it's useful for quick sanity checks.

    Args:
        answer:          the generated answer text.
        ground_truth:    the expected answer text.
        min_term_length: minimum word length to consider significant.

    Returns:
        Float between 0.0 and 1.0.
    """
    if not ground_truth or not answer:
        return 0.0

    norm_answer = _normalize_text(answer)
    norm_truth = _normalize_text(ground_truth)

    # Extract significant terms from ground truth
    truth_terms = [
        w for w in norm_truth.split()
        if len(w) >= min_term_length
    ]

    if not truth_terms:
        return 1.0  # no significant terms to check

    hits = sum(1 for term in truth_terms if term in norm_answer)
    return hits / len(truth_terms)


def score_context_hit_rate(
    contexts: List[Dict],
    expected_sources: List[str],
) -> float:
    """
    Measures what fraction of retrieved chunks come from expected sources.

    A high score means the retriever is focused on relevant documents.
    A low score means it's retrieving noise.

    Args:
        contexts:         list of chunk dicts (must have 'source' key).
        expected_sources: list of expected source filenames.

    Returns:
        Float between 0.0 and 1.0.
    """
    if not contexts or not expected_sources:
        return 0.0

    hits = sum(
        1 for chunk in contexts
        if any(src in chunk.get("source", "") for src in expected_sources)
    )

    return hits / len(contexts)


def evaluate_single(
    result: Dict,
    ground_truth: str,
    expected_sources: Optional[List[str]] = None,
) -> Dict:
    """
    Evaluates a single pipeline result against ground truth.

    Returns a dict of metric scores.

    Args:
        result:           structured result dict from run_strategy().
        ground_truth:     the expected correct answer.
        expected_sources: optional list of expected source filenames.

    Returns:
        Dict with keys:
            answer_presence:   float — keyword overlap with ground truth
            source_coverage:   float — expected sources found in retrieval
            context_hit_rate:  float — fraction of chunks from expected sources
            overall:           float — simple average of all metrics
    """
    answer = result.get("answer", "")
    contexts = result.get("contexts", [])
    metadata = result.get("context_metadata", contexts)

    scores = {
        "answer_presence": score_answer_presence(answer, ground_truth),
    }

    if expected_sources:
        scores["source_coverage"] = score_source_coverage(metadata, expected_sources)
        scores["context_hit_rate"] = score_context_hit_rate(contexts, expected_sources)
    else:
        scores["source_coverage"] = None
        scores["context_hit_rate"] = None

    # Overall = average of non-None scores
    valid_scores = [v for v in scores.values() if v is not None]
    scores["overall"] = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    return scores


def evaluate_batch(
    results: List[Dict],
    dataset: List[Dict],
) -> List[Dict]:
    """
    Evaluates a batch of results against the evaluation dataset.

    Each result is matched to its corresponding dataset item by index.

    Args:
        results: list of result dicts from run_strategy().
        dataset: list of eval items (with 'ground_truth', 'expected_sources').

    Returns:
        List of score dicts (one per result).
    """
    scores = []
    for result, eval_item in zip(results, dataset):
        score = evaluate_single(
            result,
            ground_truth=eval_item["ground_truth"],
            expected_sources=eval_item.get("expected_sources"),
        )
        scores.append(score)

    return scores
