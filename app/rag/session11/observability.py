"""
session11/observability.py
----------------------------
Observability layer for RAG pipeline runs.

Wraps Session 10's pipeline to produce structured, inspectable result
dictionaries that expose every stage of a RAG pipeline run:

    query → rewrite → retrieval → prompt construction → LLM answer

The key abstraction is `run_strategy()` — a single function that accepts
a strategy name string and returns a normalized result dict. This is what
the evaluation and comparison modules call.

Supported strategy names:
    "vector"         — semantic retrieval only
    "bm25"           — keyword retrieval only
    "hybrid"         — fused vector + BM25
    "rewrite_vector" — query rewriting + vector retrieval
    "rewrite_hybrid" — query rewriting + hybrid retrieval

Design:
    This module does NOT re-implement retrieval. It delegates to
    Session10Pipeline and normalizes the output into a consistent shape
    that downstream consumers (evaluator, comparison, debug logger) expect.
"""

import time
from typing import Dict, List, Optional

from app.rag.session10.pipeline import Session10Pipeline
from app.rag.session11.debug_logger import print_debug_report


# ---------------------------------------------------------------------------
# Strategy name constants
# ---------------------------------------------------------------------------

VECTOR = "vector"
BM25 = "bm25"
HYBRID = "hybrid"
REWRITE_VECTOR = "rewrite_vector"
REWRITE_HYBRID = "rewrite_hybrid"

ALL_STRATEGIES = [VECTOR, BM25, HYBRID, REWRITE_VECTOR, REWRITE_HYBRID]

# Maps strategy name → (base_strategy, rewrite_flag)
_STRATEGY_CONFIG = {
    VECTOR:         ("vector",  False),
    BM25:           ("bm25",    False),
    HYBRID:         ("hybrid",  False),
    REWRITE_VECTOR: ("vector",  True),
    REWRITE_HYBRID: ("hybrid",  True),
}


# ---------------------------------------------------------------------------
# Pipeline singleton management
# ---------------------------------------------------------------------------

_pipeline_instance: Optional[Session10Pipeline] = None


def get_pipeline(**kwargs) -> Session10Pipeline:
    """
    Returns a shared Session10Pipeline instance (lazy singleton).

    The pipeline is expensive to build (loads corpus, builds indexes),
    so we create it once and reuse it across all strategy runs.

    Args:
        **kwargs: forwarded to Session10Pipeline on first creation.

    Returns:
        A ready-to-use Session10Pipeline instance.
    """
    global _pipeline_instance
    if _pipeline_instance is None:
        print("\n[Observability] Building Session 10 pipeline (one-time setup)...")
        _pipeline_instance = Session10Pipeline(**kwargs)
    return _pipeline_instance


def reset_pipeline() -> None:
    """Clears the cached pipeline instance. Useful for testing."""
    global _pipeline_instance
    _pipeline_instance = None


# ---------------------------------------------------------------------------
# Core: run_strategy
# ---------------------------------------------------------------------------

def run_strategy(
    strategy_name: str,
    query: str,
    pipeline: Optional[Session10Pipeline] = None,
    top_k: int = 5,
    debug: bool = False,
) -> Dict:
    """
    Runs a single query through the specified retrieval strategy and
    returns a structured, inspectable result dictionary.

    This is the MAIN ENTRY POINT for Session 11. All evaluation,
    comparison, and demo code calls this function.

    Args:
        strategy_name: one of VECTOR, BM25, HYBRID, REWRITE_VECTOR, REWRITE_HYBRID.
        query:         the user's question.
        pipeline:      optional pre-built Session10Pipeline. If None, uses
                       the shared singleton.
        top_k:         number of chunks to retrieve.
        debug:         if True, prints a full debug report to terminal.

    Returns:
        Dict with keys:
            query            — original user query
            rewritten_query  — rewritten query (None if no rewriting)
            strategy         — strategy name used
            contexts         — list of retrieved chunk dicts
            context_metadata — list of metadata dicts for each chunk
            prompt           — the full prompt sent to the LLM
            answer           — the LLM-generated answer
            timing_ms        — dict with rewrite_ms, retrieval_ms,
                               answer_ms, total_ms

    Raises:
        ValueError: if strategy_name is not recognized.
    """
    if strategy_name not in _STRATEGY_CONFIG:
        raise ValueError(
            f"Unknown strategy '{strategy_name}'. "
            f"Choose from: {ALL_STRATEGIES}"
        )

    base_strategy, use_rewrite = _STRATEGY_CONFIG[strategy_name]

    # Get or build the pipeline
    pipe = pipeline or get_pipeline()

    # Delegate to Session 10 pipeline
    raw = pipe.query(
        query=query,
        strategy=base_strategy,
        rewrite=use_rewrite,
        top_k=top_k,
    )

    # Extract metadata from each chunk
    context_metadata = []
    for chunk in raw.get("results", []):
        context_metadata.append({
            "source": chunk.get("source", "unknown"),
            "source_type": chunk.get("source_type", "unknown"),
            "company": chunk.get("company", "unknown"),
            "chunk_id": chunk.get("chunk_id", "?"),
            "page": chunk.get("page"),
            "score": chunk.get("score", 0.0),
            "retriever": chunk.get("retriever", "unknown"),
        })

    # Build normalized result
    result = {
        "query": raw["query"],
        "rewritten_query": raw.get("rewritten_query"),
        "strategy": strategy_name,
        "contexts": raw.get("results", []),
        "context_metadata": context_metadata,
        "prompt": raw.get("context", ""),  # the assembled context string
        "answer": raw.get("answer", ""),
        "timing_ms": raw.get("timing_ms", {}),
    }

    # Optional: print debug report
    if debug:
        print_debug_report(result)

    return result


def run_all_strategies(
    query: str,
    strategies: Optional[List[str]] = None,
    pipeline: Optional[Session10Pipeline] = None,
    top_k: int = 5,
    debug: bool = False,
) -> List[Dict]:
    """
    Runs a query through multiple strategies and returns all results.

    Convenience wrapper over run_strategy() for batch evaluation.

    Args:
        query:      the user's question.
        strategies: list of strategy names. Defaults to ALL_STRATEGIES.
        pipeline:   optional pre-built pipeline.
        top_k:      chunks per strategy.
        debug:      if True, prints debug report for each strategy.

    Returns:
        List of result dicts (one per strategy), in the same order.
    """
    if strategies is None:
        strategies = ALL_STRATEGIES

    results = []
    for name in strategies:
        result = run_strategy(
            strategy_name=name,
            query=query,
            pipeline=pipeline,
            top_k=top_k,
            debug=debug,
        )
        results.append(result)

    return results
