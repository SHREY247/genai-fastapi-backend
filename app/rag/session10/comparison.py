"""
session10/comparison.py
------------------------
Multi-strategy comparison runner.

Runs the SAME query across all retrieval strategies and displays
side-by-side diagnostics: retrieved chunks, scores, latency breakdown,
and (optionally) LLM-generated answers.

Latency is split into:
    - Rewrite:   time spent rewriting the query (LLM call)
    - Retrieval: time spent searching the index (the focus of Session 10)
    - Answer:    time spent generating the LLM answer
    - Total:     wall-clock time for the full pipeline

This separation is critical for teaching: students need to see that
retrieval (the thing we're optimizing) is fast, and the LLM call
typically dominates total latency.

This is the orchestration layer for comparisons. The pipeline.py
handles single-strategy queries; comparison.py handles multi-strategy
evaluation.

Usage:
    from app.rag.session10.comparison import run_comparison
    run_comparison("Amazon interview process", pipeline=pipe)
"""

import time
from typing import Dict, List, Optional

from app.rag.session10.pipeline import Session10Pipeline


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _format_chunk_preview(text: str, max_chars: int = 120) -> str:
    """Truncates chunk text for display, preserving word boundaries."""
    if len(text) <= max_chars:
        return text.replace("\n", " ")
    truncated = text[:max_chars].rsplit(" ", 1)[0]
    return truncated.replace("\n", " ") + "..."


def _format_timing(timing: Dict) -> str:
    """Formats the timing breakdown into a readable one-liner."""
    parts = []
    if timing.get("rewrite_ms", 0) > 0:
        parts.append(f"rewrite={timing['rewrite_ms']:.0f}ms")
    parts.append(f"retrieval={timing['retrieval_ms']:.0f}ms")
    parts.append(f"answer={timing['answer_ms']:.0f}ms")
    parts.append(f"total={timing['total_ms']:.0f}ms")
    return " | ".join(parts)


def _print_results_block(
    label: str,
    results: List[Dict],
    timing: Dict,
    answer: Optional[str] = None,
):
    """
    Prints a formatted block of retrieval results for one strategy.

    Shows: timing breakdown, source, chunk_id, score, and a text preview
    for each chunk.
    """
    print(f"\n--- {label} ---")
    print(f"  {_format_timing(timing)} | Results: {len(results)}")
    print()

    for r in results:
        source = r.get("source", "?")
        chunk_id = r.get("chunk_id", "?")
        score = r.get("score", 0.0)
        preview = _format_chunk_preview(r.get("text", ""))

        # Show normalized sub-scores for hybrid results
        # These are POST-NORMALIZATION fusion inputs, not raw retriever scores
        extra = ""
        if "vec_norm" in r and "bm25_norm" in r:
            extra = f" (vec_norm={r['vec_norm']:.3f}, bm25_norm={r['bm25_norm']:.3f})"

        print(f"  [{chunk_id}] {source} | score={score:.4f}{extra}")
        print(f"    > {preview}")
        print()

    if answer:
        print(f"  Answer: {answer[:300]}{'...' if len(answer) > 300 else ''}")
        print()


# ---------------------------------------------------------------------------
# Main comparison function
# ---------------------------------------------------------------------------

def run_comparison(
    query: str,
    pipeline: Session10Pipeline,
    top_k: int = 5,
    include_answers: bool = True,
) -> Dict:
    """
    Runs the same query across all 4 modes and prints comparison output.

    Modes:
        1. vector         -- semantic retrieval only
        2. bm25           -- keyword retrieval only
        3. hybrid         -- fused vector + BM25
        4. hybrid_rewrite -- hybrid with query rewriting

    Latency is broken down per phase (rewrite, retrieval, answer) so
    students can see WHERE time is spent.

    Args:
        query:            the user's question.
        pipeline:         a pre-built Session10Pipeline instance.
        top_k:            number of chunks to retrieve per strategy.
        include_answers:  if True, calls the LLM for each strategy.

    Returns:
        Dict keyed by strategy name, each containing:
            results, answer, timing_ms
    """
    print("\n" + "=" * 70)
    print(f"  QUERY: \"{query}\"")
    print("=" * 70)

    all_results = {}

    # ---- Mode 1: Vector ----
    vec_out = pipeline.query(query, strategy="vector", rewrite=False, top_k=top_k)
    all_results["vector"] = {
        "results": vec_out["results"],
        "answer": vec_out["answer"] if include_answers else None,
        "timing_ms": vec_out["timing_ms"],
    }
    _print_results_block(
        "VECTOR (Semantic)",
        vec_out["results"],
        vec_out["timing_ms"],
        vec_out["answer"] if include_answers else None,
    )

    # ---- Mode 2: BM25 ----
    bm25_out = pipeline.query(query, strategy="bm25", rewrite=False, top_k=top_k)
    all_results["bm25"] = {
        "results": bm25_out["results"],
        "answer": bm25_out["answer"] if include_answers else None,
        "timing_ms": bm25_out["timing_ms"],
    }
    _print_results_block(
        "BM25 (Keyword)",
        bm25_out["results"],
        bm25_out["timing_ms"],
        bm25_out["answer"] if include_answers else None,
    )

    # ---- Mode 3: Hybrid ----
    hyb_out = pipeline.query(query, strategy="hybrid", rewrite=False, top_k=top_k)
    all_results["hybrid"] = {
        "results": hyb_out["results"],
        "answer": hyb_out["answer"] if include_answers else None,
        "timing_ms": hyb_out["timing_ms"],
    }
    _print_results_block(
        "HYBRID (Vector + BM25)",
        hyb_out["results"],
        hyb_out["timing_ms"],
        hyb_out["answer"] if include_answers else None,
    )

    # ---- Mode 4: Hybrid + Rewrite ----
    rw_out = pipeline.query(query, strategy="hybrid", rewrite=True, top_k=top_k)
    all_results["hybrid_rewrite"] = {
        "results": rw_out["results"],
        "answer": rw_out["answer"] if include_answers else None,
        "timing_ms": rw_out["timing_ms"],
        "rewritten_query": rw_out["rewritten_query"],
    }

    # Show the rewritten query BEFORE the results
    if rw_out["rewritten_query"]:
        print(f"\n  [Rewritten] \"{rw_out['rewritten_query']}\"")

    _print_results_block(
        "HYBRID + REWRITE",
        rw_out["results"],
        rw_out["timing_ms"],
        rw_out["answer"] if include_answers else None,
    )

    # ---- Latency Summary ----
    print("=" * 70)
    print("  LATENCY BREAKDOWN (ms)")
    print("-" * 70)
    print(f"  {'Strategy':20s} {'Rewrite':>8s} {'Retrieval':>10s} {'Answer':>8s} {'Total':>8s}")
    print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")

    for name, data in all_results.items():
        t = data["timing_ms"]
        rewrite = t.get("rewrite_ms", 0)
        retrieval = t["retrieval_ms"]
        answer = t["answer_ms"]
        total = t["total_ms"]
        print(
            f"  {name:20s} {rewrite:8.0f} {retrieval:10.0f} {answer:8.0f} {total:8.0f}"
        )

    print("=" * 70 + "\n")

    return all_results


def run_failure_case_comparison(
    pipeline: Session10Pipeline,
    top_k: int = 3,
    include_answers: bool = False,
):
    """
    Runs comparison for all pre-defined failure cases.

    Useful for instructor demos: iterates through each test case,
    shows which strategy was expected to win, and runs the comparison.

    Args:
        pipeline:         pre-built Session10Pipeline instance.
        top_k:            chunks per strategy.
        include_answers:  if True, generates LLM answers (slower).
    """
    from app.rag.session10.failure_cases import TEST_QUERIES

    for i, tc in enumerate(TEST_QUERIES, start=1):
        print(f"\n{'#' * 70}")
        print(f"  FAILURE CASE {i}/{len(TEST_QUERIES)}")
        print(f"  Expected best: {tc['best_strategy']}")
        print(f"  Why: {tc['why'][:100]}...")
        print(f"{'#' * 70}")

        run_comparison(
            tc["query"],
            pipeline=pipeline,
            top_k=top_k,
            include_answers=include_answers,
        )
