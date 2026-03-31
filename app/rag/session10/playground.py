"""
session10/playground.py
------------------------
Main demo script for Session 10: Retrieval Optimization.

This is the entry point for live demos. Run it with:

    python -m app.rag.session10.playground

It will:
    1. Build all retrievers (one-time setup)
    2. Run a set of pre-defined demo queries with full diagnostics
    3. Optionally enter interactive mode for custom queries

Output format per strategy:
    - Source / chunk_id
    - Score (normalized, and sub-scores for hybrid)
    - Text preview
    - Latency
    - LLM answer (if available)
"""

import sys
import time

from app.rag.session10.pipeline import Session10Pipeline
from app.rag.session10.comparison import run_comparison, run_failure_case_comparison
from app.rag.session10.failure_cases import TEST_QUERIES


# ---------------------------------------------------------------------------
# Demo queries — a curated subset for quick demonstrations
# ---------------------------------------------------------------------------

DEMO_QUERIES = [
    # Keyword-heavy: BM25 should win
    "Oracle OCI cloud interview questions",
    # Semantic: Vector should win
    "How should I prepare for a coding assessment?",
    # Ambiguous: Hybrid + Rewrite should win
    "Tell me about the rounds",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Entry point for the Session 10 playground.

    Modes:
        --demo       Run pre-defined demo queries (default)
        --failures   Run all failure case comparisons
        --interactive  Enter interactive query mode
        --no-answers   Skip LLM answer generation (faster)
    """
    # Parse simple CLI flags
    args = set(sys.argv[1:])
    run_demo = "--demo" in args or (not args & {"--failures", "--interactive"})
    run_failures = "--failures" in args
    run_interactive = "--interactive" in args
    include_answers = "--no-answers" not in args

    # ---- Step 1: Build pipeline (one-time) ----
    print("\n" + "#" * 70)
    print("#  SESSION 10: RETRIEVAL OPTIMIZATION — PLAYGROUND")
    print("#" * 70)
    print("\nBuilding pipeline (this takes a moment on first run)...\n")

    t0 = time.time()
    pipeline = Session10Pipeline(
        data_dir="data/interview_prep",
        chunk_size=200,
        overlap=30,
        provider="groq",
    )
    build_time = time.time() - t0
    print(f"\nPipeline built in {build_time:.1f}s\n")

    # ---- Step 2: Run demo queries ----
    if run_demo:
        print("\n" + "#" * 70)
        print("#  DEMO QUERIES")
        print("#  Running each query across: Vector, BM25, Hybrid, Hybrid+Rewrite")
        print("#" * 70)

        for query in DEMO_QUERIES:
            run_comparison(
                query,
                pipeline=pipeline,
                top_k=3,
                include_answers=include_answers,
            )

    # ---- Step 3: Run failure cases ----
    if run_failures:
        print("\n" + "#" * 70)
        print("#  FAILURE CASE ANALYSIS")
        print(f"#  Running {len(TEST_QUERIES)} pre-defined failure cases")
        print("#" * 70)

        run_failure_case_comparison(
            pipeline=pipeline,
            top_k=3,
            include_answers=include_answers,
        )

    # ---- Step 4: Interactive mode ----
    if run_interactive:
        print("\n" + "#" * 70)
        print("#  INTERACTIVE MODE")
        print("#  Type a query and see results across all strategies.")
        print("#  Type 'quit' or 'exit' to stop.")
        print("#" * 70)

        while True:
            try:
                query = input("\nEnter query: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not query or query.lower() in ("quit", "exit", "q"):
                print("Exiting.")
                break

            run_comparison(
                query,
                pipeline=pipeline,
                top_k=5,
                include_answers=include_answers,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
