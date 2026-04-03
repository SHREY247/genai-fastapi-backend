"""
session11/playground.py
------------------------
Classroom demo entry point for Session 11: RAG Evaluation & Observability.

Run with:
    python -m app.rag.session11.playground

Supports two demo modes:

    Mode A: Observability Demo (--observe)
        Runs one strategy on one query with full debug output.
        Shows every stage: query → rewrite → retrieval → prompt → answer.

    Mode B: Comparison Demo (--compare)
        Runs multiple strategies on the same query and prints
        side-by-side results with timing and chunk comparisons.

    Mode C: Evaluation Demo (--eval)
        Runs the evaluation dataset through all strategies and prints
        a final score comparison table.

    Mode D: Interactive Mode (--interactive)
        Lets the instructor type queries and see results live.

Examples:
    # Observability demo with debug output
    python -m app.rag.session11.playground --observe

    # Compare strategies on a demo query
    python -m app.rag.session11.playground --compare

    # Run full evaluation
    python -m app.rag.session11.playground --eval

    # Interactive mode
    python -m app.rag.session11.playground --interactive

    # Skip LLM calls (retrieval-only — fast classroom demo)
    python -m app.rag.session11.playground --compare --no-answers
    python -m app.rag.session11.playground --observe --no-answers
"""

import sys
import time

from app.rag.session11.observability import (
    run_strategy,
    run_all_strategies,
    get_pipeline,
    ALL_STRATEGIES,
    VECTOR,
    BM25,
    HYBRID,
    REWRITE_VECTOR,
    REWRITE_HYBRID,
)
from app.rag.session11.debug_logger import (
    print_debug_report,
    print_comparison_table,
    THICK_SEPARATOR,
    SEPARATOR,
)
from app.rag.session11.comparison import compare_single_query, compare_on_dataset
from app.rag.session11.dataset import load_eval_dataset, print_dataset_summary

# Session 12 Imports
try:
    from app.rag.session12.testing import run_basic_tests, evaluation_loop
    HAS_SESSION12 = True
except ImportError:
    HAS_SESSION12 = False


# ---------------------------------------------------------------------------
# Demo queries
# ---------------------------------------------------------------------------

DEMO_QUERIES = [
    "What are the interview rounds at Amazon for an SDE role?",
    "How should I prepare for a coding assessment?",
    "Tell me about the rounds",
    "Oracle OCI cloud interview questions",
]


# ---------------------------------------------------------------------------
# Mode A: Observability Demo
# ---------------------------------------------------------------------------

def demo_observability(pipeline, strategy: str = HYBRID, query: str = None, generate_answer: bool = True):
    """
    Demonstrates the observability layer by running a single strategy
    with full debug output.

    Shows students every stage of the RAG pipeline.
    Pass generate_answer=False to skip the LLM call (faster, retrieval-only).
    """
    if query is None:
        query = DEMO_QUERIES[0]

    mode_label = "RETRIEVAL ONLY" if not generate_answer else "with LLM answer"
    print(f"\n{'#'*70}")
    print(f"#  MODE A: OBSERVABILITY DEMO  ({mode_label})")
    print(f"#  Strategy: {strategy}")
    print(f"#  Query: \"{query}\"")
    print(f"{'#'*70}")

    result = run_strategy(
        strategy_name=strategy,
        query=query,
        pipeline=pipeline,
        top_k=5,
        debug=True,          # triggers the full debug report
        generate_answer=generate_answer,
    )

    return result


# ---------------------------------------------------------------------------
# Mode B: Comparison Demo
# ---------------------------------------------------------------------------

def demo_comparison(pipeline, query: str = None, strategies=None, generate_answer: bool = True):
    """
    Demonstrates cross-strategy comparison on a single query.

    Shows students how the same question gets different results
    from different retrieval strategies.
    Pass generate_answer=False to skip LLM calls (fast, retrieval-focused demo).
    """
    if query is None:
        query = DEMO_QUERIES[0]

    if strategies is None:
        strategies = ALL_STRATEGIES

    print(f"\n{'#'*70}")
    print(f"#  MODE B: COMPARISON DEMO")
    print(f"#  Query: \"{query}\"")
    print(f"#  Strategies: {', '.join(strategies)}")
    print(f"{'#'*70}")

    results = compare_single_query(
        query=query,
        strategies=strategies,
        pipeline=pipeline,
        top_k=5,
        show_chunks=True,
        show_answers=generate_answer,
        generate_answer=generate_answer,
    )

    return results


# ---------------------------------------------------------------------------
# Mode C: Evaluation Demo
# ---------------------------------------------------------------------------

def demo_evaluation(pipeline, use_ragas: bool = False, generate_answer: bool = True):
    """
    Demonstrates the evaluation pipeline on the full eval dataset.

    Runs all strategies, scores them, and prints a comparison table.
    Pass generate_answer=False to skip all LLM calls — only Source Coverage
    and Context Hit Rate are scored. Useful when Groq is rate-limited.
    """
    mode_label = "RETRIEVAL-ONLY metrics" if not generate_answer else "Basic metrics"
    print(f"\n{'#'*70}")
    print(f"#  MODE C: EVALUATION DEMO")
    print(f"#  Method: {'RAGAS' if use_ragas else mode_label}")
    print(f"{'#'*70}")

    # Show the dataset first
    print_dataset_summary()

    # Run comparison
    results = compare_on_dataset(
        pipeline=pipeline,
        use_ragas=use_ragas,
        generate_answer=generate_answer,
    )

    return results


# ---------------------------------------------------------------------------
# Mode E: Session 12 QA & Testing
# ---------------------------------------------------------------------------

def demo_testing(pipeline):
    """
    Demonstrates Session 12's 'RAG QA & Production Testing' logic.
    """
    if not HAS_SESSION12:
        print("\n[ERROR] Session 12 modules not found. Check app/rag/session12/.")
        return

    print(f"\n{'#'*70}")
    print(f"#  MODE E: SESSION 12 — QA & PRODUCTION TESTING")
    print(f"{'#'*70}")

    # 1. Run simulated unit tests
    run_basic_tests(strategy_name=HYBRID, pipeline=pipeline)

    # 2. Run pseudo production loop demo
    evaluation_loop(pipeline=pipeline)


# ---------------------------------------------------------------------------
# Mode D: Interactive
# ---------------------------------------------------------------------------

def demo_interactive(pipeline):
    """
    Interactive mode for live classroom demos.

    The instructor types a query, picks a mode, and sees results.
    """
    print(f"\n{'#'*70}")
    print(f"#  MODE D: INTERACTIVE")
    print(f"#  Commands:")
    print(f"#    Type a query to run comparison across all strategies")
    print(f"#    Prefix with 'debug:' for full observability output")
    print(f"#    Type 'quit' or 'exit' to stop")
    print(f"{'#'*70}")

    while True:
        try:
            raw_input = input("\n▸ Enter query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not raw_input or raw_input.lower() in ("quit", "exit", "q"):
            print("Exiting.")
            break

        if raw_input.lower().startswith("debug:"):
            # Observability mode
            query = raw_input[6:].strip()
            if not query:
                print("  Please provide a query after 'debug:'")
                continue

            for strategy in ALL_STRATEGIES:
                run_strategy(
                    strategy_name=strategy,
                    query=query,
                    pipeline=pipeline,
                    top_k=5,
                    debug=True,
                )
        else:
            # Comparison mode
            compare_single_query(
                query=raw_input,
                pipeline=pipeline,
                top_k=5,
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Entry point for the Session 11 playground.

    Parse CLI flags and run the appropriate demo mode.
    """
    args = set(sys.argv[1:])

    run_observe    = "--observe"     in args
    run_compare    = "--compare"     in args
    run_eval       = "--eval"        in args
    run_test       = "--test"        in args # Session 12
    run_interactive= "--interactive" in args
    use_ragas      = "--ragas"       in args
    generate_answer= "--no-answers" not in args  # False when --no-answers passed

    # Default: run observability + comparison demos
    if not any([run_observe, run_compare, run_eval, run_test, run_interactive]):
        run_observe = True
        run_compare = True

    print(f"\n{'#'*70}")
    print(f"#  SESSION 11: RAG EVALUATION & OBSERVABILITY — PLAYGROUND")
    if not generate_answer:
        print(f"#  Mode: RETRIEVAL ONLY (--no-answers)  — LLM calls skipped")
    print(f"{'#'*70}")
    print(f"\nBuilding pipeline (one-time setup)...\n")

    t0 = time.time()
    pipeline = get_pipeline(
        data_dir="data/interview_prep",
        chunk_size=200,
        overlap=30,
        provider="groq",
    )
    build_time = time.time() - t0
    print(f"\nPipeline built in {build_time:.1f}s\n")

    # Run selected modes
    if run_observe:
        demo_observability(pipeline, generate_answer=generate_answer)

        # Also show rewrite comparison
        print(f"\n{'─'*70}")
        print("  Now showing the same query with REWRITE + HYBRID...")
        print(f"{'─'*70}")
        demo_observability(pipeline, strategy=REWRITE_HYBRID, generate_answer=generate_answer)

    if run_compare:
        demo_comparison(pipeline, generate_answer=generate_answer)

        # Try a second query to show variety
        print(f"\n{'─'*70}")
        print("  Second comparison query...")
        print(f"{'─'*70}")
        demo_comparison(pipeline, query=DEMO_QUERIES[2], generate_answer=generate_answer)

    if run_eval:
        demo_evaluation(pipeline, use_ragas=use_ragas, generate_answer=generate_answer)

    if run_test:
        demo_testing(pipeline)

    if run_interactive:
        demo_interactive(pipeline)

    print(f"\n{THICK_SEPARATOR}")
    print("  Session 11 playground complete.")
    print(THICK_SEPARATOR + "\n")


if __name__ == "__main__":
    main()
