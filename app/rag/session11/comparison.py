"""
session11/comparison.py
------------------------
Cross-strategy comparison orchestration for Session 11.

Runs multiple retrieval strategies on the same query or evaluation dataset
and produces structured comparison reports.

This module builds on:
    - observability.py for running strategies
    - evaluator.py for basic scoring
    - ragas_eval.py for LLM-based scoring
    - debug_logger.py for pretty output

Two main entry points:
    1. compare_single_query()  — compare strategies on one question
    2. compare_on_dataset()    — compare strategies on the full eval set

Usage:
    from app.rag.session11.comparison import compare_single_query, compare_on_dataset

    # Quick single-query comparison
    compare_single_query("How does Google evaluate candidates?")

    # Full dataset evaluation comparison
    compare_on_dataset()
"""

from typing import Dict, List, Optional

from app.rag.session11.observability import (
    run_strategy,
    run_all_strategies,
    ALL_STRATEGIES,
    get_pipeline,
)
from app.rag.session11.debug_logger import (
    truncate,
    print_comparison_table,
    THICK_SEPARATOR,
    SEPARATOR,
)
from app.rag.session11.evaluator import evaluate_single
from app.rag.session11.dataset import load_eval_dataset


# ---------------------------------------------------------------------------
# Single-query comparison
# ---------------------------------------------------------------------------

def compare_single_query(
    query: str,
    strategies: Optional[List[str]] = None,
    pipeline=None,
    top_k: int = 5,
    show_chunks: bool = True,
    show_answers: bool = True,
    generate_answer: bool = True,
    max_chunk_chars: int = 150,
    max_answer_chars: int = 300,
) -> List[Dict]:
    """
    Runs a single query through multiple strategies and prints comparison.

    This is the best function for classroom demos — it shows how the same
    question gets different results from different retrieval strategies.

    Args:
        query:            the user's question.
        strategies:       list of strategy names. Defaults to ALL_STRATEGIES.
        pipeline:         optional pre-built pipeline.
        top_k:            chunks per strategy.
        show_chunks:      if True, shows top retrieved chunks per strategy.
        show_answers:     if True, shows LLM answers per strategy.
        generate_answer:  if False, skips LLM generation for all strategies.
                          Overrides show_answers when False.
        max_chunk_chars:  max characters per chunk preview.
        max_answer_chars: max characters for answer preview.

    Returns:
        List of result dicts (one per strategy).
    """
    if strategies is None:
        strategies = ALL_STRATEGIES

    # generate_answer=False implies no answers to show
    if not generate_answer:
        show_answers = False

    pipe = pipeline or get_pipeline()

    mode_label = "RETRIEVAL ONLY" if not generate_answer else "with LLM answers"
    print(f"\n{THICK_SEPARATOR}")
    print(f"  📊 SINGLE-QUERY COMPARISON  ({mode_label})")
    print(f"  Query: \"{query}\"")
    print(f"  Strategies: {', '.join(strategies)}")
    print(THICK_SEPARATOR)

    results = []
    for strategy in strategies:
        result = run_strategy(
            strategy_name=strategy,
            query=query,
            pipeline=pipe,
            top_k=top_k,
            debug=False,
            generate_answer=generate_answer,
        )
        results.append(result)

        # Print per-strategy block
        print(f"\n{SEPARATOR}")
        print(f"  ▸ {strategy.upper()}")
        print(SEPARATOR)

        # Show rewritten query if applicable
        rq = result.get("rewritten_query")
        if rq:
            print(f"  Rewritten: \"{rq}\"")

        # Show top chunks
        if show_chunks:
            chunks = result.get("contexts", [])[:3]
            for i, chunk in enumerate(chunks, start=1):
                rank   = chunk.get("rank", i)
                source = chunk.get("source", "?")
                score  = chunk.get("score", 0.0)
                text   = truncate(chunk.get("text", ""), max_chunk_chars)
                print(f"    Rank {rank}  {source}  (score={score:.4f})")
                print(f"    {'─'*62}")
                print(f"    {text}")

        # Show answer or retrieval-only notice
        if show_answers:
            answer = result.get("answer", "")
            print(f"\n  Answer: {truncate(answer, max_answer_chars)}")
        elif not generate_answer:
            print("\n  [RETRIEVAL ONLY — answer generation skipped]")

    # Print timing comparison
    print_comparison_table(results)

    return results


# ---------------------------------------------------------------------------
# Dataset comparison
# ---------------------------------------------------------------------------

def compare_on_dataset(
    strategies: Optional[List[str]] = None,
    eval_dataset: Optional[List[Dict]] = None,
    pipeline=None,
    top_k: int = 5,
    use_ragas: bool = False,
) -> Dict:
    """
    Compares strategies across the full evaluation dataset.

    For each strategy, runs all eval questions, scores the results,
    and prints a summary table.

    Args:
        strategies:   list of strategy names. Defaults to ALL_STRATEGIES.
        eval_dataset: optional eval items. Loaded from file if None.
        pipeline:     optional pre-built pipeline.
        top_k:        chunks per strategy.
        use_ragas:    if True, uses RAGAS for evaluation (requires ragas package).

    Returns:
        Dict keyed by strategy name, each containing:
            results:  list of result dicts
            scores:   dict of averaged metric scores
    """
    if strategies is None:
        strategies = ALL_STRATEGIES

    if eval_dataset is None:
        eval_dataset = load_eval_dataset()

    pipe = pipeline or get_pipeline()

    print(f"\n{'#'*70}")
    print(f"  DATASET COMPARISON")
    print(f"  Strategies: {', '.join(strategies)}")
    print(f"  Questions:  {len(eval_dataset)}")
    print(f"{'#'*70}\n")

    # If RAGAS requested, delegate to ragas_eval
    if use_ragas:
        from app.rag.session11.ragas_eval import run_ragas_comparison
        return run_ragas_comparison(
            strategies=strategies,
            eval_dataset=eval_dataset,
            pipeline=pipe,
            top_k=top_k,
        )

    # Otherwise, use basic evaluation
    all_strategy_results = {}

    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"  Evaluating: {strategy}")
        print(f"{'='*60}")

        strategy_results = []
        strategy_scores = []

        for i, item in enumerate(eval_dataset):
            print(f"  [{i+1}/{len(eval_dataset)}] {item['question'][:50]}...")

            result = run_strategy(
                strategy_name=strategy,
                query=item["question"],
                pipeline=pipe,
                top_k=top_k,
                debug=False,
            )
            strategy_results.append(result)

            score = evaluate_single(
                result=result,
                ground_truth=item["ground_truth"],
                expected_sources=item.get("expected_sources"),
            )
            strategy_scores.append(score)

        # Average scores
        avg_scores = _average_scores(strategy_scores)

        all_strategy_results[strategy] = {
            "results": strategy_results,
            "scores": avg_scores,
            "per_question_scores": strategy_scores,
        }

        print(f"\n  ✅ {strategy}: overall={avg_scores.get('overall', 0):.4f}")

    # Print final comparison table
    _print_dataset_comparison_table(all_strategy_results)

    return all_strategy_results


def _average_scores(scores: List[Dict]) -> Dict:
    """Averages a list of score dicts across all questions."""
    if not scores:
        return {}

    # Collect all metric keys
    all_keys = set()
    for s in scores:
        all_keys.update(s.keys())

    avg = {}
    for key in all_keys:
        values = [s[key] for s in scores if s.get(key) is not None]
        avg[key] = sum(values) / len(values) if values else 0.0

    return avg


def _print_dataset_comparison_table(all_results: Dict) -> None:
    """Prints the final dataset comparison score table."""
    if not all_results:
        return

    print(f"\n{'='*80}")
    print("  FINAL SCORE COMPARISON (Basic Evaluation)")
    print(f"{'='*80}")

    # Determine metric columns from first strategy's scores
    first_strategy = next(iter(all_results.values()))
    metric_names = [k for k in first_strategy["scores"].keys() if k != "overall"]

    # Header
    header_parts = [f"{'Strategy':<18s}"]
    for name in metric_names:
        short = name.replace("_", " ").title()[:14]
        header_parts.append(f"{short:>14s}")
    header_parts.append(f"{'Overall':>14s}")
    print("  " + " ".join(header_parts))
    print("  " + "-" * (18 + 15 * (len(metric_names) + 1)))

    # Rows
    for strategy_name, data in all_results.items():
        scores = data["scores"]
        parts = [f"{strategy_name:<18s}"]
        for name in metric_names:
            val = scores.get(name, 0.0)
            if val is not None:
                parts.append(f"{val:>14.4f}")
            else:
                parts.append(f"{'N/A':>14s}")
        overall = scores.get("overall", 0.0)
        parts.append(f"{overall:>14.4f}")
        print("  " + " ".join(parts))

    print(f"\n{'='*80}\n")
