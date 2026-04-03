"""
session11/ragas_eval.py
------------------------
RAGAS-based evaluation pipeline for Session 11.

Integrates with the RAGAS library to provide LLM-based evaluation metrics
for RAG pipeline outputs. RAGAS evaluates quality dimensions that simple
keyword metrics cannot capture:

    - Answer Relevancy:   Is the answer relevant to the question?
    - Faithfulness:       Is the answer grounded in the retrieved context?
    - Context Precision:  Are the retrieved chunks relevant to the question?
    - Context Recall:     Do the chunks cover the information in ground truth?

RAGAS Version Compatibility:
    This module is designed for ragas >= 0.1.0 (the refactored API).
    If you're using an older version, some metric names or APIs may differ.
    The module includes graceful fallbacks and clear error messages.

    Required package: pip install ragas

Usage:
    from app.rag.session11.ragas_eval import run_ragas_evaluation

    scores = run_ragas_evaluation(strategy="hybrid", dataset=eval_dataset)

Graceful degradation:
    If RAGAS is not installed or fails to import, all functions in this
    module will still work — they'll return placeholder scores and print
    a helpful installation message. No code path will crash.
"""

from typing import Dict, List, Optional
import os

from app.rag.session11.observability import run_strategy, ALL_STRATEGIES
from app.rag.session11.dataset import load_eval_dataset


# ---------------------------------------------------------------------------
# RAGAS import with graceful fallback
# ---------------------------------------------------------------------------

RAGAS_AVAILABLE = False
_RAGAS_IMPORT_ERROR = None

try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        answer_relevancy,
        faithfulness,
        context_precision,
        context_recall,
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
    RAGAS_METRICS = [answer_relevancy, faithfulness, context_precision, context_recall]
    RAGAS_METRIC_NAMES = ["answer_relevancy", "faithfulness", "context_precision", "context_recall"]
except ImportError as e:
    _RAGAS_IMPORT_ERROR = str(e)

    # Try alternative import for newer ragas versions
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            AnswerRelevancy,
            Faithfulness,
            ContextPrecision,
            ContextRecall,
        )
        from datasets import Dataset
        RAGAS_AVAILABLE = True
        RAGAS_METRICS = [AnswerRelevancy(), Faithfulness(), ContextPrecision(), ContextRecall()]
        RAGAS_METRIC_NAMES = ["answer_relevancy", "faithfulness", "context_precision", "context_recall"]
        _RAGAS_IMPORT_ERROR = None
    except ImportError as e2:
        _RAGAS_IMPORT_ERROR = str(e2)


def _print_ragas_not_available() -> None:
    """Prints a helpful message when RAGAS is not installed."""
    print("\n" + "=" * 70)
    print("  ⚠️  RAGAS NOT AVAILABLE")
    print("=" * 70)
    print(f"  Import error: {_RAGAS_IMPORT_ERROR}")
    print()
    print("  To install RAGAS, run:")
    print("    pip install ragas datasets")
    print()
    print("  Expected version: ragas >= 0.1.0")
    print("  Falling back to basic evaluation metrics.")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# RAGAS dataset preparation
# ---------------------------------------------------------------------------

def prepare_ragas_dataset(
    strategy_name: str,
    eval_dataset: List[Dict],
    pipeline=None,
    top_k: int = 5,
) -> Dict:
    """
    Runs a strategy on the eval dataset and prepares RAGAS-compatible data.

    RAGAS expects a HuggingFace Dataset with these columns:
        - question:      the user's question
        - answer:        the generated answer
        - contexts:      list of retrieved context strings
        - ground_truth:  the expected correct answer

    Args:
        strategy_name: one of the supported strategy names.
        eval_dataset:  list of eval items (from dataset.py).
        pipeline:      optional pre-built pipeline.
        top_k:         number of chunks to retrieve.

    Returns:
        Dict with keys matching RAGAS expected format:
            questions, answers, contexts, ground_truths
        Plus a 'results' key containing the raw run_strategy outputs.
    """
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    raw_results = []

    for i, item in enumerate(eval_dataset):
        print(f"  [{i+1}/{len(eval_dataset)}] Running: {item['question'][:60]}...")

        result = run_strategy(
            strategy_name=strategy_name,
            query=item["question"],
            pipeline=pipeline,
            top_k=top_k,
            debug=False,
        )

        questions.append(item["question"])
        answers.append(result.get("answer", ""))
        contexts.append([c.get("text", "") for c in result.get("contexts", [])])
        ground_truths.append(item["ground_truth"])
        raw_results.append(result)

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
        "results": raw_results,
    }


# ---------------------------------------------------------------------------
# RAGAS evaluation runner
# ---------------------------------------------------------------------------

def run_ragas_evaluation(
    strategy_name: str,
    eval_dataset: Optional[List[Dict]] = None,
    pipeline=None,
    top_k: int = 5,
) -> Dict:
    """
    Runs RAGAS evaluation for a single strategy on the eval dataset.

    Workflow:
        1. Load eval dataset (if not provided)
        2. Run the strategy on every question
        3. Format results for RAGAS
        4. Run RAGAS evaluation
        5. Return per-metric scores

    Args:
        strategy_name: one of the supported strategy names.
        eval_dataset:  optional eval items. Loaded from file if None.
        pipeline:      optional pre-built pipeline.
        top_k:         chunks per retrieval.

    Returns:
        Dict with:
            strategy:    the strategy name
            scores:      dict of metric_name → average score
            per_question: list of per-question score dicts (if available)
            ragas_used:  True if RAGAS was used, False if fallback
    """
    if eval_dataset is None:
        eval_dataset = load_eval_dataset()

    print(f"\n{'='*70}")
    print(f"  RAGAS EVALUATION — Strategy: {strategy_name}")
    print(f"  Questions: {len(eval_dataset)}")
    print(f"{'='*70}\n")

    # Step 1: Generate answers using the strategy
    print("  Step 1: Generating answers...")
    prepared = prepare_ragas_dataset(
        strategy_name=strategy_name,
        eval_dataset=eval_dataset,
        pipeline=pipeline,
        top_k=top_k,
    )

    # Step 2: Run RAGAS if available
    if RAGAS_AVAILABLE:
        return _run_with_ragas(strategy_name, prepared, eval_dataset)
    else:
        _print_ragas_not_available()
        return _run_with_fallback(strategy_name, prepared, eval_dataset)


def _run_with_ragas(
    strategy_name: str,
    prepared: Dict,
    eval_dataset: List[Dict],
) -> Dict:
    """
    Runs evaluation using the RAGAS library.

    This is the preferred path when RAGAS is installed.
    """
    print("\n  Step 2: Running RAGAS evaluation...")

    try:
        # Build HuggingFace Dataset
        hf_dataset = Dataset.from_dict({
            "question": prepared["question"],
            "answer": prepared["answer"],
            "contexts": prepared["contexts"],
            "ground_truth": prepared["ground_truth"],
        })

        # Run RAGAS
        ragas_result = ragas_evaluate(
            dataset=hf_dataset,
            metrics=RAGAS_METRICS,
        )

        # Extract scores
        scores = {}
        for name in RAGAS_METRIC_NAMES:
            scores[name] = ragas_result.get(name, 0.0)

        # Calculate overall
        valid = [v for v in scores.values() if v is not None and v > 0]
        scores["overall"] = sum(valid) / len(valid) if valid else 0.0

        print(f"\n  ✅ RAGAS evaluation complete for '{strategy_name}'")
        _print_scores(strategy_name, scores)

        return {
            "strategy": strategy_name,
            "scores": scores,
            "per_question": ragas_result.to_pandas().to_dict("records") if hasattr(ragas_result, "to_pandas") else [],
            "ragas_used": True,
        }

    except Exception as e:
        print(f"\n  ⚠️  RAGAS evaluation failed: {e}")
        print("  Falling back to basic evaluation...")
        return _run_with_fallback(strategy_name, prepared, eval_dataset)


def _run_with_fallback(
    strategy_name: str,
    prepared: Dict,
    eval_dataset: List[Dict],
) -> Dict:
    """
    Runs evaluation using basic metrics when RAGAS is not available.

    Uses the evaluator.py functions as a fallback.
    """
    from app.rag.session11.evaluator import evaluate_single

    print("\n  Step 2: Running basic evaluation (RAGAS fallback)...")

    all_scores = []
    for result, item in zip(prepared["results"], eval_dataset):
        score = evaluate_single(
            result=result,
            ground_truth=item["ground_truth"],
            expected_sources=item.get("expected_sources"),
        )
        all_scores.append(score)

    # Average across all questions
    avg_scores = {}
    metric_keys = ["answer_presence", "source_coverage", "context_hit_rate"]
    for key in metric_keys:
        values = [s[key] for s in all_scores if s.get(key) is not None]
        avg_scores[key] = sum(values) / len(values) if values else 0.0

    valid = [v for v in avg_scores.values() if v > 0]
    avg_scores["overall"] = sum(valid) / len(valid) if valid else 0.0

    print(f"\n  ✅ Basic evaluation complete for '{strategy_name}'")
    _print_scores(strategy_name, avg_scores)

    return {
        "strategy": strategy_name,
        "scores": avg_scores,
        "per_question": all_scores,
        "ragas_used": False,
    }


def _print_scores(strategy_name: str, scores: Dict) -> None:
    """Prints scores in a clean format."""
    print(f"\n  Scores for '{strategy_name}':")
    for name, value in scores.items():
        if value is not None:
            print(f"    {name:<25s} {value:.4f}")


# ---------------------------------------------------------------------------
# Multi-strategy RAGAS comparison
# ---------------------------------------------------------------------------

def run_ragas_comparison(
    strategies: Optional[List[str]] = None,
    eval_dataset: Optional[List[Dict]] = None,
    pipeline=None,
    top_k: int = 5,
) -> List[Dict]:
    """
    Runs RAGAS evaluation across multiple strategies and returns all scores.

    This is the main function for the comparison report.

    Args:
        strategies:   list of strategy names. Defaults to ALL_STRATEGIES.
        eval_dataset: optional eval items. Loaded from file if None.
        pipeline:     optional pre-built pipeline.
        top_k:        chunks per retrieval.

    Returns:
        List of evaluation result dicts (one per strategy).
    """
    if strategies is None:
        strategies = ALL_STRATEGIES

    if eval_dataset is None:
        eval_dataset = load_eval_dataset()

    print(f"\n{'#'*70}")
    print(f"  RAGAS MULTI-STRATEGY COMPARISON")
    print(f"  Strategies: {', '.join(strategies)}")
    print(f"  Questions:  {len(eval_dataset)}")
    print(f"{'#'*70}\n")

    all_results = []
    for strategy in strategies:
        result = run_ragas_evaluation(
            strategy_name=strategy,
            eval_dataset=eval_dataset,
            pipeline=pipeline,
            top_k=top_k,
        )
        all_results.append(result)

    # Print final comparison table
    _print_comparison_table(all_results)

    return all_results


def _print_comparison_table(results: List[Dict]) -> None:
    """Prints the final comparison score table."""
    if not results:
        return

    print(f"\n{'='*80}")
    print("  FINAL SCORE COMPARISON")
    print(f"{'='*80}")

    # Get all metric names from the first result
    sample_scores = results[0]["scores"]
    metric_names = [k for k in sample_scores.keys() if k != "overall"]

    # Header
    header_parts = [f"{'Strategy':<18s}"]
    for name in metric_names:
        # Shorten long metric names for display
        short = name[:12]
        header_parts.append(f"{short:>12s}")
    header_parts.append(f"{'Overall':>12s}")
    print("  " + " ".join(header_parts))
    print("  " + "-" * (18 + 13 * (len(metric_names) + 1)))

    # Rows
    for r in results:
        parts = [f"{r['strategy']:<18s}"]
        for name in metric_names:
            val = r["scores"].get(name, 0.0)
            if val is not None:
                parts.append(f"{val:>12.4f}")
            else:
                parts.append(f"{'N/A':>12s}")
        overall = r["scores"].get("overall", 0.0)
        parts.append(f"{overall:>12.4f}")
        print("  " + " ".join(parts))

    ragas_label = "RAGAS" if results[0].get("ragas_used") else "Basic (RAGAS fallback)"
    print(f"\n  Evaluation method: {ragas_label}")
    print(f"{'='*80}\n")
