"""
session12/ragas_wrapper.py
--------------------------
Minimal wrapper around the Session 11 RAGAS integration for Session 12.

Provides:
- run_ragas_eval() — Simple function to trigger RAGAS evaluation.
"""

from typing import Dict, List, Optional
from app.rag.session11.ragas_eval import run_ragas_comparison
from app.rag.session11.dataset import load_eval_dataset

def run_ragas_eval(pipeline=None, strategies: Optional[List[str]] = None, num_questions: int = 3):
    """
    Minimal wrapper around existing RAGAS evaluation.
    Evaluates faithfulness and relevancy using an LLM.
    """
    if strategies is None:
        strategies = ["hybrid", "rewrite_hybrid"]

    print("\n" + "=" * 70)
    print(f"  🤖 RUNNING RAGAS EVALUATION (AI-powered metrics)")
    print(f"  Strategies: {', '.join(strategies)}")
    print(f"  Questions:  {num_questions}")
    print("=" * 70)

    # Use the existing dataset and slicer to stay within rate limits
    dataset = load_eval_dataset()[:num_questions]

    # Directly use session11's comparison logic but specifically for RAGAS
    try:
        results = run_ragas_comparison(
            strategies=strategies,
            eval_dataset=dataset,
            pipeline=pipeline,
        )
        # The result printing is handled inside run_ragas_comparison
        return results
    except Exception as e:
        print(f"\n[ERROR] RAGAS evaluation failed: {e}")
        print("  Ensure 'ragas' and 'datasets' are installed and GROQ_API_KEY is valid.")
        return None
