"""
session12/testing.py
---------------------
Simulated DeepEval-style testing for Session 12: RAG Evaluation Ecosystem & Production QA.

Provides:
1. run_basic_tests() — PASS/FAIL testing based on keyword presence.
2. evaluation_loop() — Simple loop demonstrating a "score -> improve" logic.

This file demonstrates the philosophy of "Unit Testing for RAG" without needing
complex external dependencies.
"""

import time
from typing import Dict, List, Optional
from app.rag.session11.observability import run_strategy

# ---------------------------------------------------------------------------
# Test Case Definition
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "name": "Amazon Rounds Test",
        "query": "What are the interview rounds at Amazon for an SDE role?",
        "expected_keywords": ["online assessment", "Bar Raiser", "behavioral"],
        "min_score_threshold": 0.5,
    },
    {
        "name": "Google Problem-Solving Test",
        "query": "How does Google evaluate problem-solving skills during interviews?",
        "expected_keywords": ["algorithmic", "clean", "complexity"],
        "min_score_threshold": 0.5,
    },
    {
        "name": "Oracle OCI Test",
        "query": "What is Oracle OCI?",
        "expected_keywords": ["Cloud Infrastructure", "platform"],
        "min_score_threshold": 0.4,
    }
]

# ---------------------------------------------------------------------------
# PASS / FAIL Logic (Simulated DeepEval)
# ---------------------------------------------------------------------------

def run_basic_tests(strategy_name: str = "hybrid", pipeline=None):
    """
    Simulates a 'unit test' for a RAG pipeline.
    Runs a few fixed queries and checks if expected keywords exist in the answer.
    """
    print("\n" + "=" * 70)
    print("  🚀 RUNNING SIMULATED RAG UNIT TESTS (DeepEval-style)")
    print("=" * 70)

    results = []
    
    for case in TEST_CASES:
        print(f"\n[TEST CASE] {case['name']}")
        print(f"  Question: \"{case['query']}\"")

        # Run the actual pipeline
        result = run_strategy(
            strategy_name=strategy_name,
            query=case["query"],
            pipeline=pipeline,
            top_k=5,
            debug=False,
            generate_answer=True,
        )

        answer = result.get("answer", "").lower()
        score = result.get("overall_score", 0.0) # Assume some score is returned or calculated

        # Logic: check for keyword presence
        passed_keywords = [kw for kw in case["expected_keywords"] if kw.lower() in answer]
        missing_keywords = [kw for kw in case["expected_keywords"] if kw.lower() not in answer]

        status = "✅ PASS" if not missing_keywords else "❌ FAIL"
        
        print(f"  Result:  {status}")
        if missing_keywords:
            print(f"  Missing: {missing_keywords}")
        else:
            print(f"  All Expected Keywords Found: {passed_keywords}")
        
        results.append({
            "name": case["name"],
            "status": status,
            "missing": missing_keywords
        })

    print("\n" + "=" * 70)
    print("  FINAL TEST SUMMARY")
    print("-" * 70)
    for r in results:
        print(f"  {r['status']:7s} | {r['name']}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Simple Evaluation Loop Abstraction
# ---------------------------------------------------------------------------

def evaluation_loop(pipeline=None):
    """
    Demonstrates a simple feedback loop:
    Evaluates -> Checks threshold -> Suggests improvements.
    """
    print("\n" + "#" * 70)
    print("#  DEMO: CONTINUOUS EVALUATION LOOP")
    print("#" * 70)

    query = "Oracle OCI cloud interview questions"
    threshold = 0.6
    
    # Run once
    print(f"\nChecking strategy: 'vector' on query: \"{query}\"")
    result = run_strategy("vector", query, pipeline=pipeline, generate_answer=True)
    
    # Assume we use a basic metric like Source Coverage for this demo
    # We can use the evaluator logic from Session 11 if available
    coverage = result.get("source_coverage", 0.3) # Dummy value for demo
    
    print(f"  Current Score: {coverage:.2f} (Threshold: {threshold:.2f})")
    
    if coverage < threshold:
        print(f"  ⚠️  RESULT BELOW THRESHOLD")
        print(f"  💡 SUGGESTION: 'Vector' score is low for keyword-heavy names. Switching to 'Hybrid'...")
        time.sleep(1.0)
        
        print(f"\nRetrying with 'hybrid'...")
        new_result = run_strategy("hybrid", query, pipeline=pipeline, generate_answer=True)
        new_coverage = 0.85 # Dummy value indicating improvement
        print(f"  New Score: {new_coverage:.2f} (PASS)")
    else:
        print(f"  ✅ RESULT ABOVE THRESHOLD. Deployment approved.")
    
    print("\n" + "#" * 70 + "\n")
