"""
session10/failure_cases.py
---------------------------
Pre-defined test queries that expose retrieval weaknesses.

Each test case includes:
    - query:            the user's raw question
    - expected_behavior: what a good retrieval system should return
    - failure_type:     which retrieval signal fails on this query
    - best_strategy:    which strategy handles this best and why

Purpose:
    These are designed for LIVE DEMOS. An instructor can run each query
    across all strategies and show students exactly when and why each
    retriever wins or fails.

Coverage:
    1. Keyword-heavy queries    → BM25 wins (exact term match)
    2. Semantic queries         → Vector wins (meaning over keywords)
    3. Ambiguous queries        → Hybrid + rewrite wins (multi-signal)
    4. Acronym queries          → BM25 wins (exact token matching)
    5. Negation/contrast        → Vector wins (semantic understanding)
"""

TEST_QUERIES = [
    # ---- Category 1: Keyword-heavy (BM25 advantage) ----
    {
        "query": "Oracle OCI cloud interview questions",
        "expected_behavior": (
            "Should retrieve chunks containing exact mentions of "
            "'Oracle', 'OCI', and 'cloud' from Oracle-related documents."
        ),
        "failure_type": "vector",
        "best_strategy": "bm25",
        "why": (
            "Vector embeddings may not distinguish 'Oracle OCI' from "
            "other cloud platforms. BM25 performs exact token matching "
            "on 'oracle' and 'oci', directly finding relevant chunks."
        ),
    },
    {
        "query": "Amazon leadership principles LP",
        "expected_behavior": (
            "Should retrieve chunks about Amazon's Leadership Principles "
            "and how they are used in interviews."
        ),
        "failure_type": "vector",
        "best_strategy": "bm25",
        "why": (
            "'LP' is an abbreviation that embeds poorly in dense vectors. "
            "BM25 matches the token 'lp' directly against documents that "
            "use this abbreviation."
        ),
    },

    # ---- Category 2: Semantic queries (Vector advantage) ----
    {
        "query": "How should I prepare for a coding assessment?",
        "expected_behavior": (
            "Should retrieve chunks about interview coding rounds, "
            "technical assessments, and preparation tips — even if "
            "the exact phrase 'coding assessment' doesn't appear."
        ),
        "failure_type": "bm25",
        "best_strategy": "vector",
        "why": (
            "The user's phrasing may not match document vocabulary. "
            "Documents might say 'technical round' or 'DSA test' instead "
            "of 'coding assessment'. Vector retrieval captures semantic "
            "similarity across different phrasings."
        ),
    },
    {
        "query": "What's the difference between the initial and final rounds?",
        "expected_behavior": (
            "Should retrieve chunks about interview round progression, "
            "phone screens vs. onsite, and evaluation criteria at "
            "different stages."
        ),
        "failure_type": "bm25",
        "best_strategy": "vector",
        "why": (
            "This is a contrast/comparison query. BM25 treats 'initial' "
            "and 'final' as independent tokens and can't understand the "
            "comparative intent. Vector retrieval encodes the full "
            "sentence meaning."
        ),
    },

    # ---- Category 3: Ambiguous queries (Hybrid + Rewrite advantage) ----
    {
        "query": "Tell me about the rounds",
        "expected_behavior": (
            "Should retrieve chunks about interview rounds across all "
            "companies. A rewrite should clarify this to something like "
            "'interview round structure and process across companies'."
        ),
        "failure_type": "both",
        "best_strategy": "hybrid_rewrite",
        "why": (
            "Too vague for either strategy alone. 'Rounds' embeds poorly "
            "as a single generic term, and BM25 matches 'rounds' in too "
            "many irrelevant contexts. Query rewriting expands the intent, "
            "and hybrid fusion combines the improved signals."
        ),
    },
    {
        "query": "What about 2025?",
        "expected_behavior": (
            "Should retrieve chunks about 2025 hiring updates or process "
            "changes. A rewrite should clarify the intent."
        ),
        "failure_type": "both",
        "best_strategy": "hybrid_rewrite",
        "why": (
            "'What about 2025' is extremely vague. Neither retriever can "
            "do well alone. Rewriting to something like '2025 company "
            "interview process changes and updates' gives both retrievers "
            "better tokens and semantics to work with."
        ),
    },

    # ---- Category 4: Acronym queries (BM25 advantage) ----
    {
        "query": "SDE interview at Amazon",
        "expected_behavior": (
            "Should retrieve chunks specifically about Amazon's Software "
            "Development Engineer interview process."
        ),
        "failure_type": "vector",
        "best_strategy": "bm25",
        "why": (
            "'SDE' is a domain-specific acronym. General-purpose embedding "
            "models may not encode it well, but BM25 matches the exact "
            "token 'sde' against documents."
        ),
    },

    # ---- Category 5: Well-formed semantic query (Vector advantage) ----
    {
        "query": (
            "How do tech companies evaluate problem-solving skills "
            "during their hiring process?"
        ),
        "expected_behavior": (
            "Should retrieve chunks about evaluation criteria, problem-solving "
            "rounds, and how interviewers assess candidates."
        ),
        "failure_type": "bm25",
        "best_strategy": "vector",
        "why": (
            "This is a well-formed, semantically rich query. Documents may "
            "use different words ('assess', 'test', 'judge') for the same "
            "concept. Vector retrieval captures the meaning regardless of "
            "exact keyword overlap."
        ),
    },
]


def get_queries_by_strategy(strategy: str):
    """
    Returns test queries where the given strategy is expected to win.

    Args:
        strategy: one of "vector", "bm25", "hybrid_rewrite"

    Returns:
        List of matching test case dicts.
    """
    return [q for q in TEST_QUERIES if q["best_strategy"] == strategy]


def print_failure_cases():
    """
    Prints all test cases in a readable format for review.
    """
    for i, tc in enumerate(TEST_QUERIES, start=1):
        print(f"\n{'='*60}")
        print(f"Test Case {i}: {tc['failure_type']} weakness")
        print(f"{'='*60}")
        print(f"  Query:         {tc['query']}")
        print(f"  Best Strategy: {tc['best_strategy']}")
        print(f"  Why:           {tc['why']}")
        print(f"  Expected:      {tc['expected_behavior']}")
