"""
session10/query_rewriter.py
----------------------------
LLM-assisted query rewriting for pre-retrieval optimization.

Purpose:
    Vague or under-specified queries produce poor retrieval results
    regardless of which retriever you use. A query rewriter transforms
    the user's raw question into a retrieval-friendly form BEFORE
    searching.

Design constraint — Intent Preservation:
    The rewriter MUST NOT inject assumptions, facts, or details that
    are not implied by the original query. It should only:
    - Make implicit intent explicit
    - Add retrieval-friendly keywords that are clearly implied
    - Restructure for clarity

    Example (correct):
        "amazon interview" → "Amazon SDE interview process and rounds"
    Example (WRONG — injects assumptions):
        "amazon interview" → "Amazon SDE interview with 5 rounds including
         system design and behavioral in Seattle office"

    The rewriter returns ONLY the rewritten query string, never an answer.

Fallback:
    If the LLM call fails, the original query is returned unchanged.
    The pipeline never breaks due to a rewriting failure.
"""

from app.services.llm_service import ask_llm


# ---- Prompt template ----
# Carefully worded to prevent the LLM from injecting outside knowledge.

REWRITE_INSTRUCTION = """\
You are a search query optimizer for a technical knowledge base about
company interview processes.

Your ONLY job: rewrite the user's query into a clearer, more specific
search query that will retrieve better results.

Strict rules:
1. Do NOT answer the question
2. Do NOT add facts, numbers, or details not implied by the original query
3. Do NOT assume specific roles, locations, or years unless stated
4. Make implicit intent explicit (e.g., "rounds" → "interview rounds")
5. Add retrieval-friendly keywords ONLY if clearly implied
6. Keep the rewritten query to 1-2 sentences
7. Return ONLY the rewritten query — no explanation, no preamble

User query: {query}
Rewritten query:"""


def rewrite_query(query: str, provider: str = "groq") -> str:
    """
    Rewrites a user query into a retrieval-friendly form using the LLM.

    The rewriting preserves the user's intent strictly — no assumptions
    or external facts are injected.

    Args:
        query:    the original user question.
        provider: LLM provider to use (default: groq).

    Returns:
        A rewritten query string, or the original query if the LLM
        call fails.
    """
    prompt = REWRITE_INSTRUCTION.format(query=query)

    try:
        rewritten = ask_llm(provider, prompt).strip()

        # Strip surrounding quotes if the LLM wraps the output
        if rewritten.startswith('"') and rewritten.endswith('"'):
            rewritten = rewritten[1:-1].strip()

        # Sanity: don't return empty or identical rewrites
        if not rewritten or rewritten.lower() == query.lower():
            return query

        return rewritten

    except Exception as e:
        print(f"[QueryRewriter] LLM call failed ({e}), using original query")
        return query


def rewrite_and_show(query: str, provider: str = "groq") -> str:
    """
    Rewrites the query and prints both versions side by side.
    Convenience wrapper for classroom demos.

    Returns:
        The rewritten query string.
    """
    print(f"\n[QueryRewriter] Original : \"{query}\"")
    rewritten = rewrite_query(query, provider=provider)
    changed = "CHANGED" if rewritten != query else "UNCHANGED"
    print(f"[QueryRewriter] Rewritten: \"{rewritten}\" [{changed}]")
    return rewritten
