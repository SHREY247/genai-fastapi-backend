"""
rag/session9_query_rewriter.py
-------------------------------
Session 9: Query Rewriting (Pre-Retrieval Transformation)

Why this exists:
    Retrieval quality is only as good as the query.
    Vague or under-specified queries like "What about 2025?" or
    "Tell me about rounds" produce poor embedding representations
    and miss relevant chunks.

    A query rewriter calls the LLM *before* retrieval to expand or
    clarify the query. This is a lightweight but high-value technique.

Teaching point:
    - Show a vague query → retrieval results (mediocre)
    - Rewrite the query → retrieval results (much better)
    - Helps students understand: retrieval is a function of query quality

This is NOT a full Agentic RAG setup.
    An agent would *decide* whether to rewrite, and might loop.
    Here we always rewrite once — simple and demonstrable.

Agentic RAG note (for instructors):
    In a true Agentic RAG system (e.g. LangChain agents, LlamaIndex
    ReAct agents), the LLM decides WHICH tool to call, WHEN to call it,
    and whether to retrieve again after an unsatisfactory first result.
    That's Session 10 territory. Here, we just do the simplest useful
    thing: rewrite once, retrieve once.
"""

from app.services.llm_service import ask_llm

REWRITE_INSTRUCTION = """\
You are a search query optimizer for an Interview Prep knowledge base.

Your job: given a user's raw question, rewrite it into a clear, specific,
keyword-rich search query that will find the most relevant content.

Rules:
- Make the query more specific and concrete
- Include company names, years, or topic names if they can be inferred
- Keep the rewritten query to 1-2 sentences
- Do NOT answer the question — only rewrite the query
- Return ONLY the rewritten query, nothing else

User question: {query}
Rewritten query:"""


def rewrite_query(query: str, provider: str = "groq") -> str:
    """
    Rewrites a vague user query into a more retrieval-friendly form
    using the LLM.

    Args:
        query:    the original user question
        provider: LLM provider to use (default: groq)

    Returns:
        A rewritten query string. Falls back to the original if the
        LLM call fails (so the pipeline never breaks).

    Example:
        Input:  "What about the 2025 changes?"
        Output: "What interview format and process changes did companies
                 make in 2025 according to their hiring updates?"
    """
    prompt = REWRITE_INSTRUCTION.format(query=query)

    try:
        rewritten = ask_llm(provider, prompt).strip()
        # Sanity check: don't return an empty or identical rewrite
        if not rewritten or rewritten.lower() == query.lower():
            return query
        return rewritten
    except Exception as e:
        print(f"[QueryRewriter] Warning: LLM call failed ({e}), using original query")
        return query


def rewrite_and_show(query: str, provider: str = "groq") -> str:
    """
    Rewrites the query and prints both versions.
    Convenience wrapper for classroom demos.

    Returns the rewritten query.
    """
    print(f"\n[QueryRewriter] Original : \"{query}\"")
    rewritten = rewrite_query(query, provider=provider)
    print(f"[QueryRewriter] Rewritten: \"{rewritten}\"")
    return rewritten
