"""
rag/session9_comparison.py
---------------------------
Session 9: Three-Pipeline Answer Comparison

Runs the same question through three RAG approaches and prints answers
side-by-side so students can compare them directly.

Three pipelines:
    1. Manual (Session 8)  — InterviewRAGPipeline, full metadata, source labels
    2. LlamaIndex          — VectorStoreIndex, compact framework, index-centric
    3. LangChain           — RetrievalQA chain, chain-centric composition

This is the teaching centerpiece of Session 9:
    - Are the answers meaningfully different?
    - Which shows you WHERE the answer came from?
    - Which would you trust in production and why?

Usage:
    from app.rag.session9_comparison import compare_answers
    compare_answers(
        question="Has Microsoft changed its interview format in 2025?",
        manual_pipeline=manual,
        llamaindex_pipeline=llamaindex,
        langchain_pipeline=langchain,
    )
"""

from typing import Optional


def compare_answers(
    question: str,
    manual_pipeline,
    llamaindex_pipeline,
    langchain_pipeline,
) -> None:
    """
    Runs the same question through all three pipelines and prints a
    side-by-side comparison of answers and retrieved sources.

    Args:
        question:             The user's question string
        manual_pipeline:      InterviewRAGPipeline instance (Session 8)
        llamaindex_pipeline:  LlamaIndexPipeline instance
        langchain_pipeline:   LangChainPipeline instance
    """
    width = 70
    print("\n" + "=" * width)
    print(f"  COMPARISON")
    print(f"  Q: \"{question}\"")
    print("=" * width)

    # ── 1. Manual Pipeline (Session 8) ──────────────────────────────────────
    print("\n  ── 1. Session 8 Manual Pipeline ──")
    try:
        result = manual_pipeline.query(question)
        answer = result.get("answer", "[no answer]")
        sources_used = result.get("sources_used", [])

        print(f"  Answer: {answer.strip()[:400]}")
        if sources_used:
            print("  Sources:")
            for s in sources_used:
                label = "PRIVATE" if "private" in s.get("source_type", "") else "PUBLIC"
                page = f" p.{s['page']}" if s.get("page") else ""
                print(f"    [{label}] {s.get('company','?')} — {s.get('source','?')}{page}")
    except Exception as e:
        print(f"  [ERROR] {e}")

    # ── 2. LlamaIndex ────────────────────────────────────────────────────────
    print("\n  ── 2. LlamaIndex ──")
    try:
        answer_li = llamaindex_pipeline.query(question)
        print(f"  Answer: {str(answer_li).strip()[:400]}")

        retrieved = llamaindex_pipeline.retrieve_only(question)
        if retrieved:
            print("  Sources (filename only — no metadata labels):")
            for r in retrieved[:3]:
                print(f"    Rank {r['rank']}: {r['source']} (score: {r['score']:.4f})")
    except Exception as e:
        print(f"  [ERROR] {e}")

    # ── 3. LangChain ─────────────────────────────────────────────────────────
    print("\n  ── 3. LangChain ──")
    try:
        result_lc = langchain_pipeline.query(question)
        answer_lc = result_lc.get("answer", "[no answer]")
        sources_lc = result_lc.get("sources", [])

        print(f"  Answer: {str(answer_lc).strip()[:400]}")
        if sources_lc:
            print(f"  Sources (found {len(sources_lc)}):")
            for s in sources_lc:
                print(f"    - {s}")
    except Exception as e:
        print(f"  [ERROR] {e}")

    # ── Teaching observation ─────────────────────────────────────────────────
    print(f"\n  {'-' * (width - 2)}")
    print("  Observe:")
    print("    Manual → shows [PUBLIC]/[PRIVATE], company, page number")
    print("    LlamaIndex → shows filename and score, no custom labels")
    print("    LangChain  → shows filenames, no scores by default")
    print("  The more metadata you need, the more the manual pipeline pays off.")
    print("=" * width + "\n")
