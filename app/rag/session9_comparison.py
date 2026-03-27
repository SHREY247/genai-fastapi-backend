"""
rag/session9_comparison.py
---------------------------
Session 9: Side-by-Side Retrieval Comparison

Runs the same query through four retrieval approaches and prints results
side-by-side so students can compare them directly.

Four retrievers:
    1. Manual Dense  — Session 8's InterviewRAGPipeline (FAISS + HF embeddings)
    2. BM25 Keyword  — Session 9's BM25Retriever (no vectors)
    3. LlamaIndex    — Session 9's LlamaIndexPipeline (retrieve_only mode)
    4. LangChain     — Session 9's LangChainPipeline (retrieve_only mode)

This is retrieval-only (no LLM calls) for speed.
Use the individual pipeline .query() methods for full RAG answers.

Teaching intent:
    - Show that the same query can produce DIFFERENT top results
    - BM25 wins on exact keyword matches (e.g., "Oracle OCI")
    - Dense wins on semantic/paraphrased queries
    - LlamaIndex and LangChain abstract chunking — their results differ
      slightly from Session 8 due to different chunk boundaries
"""

from typing import List, Dict, Optional

from app.rag.interview_pipeline import InterviewRAGPipeline
from app.rag.session9_keyword_retriever import BM25Retriever
from app.rag.session9_llamaindex_pipeline import LlamaIndexPipeline
from app.rag.session9_langchain_pipeline import LangChainPipeline


DATA_DIR = "data/interview_prep"

# Module-level singletons — built once, reused across comparisons
# (each pipeline takes 10-30 seconds to build due to embedding model loading)
_dense_pipeline: Optional[InterviewRAGPipeline] = None
_bm25_retriever: Optional[BM25Retriever] = None
_llamaindex_pipeline: Optional[LlamaIndexPipeline] = None
_langchain_pipeline: Optional[LangChainPipeline] = None


def _get_dense():
    global _dense_pipeline
    if _dense_pipeline is None:
        _dense_pipeline = InterviewRAGPipeline(data_dir=DATA_DIR, top_k=5)
    return _dense_pipeline


def _get_bm25():
    global _bm25_retriever
    if _bm25_retriever is None:
        _bm25_retriever = BM25Retriever(data_dir=DATA_DIR)
    return _bm25_retriever


def _get_llamaindex():
    global _llamaindex_pipeline
    if _llamaindex_pipeline is None:
        _llamaindex_pipeline = LlamaIndexPipeline(data_dir=DATA_DIR, top_k=5)
    return _llamaindex_pipeline


def _get_langchain():
    global _langchain_pipeline
    if _langchain_pipeline is None:
        _langchain_pipeline = LangChainPipeline(data_dir=DATA_DIR, top_k=5)
    return _langchain_pipeline


def _print_results_table(label: str, results: List[Dict]) -> None:
    """Prints a compact ranked result table for one retriever."""
    width = 68
    print(f"\n  ┌─ {label} " + "─" * (width - len(label) - 4) + "┐")
    for r in results[:3]:   # Show top 3 for readability
        score_str = f"{r['score']:.4f}" if r.get("score") is not None else "  N/A "
        source = r.get("source", "unknown")[:28]
        company = r.get("company", r.get("source", "?"))[:12]
        text_preview = r["text"][:60].replace("\n", " ")
        print(f"  │ Rank {r['rank']:1d} [{score_str}] {company:<12} {source:<28} │")
        print(f"  │   \"{text_preview}...\" │")
    print("  └" + "─" * width + "┘")


def compare_retrievers(query: str, top_k: int = 5) -> None:
    """
    Runs `query` through all four retrievers and prints a comparison.

    Args:
        query:  the question to test
        top_k:  results to retrieve per method (default 5, prints top 3)
    """
    print("\n" + "=" * 70)
    print(f"  RETRIEVAL COMPARISON")
    print(f"  Query: \"{query}\"")
    print("=" * 70)

    # 1. Manual Dense (Session 8)
    print("\n  Building / using Manual Dense pipeline (Session 8)...")
    dense_results = _get_dense().inspect_retrieval(query, top_k=top_k)
    _print_results_table("1. Manual Dense (FAISS + HF Embeddings)", dense_results)

    # 2. BM25 Keyword (Session 9)
    print("\n  Running BM25 keyword retrieval...")
    bm25_results = _get_bm25().retrieve(query, top_k=top_k)
    _print_results_table("2. BM25 Keyword (No Vectors)", bm25_results)

    # 3. LlamaIndex (Session 9)
    print("\n  Running LlamaIndex retrieval...")
    llamaindex_results = _get_llamaindex().retrieve_only(query)
    _print_results_table("3. LlamaIndex (Framework Dense)", llamaindex_results)

    # 4. LangChain (Session 9)
    print("\n  Running LangChain retrieval...")
    langchain_results = _get_langchain().retrieve_only(query)
    _print_results_table("4. LangChain (Framework Dense)", langchain_results)

    print("\n" + "=" * 70)
    print("  Tip: Compare which sources appear across methods.")
    print("  BM25 ranks by keyword frequency; Dense by semantic similarity.")
    print("=" * 70 + "\n")


def compare_dense_vs_keyword(query: str, top_k: int = 5) -> None:
    """
    Lightweight comparison: just Dense vs BM25.
    Faster for mid-class demos (no need to build LlamaIndex/LangChain).
    """
    print("\n" + "=" * 70)
    print(f"  DENSE vs KEYWORD")
    print(f"  Query: \"{query}\"")
    print("=" * 70)

    dense_results = _get_dense().inspect_retrieval(query, top_k=top_k)
    _print_results_table("Manual Dense (FAISS)", dense_results)

    bm25_results = _get_bm25().retrieve(query, top_k=top_k)
    _print_results_table("BM25 Keyword", bm25_results)

    print("\n" + "=" * 70 + "\n")
