"""
rag/interview_playground.py
---------------------------
Session 8: Interview Prep RAG - Interactive Playground

Run from repo root:
    python -m app.rag.interview_playground

Demonstrates:
  STEP 1 - Multi-source ingestion (markdown + PDF)
  STEP 2 - Retrieval with metadata (source type, company)
  STEP 3 - Source-aware answers with conflict resolution
  STEP 4 - Cross-company and private-source queries
"""

from app.rag.interview_pipeline import InterviewRAGPipeline


# ---------------------------------------------------------------------------
# Helpers for pretty-printing
# ---------------------------------------------------------------------------

def print_header(title: str) -> None:
    """Prints a visible section header."""
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_retrieval_results(results: list) -> None:
    """Prints each retrieved result with metadata and preview."""
    for r in results:
        preview = r["text"][:200].replace("\n", " ")
        if len(r["text"]) > 200:
            preview += "..."

        source_type = r.get("source_type", "unknown")
        company = r.get("company", "Unknown")
        label = "PRIVATE" if "private" in source_type.lower() else "PUBLIC"

        print(f"\n  Rank {r['rank']}:")
        print(f"    [{label}] Company: {company}")
        print(f"    Source:   {r['source']}")
        if r.get("page"):
            print(f"    Page:     {r['page']}")
        print(f"    Chunk ID: {r['chunk_id']}")
        print(f"    Score:    {r['score']:.4f}")
        print(f"    Preview:  {preview}")


def print_sources_used(sources: list) -> None:
    """Prints the unique sources that contributed to the answer."""
    print("\n  Sources Used:")
    for s in sources:
        label = "PRIVATE" if "private" in s.get("source_type", "") else "PUBLIC"
        print(f"    - [{label}] {s['company']} ({s['source']})")


def run_query(
    pipeline: InterviewRAGPipeline,
    query: str,
    show_answer: bool = True,
    description: str = "",
) -> None:
    """Runs a single query through the pipeline and prints results."""
    if description:
        print(f"\n  Test: {description}")
    print(f"  Query: \"{query}\"")
    print("-" * 70)

    if show_answer:
        result = pipeline.query(query)
        print("\n  >> Retrieved Chunks:")
        print_retrieval_results(result["results"])
        print_sources_used(result["sources_used"])
        print(f"\n  >> Grounded Answer:")
        print(f"     {result['answer']}")
    else:
        results = pipeline.inspect_retrieval(query)
        print("\n  >> Retrieved Chunks:")
        print_retrieval_results(results)


# ---------------------------------------------------------------------------
# Main demonstration flow
# ---------------------------------------------------------------------------

def main():
    print("\n")
    print("*" * 70)
    print("*  SESSION 8: INTERVIEW PREP RAG - PLAYGROUND                        *")
    print("*" * 70)

    # ------------------------------------------------------------------
    # STEP 1: Build the pipeline (multi-source ingestion)
    # ------------------------------------------------------------------
    print_header("STEP 1: Multi-Source Ingestion")
    print("\n  Building pipeline from data/interview_prep/ ...")
    print("  (Contains: 5 company .md files + 1 private .pdf)\n")

    pipeline = InterviewRAGPipeline(
        provider="groq",
        data_dir="data/interview_prep",
        chunk_size=200,
        overlap=30,
        top_k=5,
    )

    # Show ingestion summary
    summary = pipeline.get_ingestion_summary()
    print("\n  Ingestion Summary:")
    print(f"    Documents loaded:  {summary['total_documents']}")
    print(f"    Chunks created:    {summary['total_chunks']}")
    print(f"    Companies:         {', '.join(summary['companies'])}")
    print(f"    Source types:       {', '.join(summary['source_types'])}")

    # ------------------------------------------------------------------
    # STEP 2: Public source retrieval (no LLM, retrieval only)
    # ------------------------------------------------------------------
    print_header("STEP 2: Public Source Retrieval (Retrieval Only)")

    run_query(
        pipeline,
        "What data structures are commonly asked in Amazon interviews?",
        show_answer=False,
        description="Should retrieve from amazon.md (public)",
    )

    # ------------------------------------------------------------------
    # STEP 3: Private source retrieval (with LLM)
    # ------------------------------------------------------------------
    print_header("STEP 3: Private Source Query (With LLM)")

    run_query(
        pipeline,
        "Has Microsoft changed its interview format for 2025?",
        show_answer=True,
        description="Should retrieve from private PDF (2025 updates)",
    )

    # ------------------------------------------------------------------
    # STEP 4: Conflict resolution queries
    # ------------------------------------------------------------------
    print_header("STEP 4: Conflict Resolution - Public vs Private")

    run_query(
        pipeline,
        "How many coding rounds does Google conduct for SDE interviews?",
        show_answer=True,
        description="Public says 5 rounds, Private 2025 update says 3",
    )

    # ------------------------------------------------------------------
    # STEP 5: Cross-company query
    # ------------------------------------------------------------------
    print_header("STEP 5: Cross-Company Query")

    run_query(
        pipeline,
        "Which companies have added AI-related assessments to their interview process in 2025?",
        show_answer=True,
        description="Should pull from multiple companies in private PDF",
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print_header("PLAYGROUND COMPLETE")
    print("""
  What we demonstrated:
    1. Multi-source ingestion: markdown + PDF loaded into one pipeline
    2. Unified schema: all sources normalized with rich metadata
    3. Metadata in retrieval: source type, company, page preserved
    4. Source-aware answers: LLM cites where information came from
    5. Conflict resolution: 2025 private updates override public data

  What's new compared to Session 7:
    - Heterogeneous source loading (not just .txt)
    - Normalization layer with unified Document schema
    - Metadata-aware chunking that preserves source information
    - Source-aware prompt engineering for citation
    - Real-world data preparation from a GitHub repo

  Engineering lessons:
    - Normalize early, benefit everywhere downstream
    - Metadata is free at ingestion time, expensive to add later
    - The prompt is part of your system design, not an afterthought
    - Private/internal data is where RAG creates real business value
    """)


if __name__ == "__main__":
    main()
