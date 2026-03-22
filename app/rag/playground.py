"""
rag/playground.py
-----------------
Session 7 RAG Foundations - Interactive Playground

Run from repo root:
    python -m app.rag.playground

Demonstrates:
  STEP 1 - Retrieval WITHOUT chunking (full-document baseline)
  STEP 2 - Retrieval WITH chunking (improved precision)
  STEP 3 - Additional sample queries across different document domains
"""

from app.rag.pipeline import RAGPipeline


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
    """Prints each retrieved result with rank, source, chunk_id, score, and preview."""
    for r in results:
        preview = r["text"][:200].replace("\n", " ")
        if len(r["text"]) > 200:
            preview += "..."
        print(f"\n  Rank {r['rank']}:")
        print(f"    Source:   {r['source']}")
        print(f"    Chunk ID: {r['chunk_id']}")
        print(f"    Score:    {r['score']:.4f}")
        print(f"    Preview:  {preview}")


def run_query(pipeline: RAGPipeline, query: str, show_answer: bool = True) -> None:
    """Runs a single query through the pipeline and prints results."""
    print(f"\n  Query: \"{query}\"")
    print("-" * 70)

    if show_answer:
        result = pipeline.query(query)
        print("\n  >> Retrieved Chunks:")
        print_retrieval_results(result["results"])
        print(f"\n  >> Grounded Answer:")
        print(f"     {result['answer']}")
    else:
        # Retrieval only (no LLM call)
        results = pipeline.inspect_retrieval(query)
        print("\n  >> Retrieved Chunks:")
        print_retrieval_results(results)


# ---------------------------------------------------------------------------
# Main demonstration flow
# ---------------------------------------------------------------------------

def main():
    print("\n")
    print("*" * 70)
    print("*  SESSION 7: RAG FOUNDATIONS - PLAYGROUND                           *")
    print("*" * 70)

    # ------------------------------------------------------------------
    # STEP 1: Without Chunking (Full-document baseline)
    # ------------------------------------------------------------------
    print_header("STEP 1: Retrieval WITHOUT Chunking (Full-Document Baseline)")
    print("\n  Building pipeline with use_chunking=False ...")

    pipeline_no_chunk = RAGPipeline(
        provider="groq",
        data_dir="data",
        use_chunking=False,
        top_k=3,
    )

    query_1 = "What is the maximum hotel reimbursement per night for domestic travel?"
    run_query(pipeline_no_chunk, query_1, show_answer=False)

    # ------------------------------------------------------------------
    # STEP 2: With Chunking (Improved retrieval)
    # ------------------------------------------------------------------
    print_header("STEP 2: Retrieval WITH Chunking (Improved Precision)")
    print("\n  Rebuilding pipeline with use_chunking=True ...")

    pipeline_chunked = RAGPipeline(
        provider="groq",
        data_dir="data",
        use_chunking=True,
        chunk_size=200,
        overlap=30,
        top_k=3,
    )

    # Same query as Step 1 to demonstrate improvement
    run_query(pipeline_chunked, query_1, show_answer=False)

    # ------------------------------------------------------------------
    # STEP 3: Additional Queries with LLM Grounded Answers
    # ------------------------------------------------------------------
    print_header("STEP 3: Additional Queries with Grounded Answers")

    sample_queries = [
        "What should I do on my first day at Acme Corp?",
        "What is the password policy at Acme Corp?",
        "How do I fix overheating on the SmartDevice X1?",
    ]

    for query in sample_queries:
        print("\n" + "-" * 70)
        run_query(pipeline_chunked, query, show_answer=True)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print_header("PLAYGROUND COMPLETE")
    print("""
  What we demonstrated:
    1. Full-document retrieval (weaker precision on specific questions)
    2. Chunked retrieval (better precision, relevant sections surfaced)
    3. Grounded answer generation using the existing LLM service

  Key takeaways:
    - Chunking matters for retrieval quality
    - Dense embeddings capture semantic similarity
    - FAISS enables fast vector search
    - Grounding the LLM with context reduces hallucination
    """)


if __name__ == "__main__":
    main()
