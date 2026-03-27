"""
rag/session9_playground.py
---------------------------
Session 9: Advanced RAG Frameworks — Master Interactive Demo

Run from repo root:
    python -m app.rag.session9_playground

What this demonstrates (5 steps):
    STEP 1 - Query rewriting: vague query vs rewritten query, better retrieval
    STEP 2 - Dense vs keyword: same query, different top results
    STEP 3 - LlamaIndex pipeline: full RAG answer with framework abstraction
    STEP 4 - LangChain pipeline: full RAG answer with chain composition
    STEP 5 - Full 4-way comparison on a conflict-resolution query

Teaching flow:
    This playground is designed to be run section-by-section in class,
    not all at once. Each STEP takes 2-5 minutes of discussion.
    Use Ctrl+C to pause between steps if needed.

Frameworks used (must be installed):
    pip install rank-bm25 llama-index-core llama-index-embeddings-huggingface
               llama-index-llms-groq langchain langchain-community langchain-groq

Conceptual notes shown in comments below:
    - Agentic RAG: where it fits, why it's not here
    - Graph RAG: where it fits, why it's not here
"""

from app.rag.session9_query_rewriter import rewrite_and_show
from app.rag.session9_comparison import (
    compare_dense_vs_keyword,
    compare_retrievers,
    _get_dense,
    _get_bm25,
    _get_llamaindex,
    _get_langchain,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_header(title: str, step: int) -> None:
    width = 70
    print("\n\n" + "=" * width)
    print(f"  STEP {step}: {title}")
    print("=" * width)


def ask_and_show(pipeline, query: str, label: str) -> None:
    """Calls .query() on a pipeline and prints the answer."""
    print(f"\n  [{label}] Query: \"{query}\"")
    print("-" * 70)
    answer = pipeline.query(query)
    print(f"\n  Answer:\n  {answer}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n")
    print("*" * 70)
    print("*  SESSION 9: ADVANCED RAG FRAMEWORKS — PLAYGROUND               *")
    print("*" * 70)
    print("""
  What we'll cover:
    STEP 1 — Query Rewriting (pre-retrieval transformation)
    STEP 2 — Dense vs Keyword retrieval (same query, different results)
    STEP 3 — LlamaIndex RAG (framework abstraction: index-centric)
    STEP 4 — LangChain RAG (framework abstraction: chain-centric)
    STEP 5 — Full 4-way comparison on a conflict-resolution query

  Note on Agentic RAG:
    An agent would decide WHICH retriever to use, WHEN to retrieve,
    and WHETHER to retrieve a second time. We don't build that here.
    See: LangChain Agents, LlamaIndex ReAct — explore post-session.

  Note on Graph RAG:
    Entities (companies, roles, rounds) would be nodes in a graph.
    Retrieval would traverse the graph to find connected facts.
    This requires entity extraction + graph construction — beyond scope.
    See: Microsoft's GraphRAG paper (2024) for a production example.
    """)

    # -----------------------------------------------------------------------
    # Pre-build all pipelines upfront so the demo flows without wait times
    # -----------------------------------------------------------------------
    print("\n  Building all pipelines (this takes ~1 min, do it once)...")
    dense_pipeline = _get_dense()
    bm25_retriever = _get_bm25()
    llamaindex_pipeline = _get_llamaindex()
    langchain_pipeline = _get_langchain()
    print("\n  All pipelines ready. Starting demo...\n")

    # -----------------------------------------------------------------------
    # STEP 1: Query Rewriting
    # -----------------------------------------------------------------------
    print_header("Query Rewriting — Pre-Retrieval Transformation", step=1)
    print("""
  Student mental model:
    Retrieval quality = f(query quality).
    A vague query gives a vague embedding → poor cosine matches.
    Rewriting makes the query specific before it hits the vector store.
    """)

    vague_queries = [
        "What about 2025?",
        "Tell me about rounds",
        "How does the hiring work?",
    ]

    for vq in vague_queries:
        rewrite_and_show(vq)

    print("\n  Try it: After rewriting, run the rewritten query through retrieval")
    print("  and compare the top results to the original query's results.")

    # -----------------------------------------------------------------------
    # STEP 2: Dense vs Keyword Retrieval
    # -----------------------------------------------------------------------
    print_header("Dense vs Keyword — Same Query, Different Results", step=2)
    print("""
  Dense retrieval wins on:   semantic meaning, paraphrased queries
  Keyword (BM25) wins on:   exact terms, rare company/product names

  Watch closely: which method finds the right source first?
    """)

    compare_dense_vs_keyword(
        "What data structures are commonly tested in Amazon interviews?"
    )
    print("  -- Above: 'data structures' is a common term - dense should win --\n")

    compare_dense_vs_keyword(
        "Oracle OCI cloud deployment exercise"
    )
    print("  -- Above: 'OCI' is a specific acronym - BM25 keyword may win --\n")

    # -----------------------------------------------------------------------
    # STEP 3: LlamaIndex Pipeline
    # -----------------------------------------------------------------------
    print_header("LlamaIndex RAG — Framework-Based (Index-Centric)", step=3)
    print("""
  LlamaIndex philosophy:
    Build an index ONCE, query it MANY times.
    The index is the unit — not the chain.

  What's abstracted (vs Session 8 manual pipeline):
    - Chunking (uses token-based splitter, not word-based)
    - Prompt building (internal default prompt)
    - Context assembly (handled inside the query engine)

  What's the same:
    - HuggingFace embedding model (all-MiniLM-L6-v2)
    - Groq LLM (same API key)
    """)

    ask_and_show(
        llamaindex_pipeline,
        "What system design topics should I prepare for Google interviews?",
        label="LlamaIndex",
    )

    ask_and_show(
        llamaindex_pipeline,
        "Has Microsoft changed how many rounds it conducts in 2025?",
        label="LlamaIndex",
    )

    # -----------------------------------------------------------------------
    # STEP 4: LangChain Pipeline
    # -----------------------------------------------------------------------
    print_header("LangChain RAG — Framework-Based (Chain-Centric)", step=4)
    print("""
  LangChain philosophy:
    Compose steps into CHAINS. Each step is explicit and swappable.
    The chain is the unit — not the index.

  What's abstracted (vs Session 8 manual pipeline):
    - Document loading (DirectoryLoader)
    - Chunking (RecursiveCharacterTextSplitter, character-based)
    - Chain execution (RetrievalQA.from_chain_type)

  What's different from LlamaIndex:
    - "stuff" chain type = all chunks concatenated into one context
    - Character-based splitting → different chunk boundaries
    - Score visibility: LangChain retriever doesn't expose scores by default
    """)

    ask_and_show(
        langchain_pipeline,
        "What behavioral questions are common at Adobe interviews?",
        label="LangChain",
    )

    ask_and_show(
        langchain_pipeline,
        "Which companies have added AI assessment rounds in 2025?",
        label="LangChain",
    )

    # -----------------------------------------------------------------------
    # STEP 5: Full 4-Way Comparison on a Conflict Query
    # -----------------------------------------------------------------------
    print_header("Full 4-Way Comparison — Conflict Resolution Query", step=5)
    print("""
  This query has a conflict:
    Public source (google.md):            says 5 coding rounds
    Private source (PDF, 2025 update):    says 3 rounds now

  Watch which retrievers surface the private PDF vs the public markdown.
  Dense retrieval (seeded with 2025 content) should rank the PDF higher.
  BM25 depends on keyword overlap.
    """)

    compare_retrievers(
        "How many coding rounds does Google conduct for SDE interviews in 2025?"
    )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("  PLAYGROUND COMPLETE")
    print("=" * 70)
    print("""
  What Session 9 introduced vs Session 8:

    Manual Pipeline (Session 8)     → Full control, verbose, educational
    LlamaIndex Pipeline (Session 9) → Same result, 30 lines vs 100+
    LangChain Pipeline (Session 9)  → Chain-first composition pattern
    BM25 Keyword (Session 9)        → No vectors, exact-match baseline
    Query Rewriting (Session 9)     → Pre-retrieval query improvement

  Design space expand:
    Dense retrieval        → [Session 7 + 8]  you built this
    Multi-source ingestion → [Session 8]       you built this
    Framework abstraction  → [Session 9]       today
    Keyword baseline       → [Session 9]       today
    Query rewriting        → [Session 9]       today
    Agentic RAG            → [Next module]     LLM decides what to do
    Graph RAG              → [Advanced]        entities as graph nodes

  Engineering lesson:
    Frameworks don't eliminate the need to understand fundamentals.
    They hide complexity — but bugs hide there too.
    Know what's underneath before you trust the abstraction.
    """)


if __name__ == "__main__":
    main()
