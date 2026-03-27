"""
rag/session9_playground.py
---------------------------
Session 9: Advanced RAG Frameworks — Framework Comparison Demo

Run from repo root:
    python -m app.rag.session9_playground

What this demonstrates (4 steps):

    STEP 1 — Session 8 manual pipeline recap
              Remind students what we built by hand: 8 modules, full control

    STEP 2 — LlamaIndex pipeline
              Same data, 30 lines of setup, index-centric abstraction

    STEP 3 — LangChain pipeline
              Same data, chain-centric abstraction, different composition model

    STEP 4 — Framework comparison
              Code size, abstraction level, control, source visibility

Teaching intent:
    Session 9 is about seeing the RAG design space expand.
    After building everything manually in Sessions 7–8, students now see how
    frameworks compress that work — and what they give up in exchange.

    This session does NOT cover:
        - BM25 / keyword retrieval     → Session 10
        - Query rewriting              → Session 10
        - Agentic RAG                  → Session 10
        - Graph RAG                    → Advanced module

Dependencies:
    pip install llama-index-core llama-index-embeddings-huggingface
               llama-index-llms-groq langchain langchain-community langchain-groq
"""

from app.rag.session9_comparison import compare_answers
from app.rag.session9_llamaindex_pipeline import LlamaIndexPipeline
from app.rag.session9_langchain_pipeline import LangChainPipeline
from app.rag.interview_pipeline import InterviewRAGPipeline

DATA_DIR = "data/interview_prep"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_header(step: int, title: str) -> None:
    width = 70
    print("\n\n" + "=" * width)
    print(f"  STEP {step}: {title}")
    print("=" * width)


def print_answer(label: str, answer: str) -> None:
    print(f"\n  [{label}]")
    print(f"  {answer.strip()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n")
    print("*" * 70)
    print("*  SESSION 9: ADVANCED RAG FRAMEWORKS                             *")
    print("*  Manual Pipeline vs LlamaIndex vs LangChain                     *")
    print("*" * 70)

    # -----------------------------------------------------------------------
    # Pre-build all three pipelines once
    # -----------------------------------------------------------------------
    print("\n  Building all three pipelines (this takes ~1 min)...")
    manual = InterviewRAGPipeline(
        provider="groq",
        data_dir=DATA_DIR,
        chunk_size=200,
        overlap=30,
        top_k=5,
    )
    llamaindex = LlamaIndexPipeline(data_dir=DATA_DIR, top_k=5)
    langchain  = LangChainPipeline(data_dir=DATA_DIR, top_k=5)
    print("\n  All three pipelines ready. Starting demo...\n")

    # -----------------------------------------------------------------------
    # STEP 1: Manual Pipeline — Session 8 Recap
    # -----------------------------------------------------------------------
    print_header(1, "Session 8 Manual Pipeline — What we built by hand")
    print("""
  Session 8 pipeline (8 modules, explicit control):
    loaders.py           → load .md and .pdf
    normalization.py     → unified Document schema
    chunking.py          → word-based, metadata-aware
    embedding.py         → HuggingFace all-MiniLM-L6-v2
    vector_store.py      → FAISS IndexFlatIP
    retriever.py         → top-k cosine search
    prompt_builder.py    → source-aware, conflict-resolution prompt
    interview_pipeline.py→ InterviewRAGPipeline

  Every step visible. Every step controllable.
  The cost: ~600 lines of code to get here.
    """)

    q1 = "What data structures are commonly tested in Amazon interviews?"
    print(f"  Demo query: \"{q1}\"")
    result = manual.query(q1)
    print_answer("Session 8 Manual", result["answer"])

    print("\n  Sources used:")
    for s in result.get("sources_used", []):
        label = "PRIVATE" if "private" in s.get("source_type", "") else "PUBLIC"
        print(f"    [{label}] {s['company']} — {s['source']}")

    # -----------------------------------------------------------------------
    # STEP 2: LlamaIndex Pipeline
    # -----------------------------------------------------------------------
    print_header(2, "LlamaIndex — Index-Centric Framework")
    print("""
  LlamaIndex philosophy:
    Build an index. Query the index.
    The index is the central abstraction — not the chain, not the prompt.

  What it compresses vs Session 8:
    SimpleDirectoryReader   → replaces loaders.py + normalization.py
    VectorStoreIndex        → replaces chunking + embedding + FAISS
    query_engine.query()    → replaces retriever + prompt_builder + LLM call

  What you lose:
    - Custom metadata per source (no source_type / company / page labels)
    - Control over chunk boundaries (token-based internally)
    - Custom conflict-resolution prompts
    - Visibility into what's happening inside

  Line count: ~30 lines of setup vs ~600 lines manual
    """)

    print(f"  Demo query: \"{q1}\"")
    answer_li = llamaindex.query(q1)
    print_answer("LlamaIndex", answer_li)

    print("\n  Retrieved sources (no metadata label — that's the tradeoff):")
    for r in llamaindex.retrieve_only(q1):
        print(f"    Rank {r['rank']}: {r['source']} (score: {r['score']:.4f})")

    # -----------------------------------------------------------------------
    # STEP 3: LangChain Pipeline
    # -----------------------------------------------------------------------
    print_header(3, "LangChain — Chain-Centric Framework")
    print("""
  LangChain philosophy:
    Compose steps as a chain. Each step is explicit and swappable.
    The chain is the unit — not the index.

  What it compresses vs Session 8:
    DirectoryLoader / PyPDFLoader → replaces loaders.py
    RecursiveCharacterTextSplitter→ replaces chunking.py (character-based)
    LangChain FAISS wrapper        → replaces vector_store.py
    RetrievalQA chain              → replaces retriever + prompt + LLM

  Key difference from LlamaIndex:
    - Character-based chunking (not word/token-based)
    - Chain composition is explicit (RetrievalQA.from_chain_type)
    - Better for multi-step workflows and tool integration
    - Weaker on pure document-indexing use cases

  LangChain shines when:
    - You need to chain retrieval + tool calls + external APIs
    - You're building agents (Session 10)
    - Your system has multiple retrieval sources with routing logic
    """)

    print(f"  Demo query: \"{q1}\"")
    answer_lc = langchain.query(q1)
    print_answer("LangChain", answer_lc)

    # -----------------------------------------------------------------------
    # STEP 4: Framework Comparison
    # -----------------------------------------------------------------------
    print_header(4, "Framework Comparison — Same Question, Three Answers")
    print("""
  Now let's run the same question through all three and compare.
  Look at:
    1. Answer quality — are they substantially different?
    2. Source visibility — which shows you WHERE it retrieved from?
    3. Control — which would you trust in production?
    """)

    compare_answers(
        question="Has Microsoft changed its interview format for 2025?",
        manual_pipeline=manual,
        llamaindex_pipeline=llamaindex,
        langchain_pipeline=langchain,
    )

    compare_answers(
        question="Which companies have added AI-related assessments to their interview process in 2025?",
        manual_pipeline=manual,
        llamaindex_pipeline=llamaindex,
        langchain_pipeline=langchain,
    )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("  SESSION 9 COMPLETE")
    print("=" * 70)
    print("""
  Framework comparison summary:

  ┌─────────────────────┬──────────────┬──────────────┬──────────────┐
  │                     │ Manual (S8)  │  LlamaIndex  │  LangChain   │
  ├─────────────────────┼──────────────┼──────────────┼──────────────┤
  │ Setup code          │ ~600 lines   │ ~30 lines    │ ~40 lines    │
  │ Mental model        │ Full control │ Index-first  │ Chain-first  │
  │ Chunking            │ Word-based   │ Token-based  │ Char-based   │
  │ Source metadata     │ Full (S8)    │ Filename only│ Filename only│
  │ Custom prompts      │ Yes          │ Hard         │ Configurable │
  │ Agent support       │ No           │ Yes (S10)    │ Yes (S10)    │
  │ Prototype speed     │ Slow         │ Fast         │ Fast         │
  └─────────────────────┴──────────────┴──────────────┴──────────────┘

  Lesson:
    Frameworks are not better than manual. They make different tradeoffs.
    After building it manually in Sessions 7–8, you now understand what
    LlamaIndex and LangChain are hiding — and when that matters.

  Coming in Session 10:
    - BM25 / keyword retrieval baseline
    - Query rewriting (pre-retrieval transformation)
    - Dense + keyword hybrid search
    - Agentic retrieval patterns
    """)


if __name__ == "__main__":
    main()
