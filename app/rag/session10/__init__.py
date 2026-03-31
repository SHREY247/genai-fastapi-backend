"""
app/rag/session10/
------------------
Session 10: Retrieval Optimization

Fixes retrieval failures by introducing multi-signal retrieval:
  - Vector retrieval (semantic meaning via FAISS)
  - BM25 retrieval (keyword matching)
  - Hybrid retrieval (vector + BM25 fusion)
  - Query rewriting (LLM-assisted intent clarification)

Design principle:
    Retrieval is a multi-signal system. Each retriever captures
    a different signal. This session teaches students WHEN and WHY
    to combine them.
"""
