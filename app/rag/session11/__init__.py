"""
app/rag/session11/
------------------
Session 11: RAG Evaluation & Observability

Builds on top of Session 10's retrieval strategies to add:
  - End-to-end observability (inspect every step of a RAG pipeline run)
  - Debug logging (clean, classroom-friendly terminal output)
  - Evaluation dataset management (curated question/ground-truth pairs)
  - RAGAS-based automated evaluation (answer relevance, faithfulness, context)
  - Cross-strategy comparison with quantitative scoring
  - Playground for live classroom demos

Design principle:
    You can't improve what you can't measure. Session 10 taught students
    HOW to retrieve better. Session 11 teaches them how to PROVE it,
    inspect it, and compare strategies with real numbers.

Key constraint:
    Session 10 is treated as a stable baseline. This session wraps and
    extends it — no refactoring of Session 10 internals.
"""
