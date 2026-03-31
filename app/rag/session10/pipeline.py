"""
session10/pipeline.py
---------------------
Pluggable retrieval pipeline for Session 10.

This is the single-query pipeline. It does NOT orchestrate comparisons
across strategies — that's the job of comparison.py.

Flow:
    Query
    -> (optional) query rewriting
    -> retrieval via selected strategy (vector / bm25 / hybrid)
    -> LLM answer generation

Timing:
    The pipeline returns granular timing for each phase (rewrite_ms,
    retrieval_ms, answer_ms) so that comparison.py can show WHERE
    time is spent. This is critical for Session 10 — students need to
    see that retrieval is fast and the LLM call dominates latency.

Strategy selection is done by passing a string name. The pipeline
holds references to pre-built retrievers and dispatches accordingly.

Shared preprocessing:
    Documents are loaded, normalized, and chunked ONCE. The shared
    records are passed to both VectorRetriever and BM25Retriever via
    their from_records() classmethods, avoiding duplicate I/O.
"""

import time
from typing import Dict, Optional, List

from app.rag.loaders import load_directory
from app.rag.normalization import normalize_documents
from app.rag.chunking import build_chunks_with_metadata

from app.rag.session10.retrieval_strategies.vector import VectorRetriever
from app.rag.session10.retrieval_strategies.bm25 import BM25Retriever
from app.rag.session10.retrieval_strategies.hybrid import HybridRetriever
from app.rag.session10.query_rewriter import rewrite_query
from app.rag.prompt_builder import build_interview_context, build_interview_prompt


# Strategy name -> retriever attribute mapping
_STRATEGY_NAMES = {"vector", "bm25", "hybrid"}


class Session10Pipeline:
    """
    End-to-end RAG pipeline with pluggable retrieval strategies.

    Lifecycle:
        1. Preprocess corpus once (load -> normalize -> chunk)
        2. Build all retrievers from shared records (expensive, in __init__)
        3. Call query() with different strategies as many times as needed

    Usage:
        pipe = Session10Pipeline(data_dir="data/interview_prep")
        result = pipe.query("Amazon interview rounds", strategy="hybrid", rewrite=True)
    """

    def __init__(
        self,
        data_dir: str = "data/interview_prep",
        chunk_size: int = 200,
        overlap: int = 30,
        provider: str = "groq",
        alpha: float = 0.5,
        beta: float = 0.5,
    ):
        """
        Preprocesses the corpus once and builds all three retrievers.

        The load -> normalize -> chunk pipeline runs ONCE. Both the
        VectorRetriever and BM25Retriever receive the same shared records,
        avoiding duplicate I/O and chunking work.

        Args:
            data_dir:    path to source documents.
            chunk_size:  words per chunk.
            overlap:     overlapping words between chunks.
            provider:    LLM provider for answer generation and rewriting.
            alpha:       hybrid weight for vector scores.
            beta:        hybrid weight for BM25 scores.
        """
        self.provider = provider

        # -----------------------------------------------------------
        # Shared preprocessing (runs ONCE for all retrievers)
        # -----------------------------------------------------------
        print("\n" + "=" * 60)
        print("PREPROCESSING CORPUS (shared across all retrievers)")
        print("=" * 60)

        raw_docs = load_directory(data_dir)
        documents = normalize_documents(raw_docs)
        shared_records = build_chunks_with_metadata(
            documents, chunk_size=chunk_size, overlap=overlap
        )

        print(f"[Pipeline] {len(shared_records)} shared chunks ready\n")

        # -----------------------------------------------------------
        # Build retrievers from shared records
        # -----------------------------------------------------------
        self.vector = VectorRetriever.from_records(shared_records)
        self.bm25 = BM25Retriever.from_records(shared_records)
        self.hybrid = HybridRetriever(
            vector_retriever=self.vector,
            bm25_retriever=self.bm25,
            alpha=alpha,
            beta=beta,
        )

        # Map strategy names to retriever instances
        self._strategies = {
            "vector": self.vector,
            "bm25": self.bm25,
            "hybrid": self.hybrid,
        }

        print("\n" + "=" * 60)
        print("SESSION 10 PIPELINE READY")
        print(f"  Strategies: {', '.join(sorted(self._strategies.keys()))}")
        print(f"  Provider:   {self.provider}")
        print("=" * 60 + "\n")

    def query(
        self,
        query: str,
        strategy: str = "hybrid",
        rewrite: bool = False,
        top_k: int = 5,
    ) -> Dict:
        """
        Runs a single query through the selected retrieval strategy.

        Returns granular timing for each phase so callers (comparison.py)
        can show WHERE latency comes from.

        Args:
            query:     the user's question.
            strategy:  one of "vector", "bm25", "hybrid".
            rewrite:   if True, rewrites the query before retrieval.
            top_k:     number of chunks to retrieve.

        Returns:
            Dict with keys:
                query            -- original user query
                rewritten_query  -- rewritten query (or None if rewrite=False)
                strategy         -- strategy name used
                results          -- list of retrieved chunk dicts
                context          -- formatted context string
                answer           -- LLM-generated answer
                timing_ms        -- dict with rewrite_ms, retrieval_ms,
                                    answer_ms, total_ms
        """
        if strategy not in self._strategies:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Choose from: {sorted(self._strategies.keys())}"
            )

        timing = {"rewrite_ms": 0.0, "retrieval_ms": 0.0, "answer_ms": 0.0}
        t_total_start = time.time()

        # ---- Phase 1: Optional query rewriting ----
        rewritten = None
        effective_query = query
        if rewrite:
            t0 = time.time()
            rewritten = rewrite_query(query, provider=self.provider)
            timing["rewrite_ms"] = (time.time() - t0) * 1000
            effective_query = rewritten

        # ---- Phase 2: Retrieval ----
        t0 = time.time()
        retriever = self._strategies[strategy]
        results = retriever.retrieve(effective_query, top_k=top_k)
        timing["retrieval_ms"] = (time.time() - t0) * 1000

        # ---- Phase 3: Context building + LLM answer ----
        context = build_interview_context(results)
        prompt = build_interview_prompt(query, context)

        t0 = time.time()
        try:
            from app.services.llm_service import ask_llm
            answer = ask_llm(self.provider, prompt)
        except Exception as e:
            answer = f"[LLM call failed: {e}]"
        timing["answer_ms"] = (time.time() - t0) * 1000

        timing["total_ms"] = (time.time() - t_total_start) * 1000

        return {
            "query": query,
            "rewritten_query": rewritten,
            "strategy": strategy,
            "results": results,
            "context": context,
            "answer": answer,
            "timing_ms": timing,
        }
