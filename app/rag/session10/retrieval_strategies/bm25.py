"""
retrieval_strategies/bm25.py
-----------------------------
Keyword-based retriever using BM25Okapi (Okapi BM25 variant).

BM25 is a probabilistic ranking function based on term frequency,
inverse document frequency, and document length normalization.
It requires NO embeddings and NO GPU.

When BM25 wins over vector retrieval:
    - Exact company/product names ("Oracle OCI", "Amazon LP")
    - Rare technical terms not well-represented in embedding models
    - Acronyms and abbreviations ("SDE", "DSA", "HLD", "LLD")

When it loses:
    - Paraphrased or semantic queries
    - Cross-lingual queries
    - Questions that don't share vocabulary with the answer

Tokenization design (per review feedback):
    The tokenizer preserves meaningful technical tokens:
    - Keeps alphanumeric characters and hyphens within words
    - Preserves acronyms (SDE, DSA, OCI)
    - Preserves hyphenated terms (low-level, object-oriented)
    - Preserves version numbers (v2, 3.5)
    - Lowercases for case-insensitive matching
    - Strips pure punctuation tokens

Index lifecycle:
    - BM25 index is built ONCE in __init__ (or via from_records)
    - retrieve() only scores the query against the pre-built index
"""

import re
from typing import List, Dict, Optional

from rank_bm25 import BM25Okapi

from app.rag.loaders import load_directory
from app.rag.normalization import normalize_documents
from app.rag.chunking import build_chunks_with_metadata


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

# Characters to keep: letters, digits, hyphens (within words), periods (versions)
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-\.][a-z0-9]+)*", re.IGNORECASE)


def _tokenize(text: str) -> List[str]:
    """
    Tokenizes text for BM25 indexing/matching.

    Design choices:
    - Lowercases everything for case-insensitive matching
    - Preserves hyphenated terms (e.g., "object-oriented", "low-level")
    - Preserves dotted terms (e.g., "3.5", "v2.0")
    - Preserves acronyms as-is after lowering (e.g., "SDE" -> "sde")
    - Drops pure punctuation and whitespace tokens
    - Does NOT stem/lemmatize (keeps it simple and predictable)

    Args:
        text: raw text to tokenize.

    Returns:
        List of lowercase tokens.
    """
    return _TOKEN_PATTERN.findall(text.lower())


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class BM25Retriever:
    """
    Keyword-based retriever using BM25Okapi.

    Builds the BM25 index once at construction time.
    All subsequent retrieve() calls are fast scoring operations.

    Usage (standalone):
        retriever = BM25Retriever(data_dir="data/interview_prep")

    Usage (shared preprocessing):
        records = preprocess(data_dir)
        retriever = BM25Retriever.from_records(records)
    """

    def __init__(
        self,
        data_dir: str = "data/interview_prep",
        chunk_size: int = 200,
        overlap: int = 30,
    ):
        """
        Loads documents, chunks them, tokenizes, and builds the BM25 index.

        If you already have preprocessed records, use from_records() instead
        to avoid duplicating the load/normalize/chunk work.

        Args:
            data_dir:    path to the source document directory.
            chunk_size:  words per chunk.
            overlap:     overlapping words between chunks.
        """
        print("\n" + "=" * 60)
        print("BUILDING BM25 KEYWORD RETRIEVER (Session 10)")
        print("=" * 60)

        # Step 1: Load -> Normalize -> Chunk (same pipeline as VectorRetriever)
        raw_docs = load_directory(data_dir)
        documents = normalize_documents(raw_docs)
        records = build_chunks_with_metadata(
            documents, chunk_size=chunk_size, overlap=overlap
        )

        # Step 2: Build BM25 index
        self._build_index(records)

    @classmethod
    def from_records(cls, records: List[Dict]) -> "BM25Retriever":
        """
        Builds a BM25Retriever from pre-processed chunk records.

        Use this when multiple retrievers share the same preprocessed corpus
        to avoid duplicating the load -> normalize -> chunk pipeline.

        Args:
            records: list of chunk dicts (from build_chunks_with_metadata).

        Returns:
            A fully initialized BM25Retriever.
        """
        print("\n" + "=" * 60)
        print("BUILDING BM25 KEYWORD RETRIEVER (from shared records)")
        print("=" * 60)

        instance = cls.__new__(cls)
        instance._build_index(records)
        return instance

    def _build_index(self, records: List[Dict]) -> None:
        """Internal: tokenizes records and builds the BM25 index."""
        self.records = records

        # Tokenize all chunks for BM25
        print("[BM25] Tokenizing corpus...")
        self._corpus_tokens = [_tokenize(r["text"]) for r in self.records]
        self.bm25 = BM25Okapi(self._corpus_tokens)

        # Summary stats
        total_tokens = sum(len(t) for t in self._corpus_tokens)
        avg_tokens = total_tokens / len(self._corpus_tokens) if self._corpus_tokens else 0

        print(
            f"[BM25] Ready — {len(self.records)} chunks indexed, "
            f"{total_tokens} total tokens (avg {avg_tokens:.0f}/chunk)\n"
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Scores all chunks against the query using BM25 and returns top-k.

        This is a FAST operation — no embeddings, just token matching.

        Note on scores:
            BM25 scores are NOT probabilities or cosine similarities.
            They are unbounded positive floats (higher = more relevant).
            Do NOT compare raw BM25 scores with vector cosine scores —
            use normalized scores (see scoring.py).

        Args:
            query:  the user's question.
            top_k:  number of results to return.

        Returns:
            List of dicts with: rank, source, source_type, company, page,
            chunk_id, text, score, retriever
        """
        query_tokens = _tokenize(query)
        raw_scores = self.bm25.get_scores(query_tokens)

        # Pair scores with indices, sort descending, take top_k
        scored = sorted(
            enumerate(raw_scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        results = []
        for rank, (idx, score) in enumerate(scored, start=1):
            record = self.records[idx]
            results.append({
                "rank": rank,
                **record,
                "score": float(score),
                "retriever": "bm25",
            })

        return results
