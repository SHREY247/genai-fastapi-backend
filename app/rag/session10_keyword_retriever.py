"""
rag/session9_keyword_retriever.py
----------------------------------
Session 9: Vectorless / Keyword Retrieval Baseline

Why this exists:
    Dense vector retrieval (FAISS) is powerful but not the only way.
    BM25 is a classic keyword-frequency algorithm (used by Elasticsearch
    by default) that requires NO embeddings, NO GPU, and NO neural models.

    Teaching point:
        Use this to show students *when* keyword retrieval wins over dense:
        - Queries with exact company names ("Oracle OCI")
        - Rare terms not well-represented in embedding training data
        - Very short queries that embed poorly

    And when it loses:
        - Paraphrased/semantic queries
        - Cross-lingual queries
        - Questions that don't share vocabulary with the answer

How it works:
    1. Tokenise all chunks at load time (whitespace + lowercase)
    2. BM25Okapi scores each chunk against the query tokens
    3. Top-k chunks returned in same format as FAISS retriever

Dependencies:
    rank-bm25  (added to requirements.txt)
"""

import os
import re
from typing import List, Dict, Optional

from rank_bm25 import BM25Okapi

from app.rag.loaders import load_directory
from app.rag.normalization import normalize_documents
from app.rag.chunking import build_chunks_with_metadata


def _tokenize(text: str) -> List[str]:
    """
    Simple whitespace + punctuation tokenizer.
    Lowercases and strips non-alphanumeric characters.

    In production you'd use a stemmer/lemmatizer (e.g. NLTK Porter).
    Keeping it simple here for teachability.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


class BM25Retriever:
    """
    Keyword-based retriever using BM25Okapi over chunked documents.

    This is the Session 9 'vectorless' baseline.

    Usage:
        retriever = BM25Retriever(data_dir="data/interview_prep")
        results   = retriever.retrieve("How many rounds does Amazon have?", top_k=5)
    """

    def __init__(
        self,
        data_dir: str = "data/interview_prep",
        chunk_size: int = 200,
        overlap: int = 30,
    ):
        self.data_dir = data_dir
        self.records: List[Dict] = []
        self.bm25: Optional[BM25Okapi] = None

        print("\n" + "=" * 60)
        print("BUILDING BM25 KEYWORD RETRIEVER")
        print("=" * 60)

        # Reuse Session 8 loaders + normalization + chunking
        print("\n[BM25] Loading and chunking documents...")
        raw_docs = load_directory(data_dir)
        documents = normalize_documents(raw_docs)
        self.records = build_chunks_with_metadata(
            documents, chunk_size=chunk_size, overlap=overlap
        )

        # Tokenise all chunks for BM25
        print("[BM25] Tokenising corpus for BM25 index...")
        corpus = [_tokenize(r["text"]) for r in self.records]
        self.bm25 = BM25Okapi(corpus)

        print(
            f"[BM25] Ready — {len(self.records)} chunks indexed "
            f"(no vectors, no GPU needed)\n"
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Returns top-k chunks by BM25 score for the given query.

        Result format mirrors FAISS retriever:
            {rank, source, source_type, company, page, chunk_id, text, score}

        Note: BM25 scores are NOT cosine similarities — they are
        unbounded floats (higher = more relevant). Don't compare raw
        scores across retrieval methods.
        """
        query_tokens = _tokenize(query)
        raw_scores = self.bm25.get_scores(query_tokens)

        # Pair scores with records, sort descending
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
