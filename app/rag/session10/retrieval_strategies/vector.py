"""
retrieval_strategies/vector.py
-------------------------------
Semantic (dense) retriever using FAISS and sentence-transformer embeddings.

Wraps the existing Session 7 components (HuggingFaceEmbedder, FAISSVectorStore)
into a clean class that separates index-building from query-time retrieval.

Index lifecycle:
    - Index is built ONCE in __init__ (or via from_records)
    - retrieve() performs search only — no re-indexing
    - This mirrors how a production vector DB works (load once, query many)

Usage (standalone):
    retriever = VectorRetriever(data_dir="data/interview_prep")

Usage (shared preprocessing — avoids duplicating load/chunk work):
    records = preprocess(data_dir)
    retriever = VectorRetriever.from_records(records)
"""

import numpy as np
from typing import List, Dict, Optional

from app.rag.embedding import HuggingFaceEmbedder
from app.rag.vector_store import FAISSVectorStore
from app.rag.loaders import load_directory
from app.rag.normalization import normalize_documents
from app.rag.chunking import build_chunks_with_metadata


class VectorRetriever:
    """
    Dense semantic retriever backed by FAISS.

    Builds the embedding index once at construction time.
    All subsequent retrieve() calls are fast lookups.

    Attributes:
        embedder:  HuggingFaceEmbedder instance for encoding queries
        store:     FAISSVectorStore with the built index
        records:   list of chunk dicts (for reference / debugging)
    """

    def __init__(
        self,
        data_dir: str = "data/interview_prep",
        chunk_size: int = 200,
        overlap: int = 30,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Loads documents, chunks them, embeds, and builds the FAISS index.

        This is the expensive step — happens once per VectorRetriever instance.
        If you already have preprocessed records, use from_records() instead
        to avoid duplicating the load/normalize/chunk work.

        Args:
            data_dir:             path to the source document directory.
            chunk_size:           words per chunk.
            overlap:              overlapping words between chunks.
            embedding_model_name: sentence-transformers model identifier.
        """
        print("\n" + "=" * 60)
        print("BUILDING VECTOR RETRIEVER (Session 10)")
        print("=" * 60)

        # Step 1: Load and normalize documents
        raw_docs = load_directory(data_dir)
        documents = normalize_documents(raw_docs)

        # Step 2: Chunk with metadata
        records = build_chunks_with_metadata(
            documents, chunk_size=chunk_size, overlap=overlap
        )

        # Step 3-4: Embed + build index
        self._build_index(records, embedding_model_name)

    @classmethod
    def from_records(
        cls,
        records: List[Dict],
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> "VectorRetriever":
        """
        Builds a VectorRetriever from pre-processed chunk records.

        Use this when multiple retrievers share the same preprocessed corpus
        to avoid duplicating the load → normalize → chunk pipeline.

        Args:
            records:              list of chunk dicts (from build_chunks_with_metadata).
            embedding_model_name: sentence-transformers model identifier.

        Returns:
            A fully initialized VectorRetriever.
        """
        print("\n" + "=" * 60)
        print("BUILDING VECTOR RETRIEVER (from shared records)")
        print("=" * 60)

        instance = cls.__new__(cls)
        instance._build_index(records, embedding_model_name)
        return instance

    def _build_index(
        self,
        records: List[Dict],
        embedding_model_name: str,
    ) -> None:
        """Internal: embeds records and builds the FAISS index."""
        self.records = records

        # Embed all chunks (one-time cost)
        self.embedder = HuggingFaceEmbedder(model_name=embedding_model_name)
        texts = [r["text"] for r in self.records]
        embeddings = self.embedder.embed_documents(texts)

        # Build FAISS index (one-time cost)
        self.store = FAISSVectorStore()
        self.store.build_index(embeddings, self.records)

        print(f"[VectorRetriever] Ready — {len(self.records)} chunks indexed\n")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Searches the pre-built index for semantically similar chunks.

        This is a FAST operation — just embeds the query and searches FAISS.

        Args:
            query:  the user's question.
            top_k:  number of results to return.

        Returns:
            List of dicts with: rank, source, source_type, company, page,
            chunk_id, text, score, retriever
        """
        query_embedding = self.embedder.embed_query(query)
        results = self.store.search(query_embedding, top_k=top_k)

        # Tag each result with the retriever name
        for r in results:
            r["retriever"] = "vector"

        return results
