"""
rag/interview_pipeline.py
-------------------------
Session 8: Interview Prep RAG Pipeline

Extends the Session 7 architecture to support multi-source ingestion:
  Loaders -> Normalization -> Chunking (metadata-aware) -> Embedding -> FAISS -> Retrieval -> Generation

This pipeline ingests both:
  - Public markdown files (company interview guides)
  - Private PDF files (confidential hiring updates)

Usage:
    pipeline = InterviewRAGPipeline(data_dir="data/interview_prep")
    result   = pipeline.query("How many coding rounds does Google have?")
"""

from typing import Dict, List, Optional

from app.rag.loaders import load_directory
from app.rag.normalization import normalize_documents
from app.rag.chunking import build_chunks_with_metadata
from app.rag.embedding import HuggingFaceEmbedder
from app.rag.vector_store import FAISSVectorStore
from app.rag.retriever import Retriever
from app.rag.prompt_builder import build_interview_context, build_interview_prompt


class InterviewRAGPipeline:
    """
    End-to-end RAG pipeline for Session 8: Interview Prep Knowledge Assistant.

    Key differences from Session 7's RAGPipeline:
      - Loads from multiple file formats (markdown + PDF)
      - Normalizes all sources into a unified Document schema
      - Chunks carry rich metadata (source_type, company, page)
      - Prompts are source-aware for citation-enabled answers

    Args:
        provider:             LLM provider name (e.g. "groq", "openai")
        data_dir:             path to directory with .md and .pdf files
        chunk_size:           number of words per chunk
        overlap:              overlapping words between consecutive chunks
        top_k:                number of results to retrieve per query
        embedding_model_name: sentence-transformers model to use
    """

    def __init__(
        self,
        provider: str = "groq",
        data_dir: str = "data/interview_prep",
        chunk_size: int = 200,
        overlap: int = 30,
        top_k: int = 5,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.provider = provider
        self.top_k = top_k

        print("\n" + "=" * 60)
        print("BUILDING INTERVIEW PREP RAG PIPELINE")
        print("=" * 60)

        # Step 1: Load from all supported sources
        print("\n--- Step 1: Loading sources ---")
        raw_docs = load_directory(data_dir)

        # Step 2: Normalize into unified Document schema
        print("\n--- Step 2: Normalizing documents ---")
        self.documents = normalize_documents(raw_docs)

        # Step 3: Chunk with metadata preservation
        print("\n--- Step 3: Chunking with metadata ---")
        self.records = build_chunks_with_metadata(
            self.documents, chunk_size=chunk_size, overlap=overlap
        )

        # Step 4: Generate embeddings (reusing Session 7 module)
        print("\n--- Step 4: Generating embeddings ---")
        self.embedder = HuggingFaceEmbedder(model_name=embedding_model_name)
        texts = [r["text"] for r in self.records]
        embeddings = self.embedder.embed_documents(texts)

        # Step 5: Build FAISS index (reusing Session 7 module)
        print("\n--- Step 5: Building vector index ---")
        self.store = FAISSVectorStore()
        self.store.build_index(embeddings, self.records)

        # Step 6: Initialize retriever (reusing Session 7 module)
        self.retriever = Retriever(self.embedder, self.store)

        print("\n" + "=" * 60)
        print("INTERVIEW PREP RAG PIPELINE READY")
        print("=" * 60 + "\n")

    def get_ingestion_summary(self) -> Dict:
        """
        Returns a summary of what was ingested. Useful for demos.
        """
        companies = set()
        source_types = set()
        for doc in self.documents:
            companies.add(doc.company)
            source_types.add(doc.source_type)

        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.records),
            "companies": sorted(companies),
            "source_types": sorted(source_types),
        }

    def inspect_retrieval(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Returns retrieved results without calling the LLM.
        Useful for inspecting what the retriever finds and
        verifying that metadata is preserved.
        """
        k = top_k if top_k is not None else self.top_k
        return self.retriever.retrieve(query, top_k=k)

    def query(self, query: str, top_k: Optional[int] = None) -> Dict:
        """
        Full RAG flow: retrieve -> build source-aware context -> call LLM.

        Returns:
            Dict with keys: query, results, context, answer, sources_used
        """
        k = top_k if top_k is not None else self.top_k

        # Retrieve top-k chunks
        results = self.retriever.retrieve(query, top_k=k)

        # Build source-aware context and prompt
        context = build_interview_context(results)
        prompt = build_interview_prompt(query, context)

        # Call the existing LLM service
        try:
            from app.services.llm_service import ask_llm
            answer = ask_llm(self.provider, prompt)
        except Exception as e:
            answer = f"[LLM call failed: {e}]"

        # Extract unique sources used in the answer (with page-level attribution)
        sources_used = []
        seen = set()
        for r in results:
            key = (r.get("source", ""), r.get("company", ""), r.get("page"))
            if key not in seen:
                seen.add(key)
                sources_used.append({
                    "source": r.get("source"),
                    "source_type": r.get("source_type"),
                    "company": r.get("company"),
                    "page": r.get("page"),
                })


        return {
            "query": query,
            "results": results,
            "context": context,
            "answer": answer,
            "sources_used": sources_used,
        }
