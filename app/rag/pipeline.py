"""
rag/pipeline.py
---------------
The main RAG pipeline that ties all components together:

  Ingestion -> Chunking -> Embedding -> FAISS Index -> Retrieval -> Prompt -> LLM

Usage:
    pipeline = RAGPipeline(provider="groq", data_dir="data", use_chunking=True)
    result   = pipeline.query("What is the hotel reimbursement limit?")
"""

from typing import Dict, Optional

from app.rag.ingestion import load_documents
from app.rag.chunking import build_whole_document_records, build_chunks
from app.rag.embedding import HuggingFaceEmbedder
from app.rag.vector_store import FAISSVectorStore
from app.rag.retriever import Retriever
from app.rag.prompt_builder import build_context, build_prompt


class RAGPipeline:
    """
    End-to-end RAG pipeline for Session 7.

    Args:
        provider:             LLM provider name (e.g. "groq", "openai", "anthropic")
        data_dir:             path to the folder containing .txt documents
        use_chunking:         if True, split documents into chunks; else use whole docs
        chunk_size:           number of words per chunk (only if use_chunking=True)
        overlap:              overlapping words between chunks
        top_k:                number of results to retrieve per query
        embedding_model_name: sentence-transformers model to use
    """

    def __init__(
        self,
        provider: str = "groq",
        data_dir: str = "data",
        use_chunking: bool = True,
        chunk_size: int = 300,
        overlap: int = 50,
        top_k: int = 3,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.provider = provider
        self.top_k = top_k

        # Step 1: Load documents
        print("\n" + "=" * 60)
        print("BUILDING RAG PIPELINE")
        print("=" * 60)
        documents = load_documents(data_dir)

        # Step 2: Create records (whole docs or chunks)
        if use_chunking:
            records = build_chunks(documents, chunk_size=chunk_size, overlap=overlap)
        else:
            records = build_whole_document_records(documents)

        # Step 3: Generate embeddings
        self.embedder = HuggingFaceEmbedder(model_name=embedding_model_name)
        texts = [r["text"] for r in records]
        embeddings = self.embedder.embed_documents(texts)

        # Step 4: Build FAISS index
        self.store = FAISSVectorStore()
        self.store.build_index(embeddings, records)

        # Step 5: Initialize retriever
        self.retriever = Retriever(self.embedder, self.store)

        print("=" * 60)
        print("RAG PIPELINE READY")
        print("=" * 60 + "\n")

    def inspect_retrieval(self, query: str) -> list:
        """
        Returns retrieved results without calling the LLM.
        Useful for inspecting what the retriever finds.
        """
        return self.retriever.retrieve(query, top_k=self.top_k)

    def query(self, query: str) -> Dict:
        """
        Full RAG flow: retrieve -> build context -> call LLM -> return result.

        Returns:
            Dict with keys: query, results, context, answer
        """
        # Retrieve top-k chunks
        results = self.retriever.retrieve(query, top_k=self.top_k)

        # Build context and prompt
        context = build_context(results)
        prompt = build_prompt(query, context)

        # Call the existing LLM service
        try:
            from app.services.llm_service import ask_llm
            answer = ask_llm(self.provider, prompt)
        except Exception as e:
            answer = f"[LLM call failed: {e}]"

        return {
            "query": query,
            "results": results,
            "context": context,
            "answer": answer,
        }
