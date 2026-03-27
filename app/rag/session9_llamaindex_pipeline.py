"""
rag/session9_llamaindex_pipeline.py
-------------------------------------
Session 9: LlamaIndex-Based RAG Pipeline

Why this exists:
    Session 8 built RAG manually — loader → normalizer → chunker → embedder
    → FAISS → retriever → prompt → LLM. That's ~8 modules, ~600 lines.

    LlamaIndex can do a comparable pipeline in ~30 lines.

    Teaching point:
        Frameworks are not magic. They do the same steps, but abstract
        them away. After Session 8, students should be able to read this
        code and recognise every concept they already built manually.

What LlamaIndex is best at:
    - Document-centric use cases (index a folder, query it)
    - Complex index types: tree, vector, keyword, graph (Session 10+)
    - Composable query pipelines

What we use here:
    - llama-index-core                  (indexing + querying)
    - llama-index-embeddings-huggingface (local HF embeddings, no OpenAI needed)
    - llama-index-llms-groq             (Groq LLM integration)
    - SimpleDirectoryReader             (loads .md and .pdf automatically)
    - VectorStoreIndex                  (in-memory FAISS equivalent)

Conceptual note on Agentic RAG (for instructors):
    LlamaIndex also has an 'agent' abstraction (ReAct, OpenAI function
    agents) where the LLM decides WHICH index to query and WHEN.
    We are NOT using that here — just the simple query engine.
    Agentic patterns belong to Session 10.
"""

import os
from typing import Optional

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq


class LlamaIndexPipeline:
    """
    Compact LlamaIndex-based RAG pipeline for Interview Prep data.

    Contrast with Session 8's InterviewRAGPipeline:
        - Same data, similar results
        - Less code (~30 lines of setup vs ~80)
        - Less control (chunking, metadata handling is abstracted)
        - Faster to prototype, harder to customise

    Args:
        data_dir:   path to directory with .md and .pdf files
        top_k:      number of chunks to retrieve per query
        chunk_size: words per chunk (LlamaIndex uses tokens, approx equiv)
    """

    def __init__(
        self,
        data_dir: str = "data/interview_prep",
        top_k: int = 5,
        chunk_size: int = 512,
    ):
        self.top_k = top_k

        print("\n" + "=" * 60)
        print("BUILDING LLAMAINDEX RAG PIPELINE")
        print("=" * 60)

        # --- Step 1: Configure embedding model (same HF model as Session 8) ---
        print("\n[LlamaIndex] Loading HuggingFace embedding model...")
        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # --- Step 2: Configure Groq LLM ---
        print("[LlamaIndex] Configuring Groq LLM...")
        groq_api_key = os.getenv("GROQ_API_KEY", "")
        groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

        if not groq_api_key:
            print("[LlamaIndex] WARNING: GROQ_API_KEY not set — LLM calls will fail")

        llm = Groq(model=groq_model, api_key=groq_api_key)

        # Apply globally (LlamaIndex's preferred config pattern)
        Settings.embed_model = embed_model
        Settings.llm = llm
        Settings.chunk_size = chunk_size

        # --- Step 3: Load documents (LlamaIndex handles .md and .pdf natively) ---
        print(f"\n[LlamaIndex] Reading documents from '{data_dir}'...")
        reader = SimpleDirectoryReader(input_dir=data_dir, recursive=False)
        documents = reader.load_data()
        print(f"[LlamaIndex] Loaded {len(documents)} document pages/files")

        # --- Step 4: Build the vector index (chunking + embedding happens here) ---
        print("[LlamaIndex] Building VectorStoreIndex (chunking + embedding)...")
        self.index = VectorStoreIndex.from_documents(documents)

        # --- Step 5: Create a query engine ---
        self.query_engine = self.index.as_query_engine(similarity_top_k=top_k)

        print("\n" + "=" * 60)
        print("LLAMAINDEX PIPELINE READY")
        print("=" * 60 + "\n")

    def query(self, question: str) -> str:
        """
        Runs a full RAG query: retrieve → build context → LLM answer.

        Returns the LLM's answer as a plain string.

        Note: LlamaIndex handles context building and prompting internally.
              To see how it differs from Session 8's explicit prompt, compare
              the answers on the same question.
        """
        print(f"\n[LlamaIndex] Query: \"{question}\"")
        response = self.query_engine.query(question)
        return str(response)

    def retrieve_only(self, question: str) -> list:
        """
        Returns retrieved nodes WITHOUT calling the LLM.
        Useful for comparing retrieval results against BM25 / FAISS.

        Returns:
            List of dicts: {rank, text, score, source}
        """
        retriever = self.index.as_retriever(similarity_top_k=self.top_k)
        nodes = retriever.retrieve(question)

        results = []
        for rank, node in enumerate(nodes, start=1):
            results.append({
                "rank": rank,
                "text": node.node.text[:300],
                "score": float(node.score) if node.score else 0.0,
                "source": node.node.metadata.get("file_name", "unknown"),
                "retriever": "llamaindex",
            })
        return results
