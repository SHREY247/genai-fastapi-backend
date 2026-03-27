"""
rag/session9_langchain_pipeline.py
------------------------------------
Session 9: LangChain-Based RAG Pipeline

Why this exists:
    LangChain takes a different philosophy from LlamaIndex:
    - LlamaIndex:  index-first, document-centric, structured retrieval
    - LangChain:   chain-first, composition-focused, flexible data sources

    Both achieve RAG, but the mental model is different.
    After seeing both, students should understand that RAG is a *pattern*,
    not a specific library.

What LangChain is best at:
    - Chaining multiple steps with explicit control
    - Integrating diverse data sources and tools in one pipeline
    - Building agents that can decide what to do next (Session 10+)

What we use here:
    - langchain-community   (FAISS vectorstore, PDF loader, HF embeddings)
    - langchain-groq        (Groq LLM)
    - langchain             (chain composition)

    We use LangChain's FAISS wrapper (not our own FAISSVectorStore) so
    students can see LangChain's abstraction directly. The underlying
    index is still FAISS — same algorithm, different interface.

Conceptual note on Agentic RAG (for instructors):
    LangChain's agent framework (LCEL + tools + memory) allows the LLM
    to decide when to retrieve, when to call external APIs, and when to
    stop. We're using the simple RetrievalQA chain here — one retrieval,
    one LLM call. Full agents are Session 10 territory.
"""

import os
from typing import List, Dict, Optional

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS as LangChainFAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.documents import Document as LCDocument


class LangChainPipeline:
    """
    LangChain-based RAG pipeline for Interview Prep data.

    Contrast with Session 8's InterviewRAGPipeline and the LlamaIndex pipeline:
        - Chain-based composition (RetrievalQA)
        - RecursiveCharacterTextSplitter (character-based, not word-based)
        - LangChain's FAISS wrapper

    Args:
        data_dir:   path to directory with .md and .pdf files
        top_k:      number of chunks to retrieve per query
        chunk_size: characters per chunk (LangChain splitter is char-based)
        chunk_overlap: character overlap between chunks
    """

    def __init__(
        self,
        data_dir: str = "data/interview_prep",
        top_k: int = 5,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ):
        self.top_k = top_k

        print("\n" + "=" * 60)
        print("BUILDING LANGCHAIN RAG PIPELINE")
        print("=" * 60)

        # --- Step 1: Load documents ---
        print(f"\n[LangChain] Loading documents from '{data_dir}'...")
        raw_docs = self._load_directory(data_dir)
        print(f"[LangChain] Loaded {len(raw_docs)} raw document(s)")

        # --- Step 2: Split into chunks ---
        print("[LangChain] Splitting into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = splitter.split_documents(raw_docs)
        print(f"[LangChain] Created {len(chunks)} chunk(s)")

        # --- Step 3: Embed using same HF model as Session 8 ---
        print("[LangChain] Loading HuggingFace embedding model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # --- Step 4: Build FAISS vectorstore (LangChain wrapper) ---
        print("[LangChain] Building FAISS vectorstore...")
        self.vectorstore = LangChainFAISS.from_documents(chunks, embeddings)

        # --- Step 5: Configure Groq LLM ---
        print("[LangChain] Configuring Groq LLM...")
        groq_api_key = os.getenv("GROQ_API_KEY", "")
        groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

        if not groq_api_key:
            print("[LangChain] WARNING: GROQ_API_KEY not set — LLM calls will fail")

        llm = ChatGroq(model=groq_model, groq_api_key=groq_api_key)

        # --- Step 6: Build RetrievalQA chain ---
        self.chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",          # "stuff" = concatenate all chunks into context
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": top_k}),
            return_source_documents=True,
        )

        print("\n" + "=" * 60)
        print("LANGCHAIN PIPELINE READY")
        print("=" * 60 + "\n")

    def _load_directory(self, data_dir: str) -> List[LCDocument]:
        """
        Loads .md and .pdf files from a directory into LangChain Document objects.
        Uses PyPDFLoader for PDFs and a text reader for markdown.
        """
        from langchain_community.document_loaders import PyPDFLoader, TextLoader

        documents = []
        supported = {".md": TextLoader, ".pdf": PyPDFLoader}

        for filename in sorted(os.listdir(data_dir)):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in supported:
                continue
            filepath = os.path.join(data_dir, filename)
            try:
                loader_cls = supported[ext]
                loader = loader_cls(filepath)
                docs = loader.load()
                # Attach source metadata
                for doc in docs:
                    doc.metadata["file_name"] = filename
                documents.extend(docs)
                print(f"  [LangChain] Loaded: {filename} ({len(docs)} page(s))")
            except Exception as e:
                print(f"  [LangChain] WARNING: Could not load {filename}: {e}")

        return documents

    def query(self, question: str) -> Dict:
        """
        Runs the RetrievalQA chain: retrieve → stuff context → LLM answer.

        Returns:
            Dict: {"answer": str, "sources": List[str]}
        """
        print(f"\n[LangChain] Query: \"{question}\"")
        result = self.chain.invoke({"query": question})
        answer = result.get("result", "No answer returned")
        source_docs = result.get("source_documents", [])

        unique_sources = []
        if source_docs:
            unique_sources = sorted(list({
                doc.metadata.get("file_name", "unknown") for doc in source_docs
            }))
            print(f"[LangChain] Sources used: {unique_sources}")

        return {
            "answer": answer,
            "sources": unique_sources
        }

    def retrieve_only(self, question: str) -> List[Dict]:
        """
        Returns top-k retrieved documents WITHOUT calling the LLM.
        For comparison with FAISS/BM25 retrieval results.
        """
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})
        docs = retriever.invoke(question)

        results = []
        for rank, doc in enumerate(docs, start=1):
            results.append({
                "rank": rank,
                "text": doc.page_content[:300],
                "score": None,   # LangChain retriever doesn't expose scores by default
                "source": doc.metadata.get("file_name", "unknown"),
                "retriever": "langchain",
            })
        return results
