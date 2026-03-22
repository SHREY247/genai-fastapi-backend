# GenAI FastAPI Backend

A **minimal, production-style** FastAPI backend built as a teaching scaffold for the **Applied Generative AI Engineering** course.

---

## Branch Overview

| Branch | Session | What It Covers |
|--------|---------|----------------|
| `feature/session-4-llm-gateway` | Session 4 | Provider-agnostic LLM gateway (Groq, OpenAI, Anthropic) |
| `feature/session-7-rag-foundations` | Session 7 | RAG foundations — embeddings, FAISS, grounded answers |

---

## What's New in Session 7

Session 7 adds a **standalone RAG (Retrieval-Augmented Generation) module** on top of the Session 4 gateway.

- **Document Ingestion**: Load `.txt` files from the `data/` folder
- **Chunking**: Compare whole-document retrieval vs chunked retrieval
- **Dense Embeddings**: sentence-transformers (`all-MiniLM-L6-v2`) — runs locally, no API key needed
- **FAISS Vector Store**: In-memory cosine-similarity search via normalized inner product
- **Grounded Answer Generation**: Retrieved context is fed to the existing LLM gateway
- **Playground Demo**: Run the full pipeline from the command line

### New Dependencies

```
sentence-transformers
faiss-cpu
numpy
```

### No Existing Code Changed

The Session 4 LLM gateway, providers, routes, and services are **completely untouched**. The only modified file is `requirements.txt` (3 lines added).

---

## Project Structure

```
genai-fastapi-backend/
├── app/
│   ├── main.py                  ← FastAPI entry-point
│   ├── api/
│   │   └── routes/
│   │       ├── health.py        ← GET  /health
│   │       └── chat.py          ← POST /ai/chat
│   ├── core/
│   │   ├── config.py            ← Centralised configuration
│   │   └── logging.py           ← Standardized logging setup
│   ├── models/
│   │   ├── request_models.py    ← Pydantic request schemas
│   │   └── response_models.py   ← Pydantic response schemas
│   ├── providers/               ← Session 4: Provider implementations
│   │   ├── base.py              ← Abstract base class
│   │   ├── groq_provider.py
│   │   ├── openai_provider.py
│   │   └── anthropic_provider.py
│   ├── services/
│   │   ├── llm_gateway.py       ← Session 4: Request dispatcher
│   │   └── llm_service.py       ← Thin service wrapper
│   └── rag/                     ← Session 7: RAG module
│       ├── __init__.py
│       ├── ingestion.py         ← Load .txt documents
│       ├── chunking.py          ← Whole-doc vs chunked records
│       ├── embedding.py         ← HuggingFaceEmbedder (MiniLM)
│       ├── vector_store.py      ← FAISS index (cosine similarity)
│       ├── retriever.py         ← Query embedding + search
│       ├── prompt_builder.py    ← Grounded prompt construction
│       ├── pipeline.py          ← RAGPipeline orchestration
│       └── playground.py        ← Standalone demo entrypoint
├── data/                        ← Session 7: Sample documents
│   ├── hr_policy_expanded.txt
│   ├── onboarding_guide.txt
│   ├── product_manual_expanded.txt
│   ├── reimbursement_policy.txt
│   └── security_guidelines.txt
├── requirements.txt
└── .env.example
```

---

## Configuration

1. Copy `.env.example` to `.env`.
2. Fill in the API keys for the providers you wish to use:

```env
GROQ_API_KEY=gsk_...
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-...
```

> **Note**: The embedding model runs locally — no API key needed for embeddings or FAISS retrieval. An LLM API key is only required for the final answer generation step.

---

## How to Run

### 1. Setup

```bash
# Activate virtual environment
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the FastAPI Server (Session 4)

```bash
uvicorn app.main:app --reload
```

Open `http://localhost:8000/docs` to test the multi-provider gateway via Swagger UI.

### 3. Run the RAG Playground (Session 7)

```bash
python -m app.rag.playground
```

This demonstrates:
1. **Step 1** — Retrieval without chunking (full-document baseline)
2. **Step 2** — Retrieval with chunking (improved precision, same query)
3. **Step 3** — Grounded LLM answers for 3 sample queries

---

## Supported LLM Providers

| Provider | Default Model |
|----------|---------------|
| Groq | `llama-3.3-70b-versatile` |
| OpenAI | `gpt-4o-mini` |
| Anthropic | `claude-3-5-sonnet-20240620` |

---

## Sample API Request (Session 4)

```json
{
  "provider": "openai",
  "prompt": "Explain vector databases in simple terms."
}
```
