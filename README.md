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

## Getting Started (Students)

If you already have the repo cloned, fetch and switch to the Session 7 branch:

```bash
git fetch origin
git checkout feature/session-7-rag-foundations
```

If you don't have the repo yet:

```bash
git clone https://github.com/SHREY247/genai-fastapi-backend.git
cd genai-fastapi-backend
git checkout feature/session-7-rag-foundations
```

Then set up your environment:

```bash
python -m venv venv

# Activate virtual environment
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# Install all dependencies
pip install -r requirements.txt
```

Copy the environment file and add your API key(s):

```bash
cp .env.example .env
# Edit .env and fill in at least one provider key (e.g. GROQ_API_KEY)
```

You're ready — jump to **How to Run** below.

---

## How to Run

### 1. Activate Environment

```bash
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
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

---

## Session 7 — Step-by-Step Demo Commands

Run these in order during the live demo. Each command is self-contained and can be copy-pasted directly into the terminal.

### Step 1 — Show what data we have

```bash
ls -la data/
```

---

### Step 2 — Ingest the documents

```bash
python -c "
from app.rag.ingestion import load_documents
docs = load_documents('data')
for d in docs:
    print(f'{d[\"source\"]:45s} {len(d[\"text\"]):>5} chars')
"
```

---

### Step 3 — Compare whole-doc vs chunked record counts

```bash
python -c "
from app.rag.ingestion import load_documents
from app.rag.chunking import build_whole_document_records, build_chunks
docs = load_documents('data')
whole = build_whole_document_records(docs)
chunks = build_chunks(docs, chunk_size=200, overlap=30)
print('Whole-doc records:', len(whole))
print('Chunked records  :', len(chunks))
"
```

---

### Step 4 — Show what an embedding looks like

```bash
python -c "
from app.rag.embedding import HuggingFaceEmbedder
e = HuggingFaceEmbedder()
v = e.embed_query('hotel reimbursement limit')
print('Vector dimension:', len(v[0]))
print('First 5 values  :', v[0][:5].tolist())
"
```

---

### Step 5 — Key comparison: whole-doc vs chunked retrieval ⬅ highlight of the demo

```bash
python -c "
from app.rag.ingestion import load_documents
from app.rag.chunking import build_whole_document_records, build_chunks
from app.rag.embedding import HuggingFaceEmbedder
from app.rag.vector_store import FAISSVectorStore
from app.rag.retriever import Retriever

query = 'What is the maximum hotel reimbursement per night for domestic travel?'
embedder = HuggingFaceEmbedder()
docs = load_documents('data')

wr = build_whole_document_records(docs)
ws = FAISSVectorStore.from_embeddings(wr, embedder.embed_documents([r['text'] for r in wr]))
print('=== WHOLE DOC ===')
for r in Retriever(embedder, ws).retrieve(query, k=3):
    print(f'  {r[\"score\"]:.4f}  {r[\"source\"]:40s}  {r[\"chunk_id\"]}')

cr = build_chunks(docs, chunk_size=200, overlap=30)
cs = FAISSVectorStore.from_embeddings(cr, embedder.embed_documents([r['text'] for r in cr]))
print()
print('=== CHUNKED ===')
for r in Retriever(embedder, cs).retrieve(query, k=3):
    print(f'  {r[\"score\"]:.4f}  {r[\"source\"]:40s}  {r[\"chunk_id\"]}')
"
```

> Observe: Whole-doc Rank 2–3 are noise (HR handbook, security doc). Chunked Rank 1–3 all come from `reimbursement_policy.txt`.

---

### Step 6 — Full pipeline with grounded LLM answers

```bash
python -m app.rag.playground
```

---

### Step 7 — Bonus: try any question interactively

```bash
python -c "
from app.rag.pipeline import RAGPipeline
p = RAGPipeline(provider='groq', data_dir='data', use_chunking=True)
results = p.inspect_retrieval('YOUR QUESTION HERE')
for r in results:
    print(r['source'], r['chunk_id'], round(r['score'], 4))
    print(r['text'][:200])
    print()
"
```

Replace `YOUR QUESTION HERE` with anything the audience asks.
