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

## What This Demo Proves

This session demonstrates three facts through live code:

| Fact | What You'll See |
|------|-----------------|
| Full-document retrieval fails | Top results have low, noisy scores. Rank 2–3 are completely unrelated documents. |
| Chunking fixes retrieval | Same query returns higher scores. All top-3 results come from the correct source. |
| Better retrieval = better answers | The LLM answer is grounded, specific, and accurate — not hallucinated. |

The **same query** is used in whole-doc mode and chunked mode back-to-back. The improvement is not theoretical. You will see the numbers change in the terminal.

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

---

### Step 1 — Inspect the data folder
**Goal**: Confirm the 5 source documents are present before anything runs.

```bash
ls -la data/
```

**Expected**: 5 `.txt` files — HR policy, onboarding guide, product manual, reimbursement policy, security guidelines.

---

### Step 2 — Ingest the documents
**Goal**: Show that loading is transparent — you see exactly what goes in.

```bash
python -c "
from app.rag.ingestion import load_documents
docs = load_documents('data')
for d in docs:
    print(f'{d[\"source\"]:45s} {len(d[\"text\"]):>5} chars')
"
```

**Expected**: 5 documents loaded, sizes ranging from ~4,000 to ~9,600 characters. Notice `hr_policy_expanded.txt` is large — reimbursement information is buried inside it.

---

### Step 3 — Compare whole-doc vs chunked record counts
**Goal**: Make the scale difference concrete before retrieval.

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

**Expected**:
```
Whole-doc records: 5
Chunked records  : 31
```

When we retrieve top-3 in whole-doc mode, we retrieve 60% of the entire corpus. With chunks, we retrieve 3 focused paragraphs. That precision gap is what this session is about.

---

### Step 4 — Show what an embedding looks like
**Goal**: Demystify vectors — text becomes a list of 384 numbers.

```bash
python -c "
from app.rag.embedding import HuggingFaceEmbedder
e = HuggingFaceEmbedder()
v = e.embed_query('hotel reimbursement limit')
print('Vector dimension:', len(v[0]))
print('First 5 values  :', v[0][:5].tolist())
"
```

**Expected**:
```
Vector dimension: 384
First 5 values  : [0.026..., 0.063..., -0.020..., 0.013..., -0.029...]
```

The exact values don't matter. What matters: 384 numbers capture the *meaning* of the query. "Hotel limit" and "maximum accommodation rate" will produce nearby vectors even though they share no words.

---

### Step 5 — THE PROOF: Why Chunking Wins
**Goal**: Show side-by-side that chunking produces higher scores and eliminates noise.

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

**Expected**:
```
=== WHOLE DOC ===
  0.5042  reimbursement_policy.txt                  doc-3
  0.1684  hr_policy_expanded.txt                    doc-0     ← noise
  0.1548  security_guidelines.txt                   doc-4     ← noise

=== CHUNKED ===
  0.5476  reimbursement_policy.txt                  chunk-1   ← relevant
  0.5042  reimbursement_policy.txt                  chunk-0   ← relevant
  0.4598  reimbursement_policy.txt                  chunk-2   ← relevant
```

**Observe**:
- Whole-doc: Rank 1 is correct but Rank 2–3 are completely wrong — the LLM would receive irrelevant context.
- Chunked: All 3 results come from the correct file. Scores are tighter and higher. The LLM receives focused, accurate context.
- This is why RAG without chunking often underperforms naive expectations.

---

### Step 6 — Full pipeline with grounded LLM answers
**Goal**: Show the end-to-end system: retrieval feeds the LLM and produces a grounded answer.

```bash
python -m app.rag.playground
```

**Expected**: Three queries answered with retrieved chunks printed before each answer. The LLM does not hallucinate — it quotes directly from the retrieved content.

---

### Step 7 — Bonus: answer any live audience question
**Goal**: Let the audience ask anything and show retrieval + answer in real time.

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

---

## Common Issues During Demo

| Issue | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: faiss` | `faiss-cpu` not installed | `pip install faiss-cpu` |
| `ModuleNotFoundError: sentence_transformers` | Missing dependency | `pip install sentence-transformers` |
| `GROQ_API_KEY is not configured` | `.env` not set up | `cp .env.example .env` then add key |
| Low similarity scores across all results | Query too vague or no matching content | Try a more specific query; check `data/` files |
| All top-3 results from wrong file | Chunking not enabled | Confirm `use_chunking=True` in pipeline |
| `FileNotFoundError: data/` | Running from wrong directory | Run all commands from repo root |
| Model download hangs | Slow network / HF Hub rate limit | Pre-download before class: run Step 4 once; model caches in `~/.cache/huggingface/` |
