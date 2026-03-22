# GenAI FastAPI Backend — Session 7: RAG Foundations

A **minimal, modular** FastAPI backend for the **Applied Generative AI Engineering** course.

---

## System Framing — Read This First

This is **not** an LLM application with some retrieval bolted on.

This is a **retrieval system** with an LLM at the end.

The LLM is the last 5% of the pipeline. The other 95% — ingestion, chunking, embedding, indexing, retrieval — determines whether the LLM produces a useful answer or a hallucinated one.

**Core principle**: The quality of your retrieval defines the ceiling of your generation. A perfect LLM cannot fix bad retrieval. A mediocre LLM with excellent retrieval will outperform a frontier model with poor retrieval.

This session teaches you to build and reason about that retrieval system.

---

## Branch Overview

| Branch | Session | What It Covers |
|--------|---------|----------------|
| `feature/session-4-llm-gateway` | Session 4 | Provider-agnostic LLM gateway (Groq, OpenAI, Anthropic) |
| `feature/session-7-rag-foundations` | Session 7 | RAG foundations — embeddings, FAISS, grounded answers |

---

## Two Pipelines: Offline and Online

Every RAG system has two distinct execution paths. Understanding this separation is fundamental to system design.

### Offline Pipeline (Indexing)

Runs **once** (or on document update). Not latency-sensitive.

```
data/*.txt → ingestion → chunking → embedding → FAISS index
```

This is a **batch job**. You pay the cost of embedding all documents upfront. The result is a searchable vector index stored in memory.

### Online Pipeline (Query)

Runs **per user query**. Latency-sensitive.

```
query → embed query → FAISS search → build prompt → LLM → answer
```

Only **one** embedding call (the query) and one LLM call per request. The FAISS search itself is sub-millisecond. This is why RAG scales — the expensive work is done offline.

---

## What's New in Session 7

Session 7 adds a **standalone RAG module** on top of the Session 4 gateway.

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

---

### Step 1 — Inspect the raw data
**Goal**: Know your corpus before building anything.

```bash
ls -la data/
```

**Expected**: 5 `.txt` files — HR policy, onboarding guide, product manual, reimbursement policy, security guidelines.

---

### Step 2 — Ingest the documents
**Goal**: Verify the ingestion layer is transparent — you see exactly what enters the system.

```bash
python -c "
from app.rag.ingestion import load_documents
docs = load_documents('data')
for d in docs:
    print(f'{d[\"source\"]:45s} {len(d[\"text\"]):>5} chars')
"
```

**Expected**: 5 documents loaded, sizes ranging from ~4,000 to ~9,600 characters. Notice `hr_policy_expanded.txt` is large — reimbursement information is buried deep inside broader policy text. This is by design.

---

### Step 3 — Chunking: see the retrieval resolution change
**Goal**: Understand that chunking defines the granularity of what can be retrieved.

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

**Why this matters**: Chunking is the **single most important design decision** in a RAG system. It defines your retrieval resolution — the smallest unit that can be independently retrieved.

- **Too large** (whole-doc): retrieval returns everything, diluting relevance. Top-3 from 5 docs = 60% of corpus. That's not retrieval, that's a dump.
- **Too small** (10 words): individual chunks lose context, meaning is fragmented.
- **Right-sized** (~150–300 words with overlap): each chunk is a self-contained semantic unit. Top-3 from 31 chunks = focused, relevant context.

The overlap prevents information loss at chunk boundaries. A sentence split between two chunks is captured in both.

---

### Step 4 — Understand embeddings: text as vectors
**Goal**: See that text is projected into a 384-dimensional vector space where *meaning* defines proximity.

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

The exact values are irrelevant. What matters is the property these vectors have: `"hotel reimbursement limit"` and `"maximum accommodation rate per night"` will produce **nearby vectors** even though they share **zero words**. This is semantic similarity — the foundation of dense retrieval and the reason we use embeddings instead of keyword search.

The model (`all-MiniLM-L6-v2`) is a 22M-parameter transformer trained on millions of text pairs. It runs locally on CPU in milliseconds. No API key, no network call.

---

### Step 5 — THE PROOF: Why Chunking Wins

**Before you run this, predict the failure.**

Think about it: if you embed a 7,000-character HR policy document as a single vector, that vector represents *everything* — leave policies, performance reviews, reimbursement rules, and termination procedures all compressed into 384 numbers. When you search for "hotel reimbursement," the vector has to simultaneously represent all those topics. It will be a mediocre match for everything and an excellent match for nothing.

Now predict: will chunking fix this?

**Goal**: Confirm the prediction with real numbers.

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

**Analysis**:
- Whole-doc Rank 1 is correct, but Rank 2–3 are irrelevant. The LLM would receive security guidelines and HR code-of-conduct alongside reimbursement rules — guaranteed confusion.
- Chunked: all 3 results come from the correct file. Scores are tighter and higher. The LLM receives 3 focused paragraphs that directly answer the question.

**Why top-3 instead of top-1?** Retrieving a single result is brittle — the answer might span two chunks, or the best chunk might rank second due to embedding noise. Top-3 increases recall while keeping context small. This is a precision-recall tradeoff: too few results and you miss information, too many and you dilute the context. For this corpus and chunk size, `k=3` is the right balance.

---

### Retrieval Inspection Discipline

**Rule**: Never pass retrieval output blindly to the LLM.

Before every LLM call, inspect:
1. **Scores** — are they above a meaningful threshold? Scores below 0.3 are likely noise.
2. **Sources** — do the results come from the expected file(s)?
3. **Content** — does the preview text actually contain the answer?

Use `pipeline.inspect_retrieval(query)` to retrieve without calling the LLM. This is your debugging tool. If retrieval is wrong, the LLM answer *will* be wrong — no amount of prompt engineering will fix bad context.

---

### Step 6 — End-to-end: grounded LLM answers
**Goal**: Observe the full system — retrieval feeds the LLM, grounding prevents hallucination.

```bash
python -m app.rag.playground
```

**Expected**: Three queries answered with retrieved chunks printed before each answer. The LLM does not hallucinate — it quotes directly from retrieved content.

**Why grounded prompting works**: The prompt instructs the LLM to answer *only* from the provided context and to say "I don't know based on the provided documents" when the answer isn't present. This is not politeness — it's a hard constraint that exploits how attention mechanisms work. When context is placed before the question, the model attends to it preferentially. The explicit refusal instruction prevents the model from falling back to parametric memory when the context is insufficient.

---

### Step 7 — Bonus: answer any live audience question
**Goal**: Let the audience test the system with arbitrary queries.

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

## This Is Not Production

This system is designed for learning, not deployment. Know the gaps:

| Limitation | What production systems add |
|---|---|
| **No persistence** | Index is rebuilt from scratch on every run. Production uses persistent vector databases (Pinecone, Weaviate, pgvector). |
| **No re-ranking** | Results go directly from FAISS to the LLM. Production adds a cross-encoder re-ranker to refine relevance after retrieval. |
| **No hybrid search** | We use dense retrieval only. Production combines dense (semantic) + sparse (BM25/keyword) for robustness. |
| **No evaluation** | We eyeball results. Production measures recall@k, MRR, and answer faithfulness systematically. |
| **No streaming** | The LLM call blocks until complete. Production streams tokens for perceived latency reduction. |
| **Fixed chunk size** | One size for all documents. Production tunes per document type or uses semantic chunking. |

These are future sessions. Today's goal is to internalize the retrieval fundamentals that every production system builds on.

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

---

## Core Insight

> **RAG is a retrieval problem disguised as a generation problem.**
>
> Most teams debug the LLM when the answer is wrong. The fix is almost always in the retrieval layer — better chunking, better embeddings, better scoring. Fix retrieval first. The generation will follow.
