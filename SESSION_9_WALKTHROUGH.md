# Session 9: Advanced RAG Frameworks
## Manual Pipeline vs LlamaIndex vs LangChain

**Branch:** `session-9-advanced-rag-frameworks`
**Prerequisite:** Complete Session 8 (`session-8-interview-rag-ingestion`)

---

## Setup

```bash
git fetch origin
git checkout session-9-advanced-rag-frameworks
pip install -r requirements.txt
```

New dependencies for Session 9:
```
llama-index-core
llama-index-embeddings-huggingface
llama-index-llms-groq
langchain
langchain-community
langchain-groq
```

> **Before class:** Run `python -m app.rag.session9_playground` once to confirm all three pipelines build successfully. These frameworks have version-sensitive dependencies — dry-run first.

---

## What This Session Covers

After building RAG manually in Sessions 7 and 8, Session 9 answers:

> *"Was all that code necessary? What do frameworks give us — and what do they take away?"*

Three pipelines. Same data. Same question. Different answers to the tradeoff.

| | Manual (Session 8) | LlamaIndex | LangChain |
|--|---------------------|-----------|-----------|
| **Setup** | ~600 lines | ~30 lines | ~40 lines |
| **Mental model** | Full control | Index-first | Chain-first |
| **Chunking** | Word-based | Token-based | Character-based |
| **Source metadata** | Full (company, type, page) | Filename only | Filename only |
| **Custom prompts** | Yes | Hard | Configurable |
| **Agent support** | No | Yes (Session 10) | Yes (Session 10) |

---

## Running the Demo

### Full 4-step playground

```bash
python -m app.rag.session9_playground
```

**Step 1** — Manual Session 8 pipeline recap and live query
**Step 2** — LlamaIndex: same data, same embedding model, ~30 lines
**Step 3** — LangChain: chain-centric abstraction, character-based chunking
**Step 4** — Side-by-side comparison: two questions through all three

### Run the comparison directly

```python
from app.rag.session9_comparison import compare_answers
from app.rag.interview_pipeline import InterviewRAGPipeline
from app.rag.session9_llamaindex_pipeline import LlamaIndexPipeline
from app.rag.session9_langchain_pipeline import LangChainPipeline

manual    = InterviewRAGPipeline(provider="groq", data_dir="data/interview_prep")
llamaindex = LlamaIndexPipeline(data_dir="data/interview_prep")
langchain  = LangChainPipeline(data_dir="data/interview_prep")

compare_answers(
    question="Has Microsoft changed its interview format for 2025?",
    manual_pipeline=manual,
    llamaindex_pipeline=llamaindex,
    langchain_pipeline=langchain,
)
```

---

## New Files in Session 9

| File | Role |
|------|------|
| `app/rag/session9_llamaindex_pipeline.py` | LlamaIndex compact RAG |
| `app/rag/session9_langchain_pipeline.py` | LangChain RetrievalQA chain |
| `app/rag/session9_comparison.py` | 3-way answer comparison runner |
| `app/rag/session9_playground.py` | 4-step classroom demo |

**Moved to Session 10 (not taught today):**

| File | Belongs to |
|------|-----------|
| `app/rag/session10_keyword_retriever.py` | Session 10 — BM25 baseline |
| `app/rag/session10_query_rewriter.py` | Session 10 — query rewriting |

---

## Key Teaching Points

### LlamaIndex — Index-Centric

- Central abstraction: the **index**
- `SimpleDirectoryReader` → `VectorStoreIndex` → `query_engine.query()`
- Best for: document-heavy use cases, rapid prototyping
- Tradeoff: less visibility into chunking, metadata, prompts

### LangChain — Chain-Centric

- Central abstraction: the **chain**
- Compose steps explicitly: loader → splitter → vectorstore → retriever → LLM
- Best for: multi-step workflows, tool integration, future agent patterns
- Tradeoff: markdown preprocessing from Session 8 not mirrored (different chunks → slightly different answers)

### The Core Lesson

> Frameworks don't eliminate complexity — they move it.
> After Sessions 7–8 you know what's inside the black box.
> That's exactly when frameworks become useful instead of dangerous.

---

## 10 Demo Questions

| # | Question | Watch for |
|---|----------|-----------|
| 1 | "What data structures are commonly tested at Amazon?" | All three should agree on source |
| 2 | "Has Microsoft changed its interview format for 2025?" | Manual labels it [PRIVATE]; frameworks don't |
| 3 | "How many coding rounds does Google conduct for SDE roles?" | Conflict: public says 5, PDF says 3 |
| 4 | "What system design topics should I prepare for Google interviews?" | Source visibility difference |
| 5 | "Which companies added AI assessments in 2025?" | Multi-company retrieval |
| 6 | "What behavioral questions are common at Adobe?" | Single company, public source |
| 7 | "What is Oracle's OCI interview process?" | Exact term — all frameworks should retrieve |
| 8 | "What is the typical interview timeline for Amazon hires?" | Answer quality comparison |
| 9 | "Which company has the most rounds overall?" | Cross-document reasoning |
| 10 | "What advice do you give for someone preparing for FAANG in 2025?" | Open-ended, tests context quality |

---

## What's NOT in Session 9

These belong to Session 10:

- BM25 / keyword retrieval baseline
- Query rewriting / pre-retrieval transformation
- Dense vs keyword comparison (hybrid search)
- Agentic RAG patterns
- Graph RAG concepts

---

## Framework Gotchas (Know Before Class)

**LangChain chunking:** Uses `RecursiveCharacterTextSplitter` (character-based), not Session 8's word-based chunker. Chunk boundaries differ → context differs → answers may differ slightly. **This is a good teaching point, not a bug.**

> "Changing the chunking strategy changes the answer. That's why chunking is a design decision, not a detail."

**LlamaIndex PDF loading:** `SimpleDirectoryReader` handles PDFs natively but token-based chunking differs from Session 8. Expect slightly different retrieval on PDF content.

**Version sensitivity:** LlamaIndex and LangChain release frequently. Pin versions in production. For class demos, install fresh and run `session9_playground.py` before students arrive.
