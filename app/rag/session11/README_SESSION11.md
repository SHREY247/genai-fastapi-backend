# Session 11: RAG Evaluation & Observability

## What This Session Teaches

Session 11 builds on Session 10's retrieval strategies to add the **measurement and inspection layer** that turns RAG experimentation into engineering:

1. **Observability** — See exactly what your RAG pipeline does at every step
2. **Debug Visibility** — Clean, classroom-friendly terminal output for pipeline inspection
3. **Structured Comparison** — Run multiple strategies on the same query and compare results
4. **Evaluation Datasets** — Curated question/ground-truth pairs for systematic testing
5. **RAGAS-based Scoring** — Automated LLM-based evaluation metrics
6. **Side-by-Side Reporting** — Quantitative strategy comparison tables

### Core Insight

> "You can't improve what you can't measure."
>
> Session 10 taught HOW to retrieve better. Session 11 teaches how to PROVE it.

---

## Architecture

```
app/rag/session11/
├── __init__.py          # Package docstring
├── observability.py     # Core: run_strategy() — normalized strategy runner
├── debug_logger.py      # Pretty-printing utilities for terminal output
├── dataset.py           # Evaluation dataset loader
├── evaluator.py         # Basic scoring (fallback when RAGAS unavailable)
├── ragas_eval.py        # RAGAS integration for LLM-based evaluation
├── comparison.py        # Cross-strategy comparison orchestration
├── playground.py        # Classroom demo entry point
└── README_SESSION11.md  # This file

data/session11/
└── eval_questions.json  # 10 curated evaluation questions
```

### How It Connects to Session 10

Session 11 **imports from** Session 10 — it does not modify it:

```
Session 10 (retrieval)          Session 11 (evaluation)
┌─────────────────────┐        ┌──────────────────────────┐
│ Session10Pipeline    │◀───────│ observability.py          │
│ - vector retriever   │        │  └─ run_strategy()       │
│ - bm25 retriever     │        │                          │
│ - hybrid retriever   │        │ comparison.py            │
│ - query rewriter     │        │  └─ compare_on_dataset() │
└─────────────────────┘        │                          │
                                │ ragas_eval.py            │
                                │  └─ run_ragas_evaluation()│
                                └──────────────────────────┘
```

---

## Quick Start & Classroom Flow

We recommend running the following checks/demos in this exact order during class:

### 1. Observability (Retrieval-Only)
Fastest demo. Shows the pipeline chunks and prompt without waiting for an LLM:
```bash
python -m app.rag.session11.playground --observe --no-answers
```
*Checks: retrieved chunks appear, prompt preview appears, answer says "RETRIEVAL-ONLY MODE".*

### 2. Observability (With LLM Answer)
Same as above, but completes the final step:
```bash
python -m app.rag.session11.playground --observe
```
*Checks: pipeline executes fully, final output matches chunks.*

### 3. Strategy Comparison (Retrieval-Only)
Run all 5 strategies on the same query to compare retrieval and rewrite behaviour without heavy LLM latency:
```bash
python -m app.rag.session11.playground --compare --no-answers
```
*Checks: all strategies run, comparison table prints cleanly.*

### 4. Full Evaluation
Run all strategies across the full 10-question evaluation dataset to see the final quantitative score table:
```bash
python -m app.rag.session11.playground --eval
```
*Checks: basic evaluation completes.*

### 5. RAGAS Evaluation (Optional)
If the environment is stable and `ragas` + `datasets` are installed:
```bash
python -m app.rag.session11.playground --eval --ragas
```

---

### Interactive Mode (Mode D)

```bash
python -m app.rag.session11.playground --interactive
```

Type queries to see comparison results. Prefix with `debug:` for full observability output.

---

## Supported Strategies

| Name | Base | Rewrite | Description |
|------|------|---------|-------------|
| `vector` | vector | No | Semantic retrieval via FAISS embeddings |
| `bm25` | bm25 | No | Keyword retrieval via BM25Okapi |
| `hybrid` | hybrid | No | Fused vector + BM25 |
| `rewrite_vector` | vector | Yes | Query rewriting → vector retrieval |
| `rewrite_hybrid` | hybrid | Yes | Query rewriting → hybrid retrieval |

---

## Using the API Programmatically

### Run a single strategy with debug output

```python
from app.rag.session11.observability import run_strategy

result = run_strategy("hybrid", "What are Amazon's interview rounds?", debug=True)
# Returns structured dict with: query, rewritten_query, strategy,
# contexts, context_metadata, prompt, answer, timing_ms
```

### Compare strategies on a query

```python
from app.rag.session11.comparison import compare_single_query

results = compare_single_query("How should I prepare for a coding assessment?")
```

### Run evaluation

```python
from app.rag.session11.ragas_eval import run_ragas_comparison

# With RAGAS (if installed)
scores = run_ragas_comparison()

# Or with basic metrics
from app.rag.session11.comparison import compare_on_dataset
scores = compare_on_dataset()
```

---

## Evaluation Dataset

The eval dataset (`data/session11/eval_questions.json`) contains 10 curated questions with:

- **question** — the user's query
- **ground_truth** — expected correct answer
- **reference_answer** — longer reference for quality checks
- **expected_sources** — which source files should be retrieved

Questions are designed to cover different retrieval challenges:
- Keyword-heavy queries (BM25 advantage)
- Semantic queries (Vector advantage)
- Ambiguous queries (Hybrid + Rewrite advantage)
- Multi-source queries (tests retrieval breadth)

---

## Evaluation Metrics

### Basic Metrics (always available)

| Metric | Description |
|--------|-------------|
| Answer Presence | Keyword overlap between answer and ground truth |
| Source Coverage | Fraction of expected sources found in retrieval |
| Context Hit Rate | Fraction of chunks from expected sources |

### RAGAS Metrics (requires `ragas` package)

| Metric | Description |
|--------|-------------|
| Answer Relevancy | Is the answer relevant to the question? (LLM-judged) |
| Faithfulness | Is the answer grounded in retrieved context? (LLM-judged) |
| Context Precision | Are retrieved chunks relevant? (LLM-judged) |
| Context Recall | Do chunks cover the ground truth info? (LLM-judged) |

---

## Package/Version Caveats

- **RAGAS**: Tested with `ragas >= 0.1.0`. The API changed significantly between versions. This module tries both import styles (lowercase functions and PascalCase classes) for compatibility.
- **datasets**: Required by RAGAS for the HuggingFace Dataset format. Install with `pip install datasets`.
- **Graceful fallback**: If RAGAS fails to import or errors during evaluation, all code paths fall back to basic metrics without crashing.
- **LLM provider**: Uses `groq` by default (same as Session 10). Requires a valid `GROQ_API_KEY` in `.env`.

---

## Teaching Flow Suggestion

1. **Start with observability (no-answers)** — show students what's happening inside the pipeline first, avoiding LLM wait times.
2. **Show full observability** — turn on the LLM to prove the pipeline connects end-to-end.
3. **Compare strategies (no-answers)** — show that different strategies retrieve different chunks for the same question.
4. **Introduce evaluation** — show that we can measure quality systematically over a dataset.
5. **Discuss metrics** — what does each metric tell us? What are the limitations?
6. **Key takeaway**: RAG outputs cannot be trusted blindly. Measurement turns optimization into engineering.
