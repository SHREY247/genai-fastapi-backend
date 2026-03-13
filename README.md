# GenAI FastAPI Backend - Session 4 LLM Gateway

A **minimal, production-style** FastAPI backend that implements a **Provider-Agnostic LLM Gateway**.  
Built as a teaching scaffold for Session 4 of the **Applied Generative AI Engineering** course.

In Session 4, we evolved the backend from a single-provider (Groq) implementation into a modular gateway that supports **Groq, OpenAI, and Anthropic**.

---

## What's New in Session 4

- **Modular Backend Architecture**: Separated provider-specific logic from route and service layers.
- **LLM Gateway**: A central dispatcher that routes requests to the selected provider.
- **Provider Abstraction**: Common interface for adding new LLM providers easily.
- **Standardized Logging**: Application-wide logging for better observability.
- **Improved Error Handling**: Graceful handling of missing API keys and unsupported providers.

---

## Project Structure

```
genai-fastapi-backend/
├── app/
│   ├── main.py                ← FastAPI entry-point
│   ├── api/
│   │   └── routes/
│   │       ├── health.py      ← GET  /health
│   │       └── chat.py        ← POST /ai/chat
│   ├── core/
│   │   ├── config.py          ← Centralised configuration
│   │   └── logging.py         ← Standardized logging setup
│   ├── models/
│   │   ├── request_models.py  ← Pydantic request schemas (added 'provider')
│   │   └── response_models.py ← Pydantic response schemas
│   ├── providers/             ← NEW: Provider implementations
│   │   ├── base.py            ← Abstract base class
│   │   ├── groq_provider.py
│   │   ├── openai_provider.py
│   │   └── anthropic_provider.py
│   └── services/
│       ├── llm_gateway.py     ← NEW: Request dispatcher
│       └── llm_service.py     ← Thin service wrapper
└── .env.example               ← Updated with new provider keys
```

---

## Supported Providers

- **Groq** (Default)
- **OpenAI**
- **Anthropic**

---

## Configuration

1. Copy `.env.example` to `.env`.
2. Fill in the API keys for the providers you wish to use:

```env
GROQ_API_KEY=gsk_...
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Sample Request Payload

The `/ai/chat` endpoint now expects a `provider` field.

```json
{
  "provider": "openai",
  "prompt": "Explain vector databases in simple terms."
}
```

---

## How to Run

1. **Activate Virtual Environment**:
   - macOS/Linux: `source venv/bin/activate`
   - Windows: `venv\Scripts\activate`

2. **Run Server**:
   ```bash
   uvicorn app.main:app --reload
   ```

3. **Open Documentation**:
   Go to `http://localhost:8000/docs` to test the multi-provider gateway using Swagger UI.
