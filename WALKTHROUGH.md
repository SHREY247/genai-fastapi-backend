# Teaching Walkthrough — GenAI FastAPI Backend

> **Audience:** Final-year engineering students
> **Duration:** ~60–75 minutes
> **Pre-requisite knowledge:** Basic Python, what an API is (at a high level)
> **Goal:** By the end of this walkthrough, students will understand how to build a production-style REST API that wraps an LLM (Large Language Model) — the same pattern used inside real AI products.

---

## Part 0 — Set the Stage (5 min)

Start with the *"Why"* before showing a single line of code.

### Talking Points

- **Every AI product you use — ChatGPT, Gemini, Perplexity — works essentially the same way under the hood:**
  1. A frontend (web/mobile app) sends your message to a **backend server**.
  2. The backend server calls an **LLM API** (OpenAI, Groq, Anthropic, etc.).
  3. The LLM returns a response, and the backend sends it back to the frontend.

- **What we're building today is step 2 — the backend.** It's the bridge between a user interface and an AI model.

- **Why does the backend exist? Why not call the LLM directly from the frontend?**
  - Security: API keys would be exposed in the browser.
  - Control: You can add rate limiting, logging, caching, billing.
  - Flexibility: You can swap models, add RAG, chain prompts — all without touching the frontend.

> **Draw this on the whiteboard:**
> ```
> [ Frontend ]  --->  [ Our Backend (FastAPI) ]  --->  [ LLM API (Groq) ]
>      |                       |                              |
>   User types            Receives request,              Runs the AI model,
>   a prompt              calls the model                returns text
> ```

---

## Part 1 — The Big Picture: Project Structure (5 min)

Open the repo root in your IDE/terminal and show the folder structure:

```
genai-fastapi-backend/
│
├── README.md                  ← Setup instructions
├── requirements.txt           ← Python dependencies
├── .env.example               ← Template for secrets
├── .gitignore                 ← Files Git should ignore
│
└── app/
    ├── main.py                ← Application entry-point
    │
    ├── core/
    │   └── config.py          ← Loads environment variables
    │
    ├── api/
    │   └── routes/
    │       ├── health.py      ← GET  /health
    │       └── chat.py        ← POST /ai/chat
    │
    ├── models/
    │   ├── request_models.py  ← Pydantic request schemas
    │   └── response_models.py ← Pydantic response schemas
    │
    └── services/
        └── llm_service.py     ← LLM API call logic
```

### Talking Points

- **This is not a random layout.** This is the standard way production FastAPI applications are structured. If you join a company building AI products, you'll see this exact pattern.
- **Separation of concerns:** Each folder has one job:
  - `core/` — configuration & settings
  - `api/routes/` — HTTP endpoint definitions (thin controllers)
  - `models/` — data shapes (what goes in, what comes out)
  - `services/` — business logic (the actual work)
- **Why does this matter?** When your codebase grows to 50+ files, this structure is what keeps it manageable. Imagine cramming everything into a single `main.py` — that's a nightmare to debug, test, or hand off to a teammate.

---

## Part 2 — Configuration: `core/config.py` (5 min)

> **Open file:** `app/core/config.py`

```python
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_BASE_URL: str = os.getenv(
    "GROQ_BASE_URL",
    "https://api.groq.com/openai/v1/chat/completions",
)
```

### Talking Points

- **`load_dotenv()`** reads the `.env` file and loads its values into the operating system's environment variables. This means your code never contains hardcoded secrets.
- **`os.getenv("KEY", "default")`** reads an environment variable. The second argument is a fallback default.
- **Why not just write `GROQ_API_KEY = "gsk_abc123..."` directly?**
  - Security: If you push that to GitHub, your key is public. Bots scrape GitHub for exposed API keys — your account could be compromised in minutes.
  - Flexibility: Different environments (dev, staging, production) use different keys. Environment variables let you change config without changing code.

> **Show students:** the `.env.example` file — this is the *template*. Students copy it to `.env` and fill in their own key.

> **Show students:** the `.gitignore` — notice `.env` is listed there. Git will never track it.

### Key Concept: Configuration should live *outside* your code.

---

## Part 3 — Data Contracts: `models/` (10 min)

This is where we define *what data looks like* going in and coming out of our API.

### 3a. Request Model

> **Open file:** `app/models/request_models.py`

```python
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
        description="The user prompt to send to the LLM.",
        examples=["Explain what an API is in simple terms."],
    )
```

### Talking Points

- **Pydantic** is a data validation library. When you define a model like this, FastAPI automatically:
  1. **Validates** incoming JSON — if `prompt` is missing or empty, the API returns a 422 error *before your code even runs*.
  2. **Converts types** — if someone sends a number, Pydantic coerces it to a string.
  3. **Documents** — the `description` and `examples` fields show up in Swagger UI automatically.
- **`...` (Ellipsis)** means the field is required. There's no default — the client *must* provide it.
- **`min_length=1`** prevents empty strings. Without this, someone could send `""` and waste an API call.

### 3b. Response Models

> **Open file:** `app/models/response_models.py`

```python
class ChatResponse(BaseModel):
    response: str = Field(
        ...,
        description="The assistant's reply from the LLM.",
    )

class HealthResponse(BaseModel):
    status: str = Field(..., description="Current server status.", examples=["ok"])
    message: str = Field(..., description="Human-readable status message.")
```

### Talking Points

- **Why define response models?** Three reasons:
  1. **Contract:** Frontend teams can see exactly what the API will return — they don't need to guess.
  2. **Consistency:** Every response has the same shape. No surprises.
  3. **Documentation:** Swagger UI shows the exact response schema.
- This is an industry best practice called **"API-first design"** or **"schema-driven development."**

### Interactive Moment

> Ask students: *"What would happen if the `ask_llm` function returned a dictionary instead of a string? Would Pydantic catch it?"*
> Answer: Yes — Pydantic would raise a validation error because `response` expects a `str`.

---

## Part 4 — The Entry Point: `main.py` (5 min)

> **Open file:** `app/main.py`

```python
from fastapi import FastAPI
from app.api.routes import health, chat

app = FastAPI(
    title="GenAI FastAPI Backend",
    description="A simple FastAPI service that wraps an LLM API (Groq)...",
    version="1.0.0",
)

@app.get("/", tags=["Root"])
def root():
    return {"message": "GenAI FastAPI Backend is running"}

app.include_router(health.router, tags=["Health"])
app.include_router(chat.router, tags=["Chat"])
```

### Talking Points

- **`FastAPI()`** creates the application instance. The `title`, `description`, and `version` appear in Swagger UI's header.
- **The root route (`/`)** is a sanity check. When you deploy, you can hit `http://yourserver.com/` to confirm it's alive. This takes 2 lines and saves hours of debugging.
- **`include_router()`** — this is how FastAPI lets you split your API across multiple files. Each router is a self-contained module. This is the backbone of the separation of concerns.
  - Without routers, your `main.py` would have every endpoint, every import, every piece of logic in a single file. With 50 endpoints, that file would be 2000+ lines.
- **`tags=["Health"]`** groups endpoints in Swagger UI. It's purely organizational but makes the docs readable.

### Key Concept: `main.py` is the orchestrator — it doesn't do work itself, it wires things together.

---

## Part 5 — Route Handlers: `api/routes/` (10 min)

### 5a. Health Check

> **Open file:** `app/api/routes/health.py`

```python
from fastapi import APIRouter
from app.models.response_models import HealthResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        message="GenAI FastAPI backend is running.",
    )
```

### Talking Points

- **Every production API has a health endpoint.** AWS load balancers, Kubernetes, Docker, monitoring tools — they all ping `/health` to decide if your server is alive. If it stops responding, traffic is routed elsewhere.
- **`response_model=HealthResponse`** tells FastAPI to validate the *output* too, and to generate accurate Swagger docs.
- Notice how thin this handler is — 3 lines of actual logic. That's intentional.

### 5b. Chat Endpoint

> **Open file:** `app/api/routes/chat.py`

```python
from fastapi import APIRouter, HTTPException
from app.models.request_models import ChatRequest
from app.models.response_models import ChatResponse
from app.services.llm_service import ask_llm

router = APIRouter()

@router.post("/ai/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        reply = ask_llm(prompt=request.prompt)
        return ChatResponse(response=reply)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {exc}",
        )
```

### Talking Points

- **This is a POST endpoint** because we're *sending* data (the prompt) to the server. GET is for reading, POST is for creating/sending.
- **`request: ChatRequest`** — FastAPI sees the Pydantic model as a type hint and automatically:
  1. Reads the JSON body from the request.
  2. Validates it against the `ChatRequest` schema.
  3. Returns a 422 error if validation fails.
  4. Otherwise, gives you a clean, typed Python object.
- **Notice the handler is thin.** It does three things:
  1. Receive validated input.
  2. Call the service (`ask_llm`).
  3. Return a typed response.
  - All the actual work (calling the LLM, handling errors) is in the service layer. This is called the **"thin controller"** pattern.
- **Error handling pattern:**
  - `HTTPException` from the service layer? Re-raise it as-is (it already has the right status code).
  - Any other exception? Catch it and return a clean 500 error. Never let raw stack traces leak to the client — that's a security risk.

### Interactive Moment

> Ask students: *"What HTTP status code would you get if you send `{}` (no prompt field) to `/ai/chat`?"*
> Answer: 422 Unprocessable Entity — Pydantic catches it before `chat()` even runs.

---

## Part 6 — The Service Layer: `llm_service.py` (15 min)

> **This is the heart of the application.** Spend the most time here.

> **Open file:** `app/services/llm_service.py`

Walk through it section by section:

### Section 1: Guard Clause

```python
if not GROQ_API_KEY:
    raise HTTPException(
        status_code=400,
        detail="GROQ_API_KEY is not set. Please add it to your .env file..."
    )
```

- **Fail fast.** If the key isn't configured, there's no point making the API call. Return a clear error immediately.
- This saves the student 30 minutes of debugging cryptic "401 Unauthorized" errors from Groq.

### Section 2: Building the Request

```python
headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json",
}

payload = {
    "model": GROQ_MODEL,
    "messages": [
        {"role": "user", "content": prompt},
    ],
}
```

- **This is the OpenAI-compatible Chat Completions format.** Groq, OpenAI, Anthropic, Together AI — they all use this same structure. Learn it once, use it everywhere.
- **`Authorization: Bearer <token>`** — this is the standard way APIs authenticate. The word "Bearer" is part of the HTTP spec (RFC 6750).
- **`messages` is an array** because LLMs are conversational. In a more advanced app, you'd include the full conversation history here. For now, we send a single user message.
- **`role: "user"`** tells the model this is a human's message. Other roles: `"system"` (instructions) and `"assistant"` (previous AI responses).

### Section 3: Making the HTTP Call

```python
response = requests.post(
    GROQ_BASE_URL,
    headers=headers,
    json=payload,
    timeout=30,
)
```

- **`requests.post()`** — this is Python's most popular HTTP library making an outbound API call. Our backend is now acting as a *client* to the Groq API.
- **`json=payload`** automatically serializes the dict to JSON and sets the correct Content-Type header.
- **`timeout=30`** — critical in production. Without a timeout, if the LLM API hangs, your server hangs too. Set a reasonable timeout and handle it gracefully.

### Section 4: Error Handling

```python
except requests.Timeout:
    raise HTTPException(status_code=504, detail="LLM API request timed out...")
except requests.ConnectionError as exc:
    raise HTTPException(status_code=502, detail=f"Could not connect to the LLM API...")
except requests.RequestException as exc:
    raise HTTPException(status_code=502, detail=f"LLM API request failed: {exc}")
```

- **Three layers of error handling**, each with a specific HTTP status code:
  - **504 Gateway Timeout** — the upstream took too long.
  - **502 Bad Gateway** — we couldn't reach the upstream at all.
  - **502 (generic)** — something else went wrong.
- **Why different status codes?** So the frontend (and monitoring tools) can distinguish between "the model is slow" vs "the network is down" vs "something unexpected happened."

### Section 5: Parsing the Response

```python
data = response.json()

try:
    assistant_message: str = data["choices"][0]["message"]["content"]
except (KeyError, IndexError) as exc:
    raise HTTPException(
        status_code=502,
        detail=f"Unexpected response structure from LLM API: {exc}",
    )

return assistant_message
```

- **The LLM returns a deeply nested JSON structure.** We extract just the text we need: `choices[0].message.content`.
- **Why the try/except around parsing?** APIs can change. If Groq modifies their response format tomorrow, we get a clear error instead of a cryptic `KeyError` crash.
- **This is defensive programming** — assume external systems can behave unexpectedly.

### Key Concept: The service layer is where business logic lives. Routes handle HTTP; services handle *work*.

---

## Part 7 — Live Demo (10 min)

Now bring it all to life. Run the server in front of the students.

### Step 1: Start the server

```bash
uvicorn app.main:app --reload
```

> Explain: `app.main` = the file path (`app/main.py`), `:app` = the FastAPI instance inside it, `--reload` = auto-restart on code changes.

### Step 2: Open Swagger UI

Open `http://localhost:8000/docs` in the browser.

> Show students:
> - The title and description from `main.py` appear at the top.
> - Endpoints are grouped by tags (Root, Health, Chat).
> - Click on any endpoint — the request/response schemas from Pydantic models are shown automatically.

### Step 3: Test the health endpoint

- Click **GET /health** → **Try it out** → **Execute**
- Show the 200 response: `{"status": "ok", "message": "GenAI FastAPI backend is running."}`

### Step 4: Test the chat endpoint

- Click **POST /ai/chat** → **Try it out**
- Enter: `{"prompt": "Explain what an API is in one sentence."}`
- Click **Execute**
- Show the response from the LLM.

### Step 5: Trigger an error

- Send an empty prompt: `{"prompt": ""}`
- Show the 422 validation error — Pydantic caught it.

### Step 6: Show the request flow

> Trace the flow out loud:
> 1. **Swagger UI** sends a POST request to `/ai/chat` with `{"prompt": "..."}`.
> 2. **FastAPI** receives it, validates it against `ChatRequest` (Pydantic).
> 3. **`chat()` route handler** calls `ask_llm(prompt)`.
> 4. **`ask_llm()`** builds the payload, calls Groq's API via `requests.post()`.
> 5. **Groq's LLM** processes the prompt and returns a response.
> 6. **`ask_llm()`** extracts the text and returns it.
> 7. **`chat()`** wraps it in a `ChatResponse` and returns it.
> 8. **FastAPI** serializes it to JSON and sends it back to Swagger UI.

---

## Part 8 — Architecture Recap & Real-World Connections (5 min)

### The Patterns You Just Learned

| Pattern | Where You Saw It | Why It Matters |
|---|---|---|
| **Separation of Concerns** | Routes vs. Services vs. Models | Each file has one job. Easy to find, change, and test. |
| **Thin Controllers** | `chat.py` is 15 lines | Route handlers validate & delegate — they don't do heavy lifting. |
| **Schema-Driven Design** | Pydantic models | Input/output contracts are explicit. Swagger docs are auto-generated. |
| **Environment-Based Config** | `.env` + `config.py` | Secrets stay out of code. Different configs for dev/prod. |
| **Defensive Error Handling** | `llm_service.py` | Every failure mode has a specific, informative error. |
| **Health Checks** | `health.py` | Industry standard for monitoring and deployment. |

### Where This Goes Next

- **Add a system prompt** → give the LLM a personality or role (e.g., "You are a helpful tutor").
- **Conversation memory** → pass the full message history instead of a single prompt.
- **Embeddings + RAG** → retrieve relevant documents before asking the LLM (Retrieval-Augmented Generation).
- **Streaming** → send the response token-by-token instead of waiting for the full reply.
- **Authentication** → protect your endpoints with API keys or JWT tokens.
- **Deployment** → containerize with Docker, deploy to AWS/GCP/Railway.

---

## Part 9 — Student Exercise Ideas

### Exercise 1: Add a System Prompt (Easy — 10 min)
> Modify `llm_service.py` to accept an optional `system_prompt` parameter. Add a `{"role": "system", "content": system_prompt}` message before the user message in the `messages` array. Update the Pydantic model to include this optional field.

### Exercise 2: Add a `/ai/summarize` Endpoint (Medium — 20 min)
> Create a new route that accepts a block of text and returns a summary. Reuse the `ask_llm` service but prepend "Summarize the following text:" to the prompt.

### Exercise 3: Add Request Logging (Medium — 15 min)
> Use Python's `logging` module to log every incoming request (prompt, timestamp) and every response (first 100 chars, response time). This is how production systems track usage.

### Exercise 4: Rate Limiting (Hard — 30 min)
> Add a simple in-memory rate limiter that allows a maximum of 10 requests per minute per IP address. Return 429 Too Many Requests if the limit is exceeded.

---

## Quick Reference — HTTP Status Codes Used in This Project

| Code | Name | When We Use It |
|---|---|---|
| **200** | OK | Successful response |
| **400** | Bad Request | API key not configured |
| **422** | Unprocessable Entity | Pydantic validation failed (auto) |
| **500** | Internal Server Error | Unexpected crash in our code |
| **502** | Bad Gateway | LLM API returned an error |
| **504** | Gateway Timeout | LLM API took too long |

---

## Quick Reference — Files at a Glance

| File | Lines | Purpose |
|---|---|---|
| `app/main.py` | 34 | Creates FastAPI app, registers routers |
| `app/core/config.py` | 25 | Loads `.env` variables |
| `app/api/routes/health.py` | 30 | GET /health endpoint |
| `app/api/routes/chat.py` | 46 | POST /ai/chat endpoint |
| `app/models/request_models.py` | 22 | ChatRequest Pydantic model |
| `app/models/response_models.py` | 37 | ChatResponse, HealthResponse models |
| `app/services/llm_service.py` | 111 | Core logic — calls the LLM API |
| `requirements.txt` | 5 | Python packages |
| `.env.example` | 16 | Template for environment variables |
| `.gitignore` | 36 | Files excluded from Git |
