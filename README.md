# 🚀 GenAI FastAPI Backend

A **minimal, production-style** FastAPI backend that wraps a **Groq / OpenAI-compatible** LLM API.  
Built as a teaching scaffold for the **AI Engineering** course.

Students will learn how to:

- Expose LLM functionality through a REST API
- Structure a FastAPI project with routers, services, and models
- Use Pydantic for request/response validation
- Test APIs interactively using Swagger UI

---

## 📁 Project Structure

```
genai-fastapi-backend/
│
├── README.md                  ← You are here
├── requirements.txt           ← Python dependencies
├── .env.example               ← Template for environment variables
├── .gitignore
│
└── app/
    ├── main.py                ← FastAPI application entry-point
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

---

## ✅ Prerequisites

Before you begin, make sure you have the following installed:

| Tool               | Minimum Version | Check Command          |
| ------------------- | --------------- | ---------------------- |
| **Python**          | 3.9+            | `python --version`     |
| **pip**             | (bundled)       | `pip --version`        |
| **Git** *(optional)* | any             | `git --version`        |

> 💡 On **macOS / Linux**, you may need to use `python3` and `pip3` instead of `python` and `pip`.

---

## 📥 1. Clone the Repository

```bash
git clone https://github.com/SHREY247/genai-fastapi-backend.git
cd genai-fastapi-backend
```

Or download the ZIP from GitHub and extract it manually.

---

## 🐍 2. Create a Virtual Environment

A virtual environment keeps project dependencies isolated from your system Python.

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux / Ubuntu

```bash
python3 -m venv venv
source venv/bin/activate
```

> ✅ After activation you should see `(venv)` at the beginning of your terminal prompt.

---

## 📦 3. Install Dependencies

With the virtual environment **activated**, run:

```bash
pip install -r requirements.txt
```

This installs:

- **FastAPI** — web framework
- **Uvicorn** — ASGI server
- **Pydantic** — data validation
- **Requests** — HTTP client for calling the LLM API
- **python-dotenv** — loads `.env` files

---

## 🔑 4. Configure Environment Variables

### Step A — Create the `.env` file

#### macOS / Linux / Ubuntu

```bash
cp .env.example .env
```

#### Windows (Command Prompt)

```bash
copy .env.example .env
```

#### Windows (PowerShell)

```powershell
Copy-Item .env.example .env
```

### Step B — Add your Groq API Key

Open the `.env` file in any text editor and fill in your values:

```env
GROQ_API_KEY=gsk_your_actual_api_key_here
GROQ_MODEL=llama3-8b-8192
```

> 🔗 Get a free API key at [console.groq.com](https://console.groq.com)

### Environment Variables Reference

| Variable        | Description                                   | Default                                            |
| --------------- | --------------------------------------------- | -------------------------------------------------- |
| `GROQ_API_KEY`  | Your Groq API key                             | *(required)*                                       |
| `GROQ_MODEL`    | Chat model to use                             | `llama3-8b-8192`                                   |
| `GROQ_BASE_URL` | Groq / OpenAI-compatible completions endpoint | `https://api.groq.com/openai/v1/chat/completions`  |

> ⚠️ **Never commit your `.env` file.** It is already listed in `.gitignore`.

---

## ▶️ 5. Run the Server

Make sure your virtual environment is activated, then run:

```bash
uvicorn app.main:app --reload
```

You should see output like:

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process ...
INFO:     Application startup complete.
```

The server is now running at **http://localhost:8000**.

> 💡 The `--reload` flag auto-restarts the server when you edit code — useful during development.

---

## 🧪 6. Open API Documentation (Swagger UI)

Open your browser and go to:

```
http://localhost:8000/docs
```

This opens **Swagger UI** — an interactive API playground built into FastAPI.  
You can **test every endpoint** directly from the browser without writing any code or using Postman.

### Available Endpoints

| Method | Endpoint    | Description                        |
| ------ | ----------- | ---------------------------------- |
| GET    | `/`         | Quick server health check          |
| GET    | `/health`   | Detailed health status             |
| POST   | `/ai/chat`  | Send a prompt to the LLM           |

---

## 💬 7. Example Request

### `POST /ai/chat`

Click the **POST /ai/chat** row in Swagger UI, then click **"Try it out"**.

Enter the following JSON body:

```json
{
  "prompt": "Explain embeddings in simple terms"
}
```

Click **Execute**.

### Expected Response

```json
{
  "response": "Embeddings are a way to represent words, sentences, or other data as numerical vectors..."
}
```

### Using cURL instead

#### macOS / Linux / Ubuntu

```bash
curl -X POST http://localhost:8000/ai/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain embeddings in simple terms"}'
```

#### Windows (PowerShell)

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/ai/chat `
  -ContentType "application/json" `
  -Body '{"prompt": "Explain embeddings in simple terms"}'
```

#### Windows (Command Prompt)

```bash
curl -X POST http://localhost:8000/ai/chat -H "Content-Type: application/json" -d "{\"prompt\": \"Explain embeddings in simple terms\"}"
```

---

## 🛠️ 8. Troubleshooting

### `python: command not found`

| Platform          | Fix                                                                                         |
| ----------------- | ------------------------------------------------------------------------------------------- |
| **macOS / Linux** | Use `python3` instead of `python`. Install via: `sudo apt install python3` (Ubuntu) or `brew install python` (macOS). |
| **Windows**       | Reinstall Python from [python.org](https://www.python.org/downloads/) and **check "Add Python to PATH"** during installation. |

### `pip: command not found`

Try `pip3` instead of `pip`, or run:

```bash
python -m pip install -r requirements.txt
```

### `ModuleNotFoundError: No module named 'fastapi'`

You likely forgot to activate the virtual environment or install dependencies:

```bash
# Activate first
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows

# Then install
pip install -r requirements.txt
```

### `ERROR: Address already in use` (port 8000)

Another process is using port 8000. Either stop it, or run on a different port:

```bash
uvicorn app.main:app --reload --port 8001
```

Then open `http://localhost:8001/docs` instead.

### `GROQ_API_KEY is not set`

Make sure you:

1. Created the `.env` file (not `.env.example`)
2. Added your actual API key: `GROQ_API_KEY=gsk_...`
3. **Restarted the server** after editing `.env`

---

## 🗺️ Roadmap

This repo is the **starting point** for future lessons:

- 🔹 Embeddings & vector representations
- 🔹 Vector database integration (ChromaDB / Pinecone)
- 🔹 RAG (Retrieval-Augmented Generation) pipelines
- 🔹 Streaming responses
- 🔹 Authentication & rate limiting

---

## 📄 License

This project is for **educational purposes** as part of the AI Engineering course.
