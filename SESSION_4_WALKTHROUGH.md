# Session 4 Walkthrough: Building a Provider-Agnostic LLM Gateway

This walkthrough explains how we refactored our single-provider FastAPI backend into a production-style modular LLM Gateway.

## 1. The Goal: From Hardcoded to Flexible
In Session 3, we had a direct connection:
`Route -> Service -> Groq API`

In Session 4, we evolved this to:
`Route -> Service -> Gateway -> [Provider A, B, or C]`

## 2. Updated Request Model (`app/models/request_models.py`)
We updated `ChatRequest` to include a `provider` field. This allows the client to choose which LLM to use.
```python
class ChatRequest(BaseModel):
    provider: str = Field(default="groq", description="The LLM provider to use")
    prompt: str = Field(..., description="The user prompt")
```

## 3. Provider Abstraction (`app/providers/`)
We introduced an interface to ensure all providers behave the same way.
- **`base.py`**: Defines the `BaseLLMProvider` abstract class.
- **`groq_provider.py`**: Contains the logic extracted from Session 3.
- **`openai_provider.py` & `anthropic_provider.py`**: New implementations following the same interface.

## 4. The LLM Gateway (`app/services/llm_gateway.py`)
This is the heart of the refactor. It’s a central dispatcher that:
1.  Holds instances of all available providers.
2.  Takes the `provider_name` from the request.
3.  Routes the `prompt` to the matching provider.
4.  Handles errors if a provider is missing or unsupported.

## 5. Centralized Logging (`app/core/logging.py`)
We added a standard logger to track what’s happening. You can now see:
- When the gateway selects a provider.
- Which specific model is being called.
- Clear error messages if an API key is missing.

## 6. How to Demo in Class

### Scenario A: Successful Routing
1.  Start the server: `uvicorn app.main:app --reload`
2.  Open Swagger UI (`/docs`).
3.  Send a request with `"provider": "groq"`.
4.  Check terminal logs to see the gateway in action.

### Scenario B: Error Handling (Missing Key)
1.  Send a request with `"provider": "openai"`.
2.  If `OPENAI_API_KEY` is empty in `.env`, show the student the clean 400 error response.
3.  Explain how this is better than a server crash.

### Scenario C: Unsupported Provider
1.  Send a request with `"provider": "unknown"`.
2.  Show the 400 error listing the supported options.

## Summary for Students
This architecture demonstrates **separation of concerns**. The route doesn't care *how* the LLM is called; the service doesn't care *which* provider is used. Each piece of the puzzle has one specific job.
