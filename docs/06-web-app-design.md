# Web App Design

## Architecture overview

```
┌─────────────────────┐         SSE streaming
│   Frontend          │◄────────────────────────────┐
│   Vite + React SPA  │                             │
│   localhost:5173    │──── POST /v1/generate ──►   │
└─────────────────────┘                             │
                                                    │
┌─────────────────────────────────────────────────┐ │
│   Backend (FastAPI)   localhost:8000            │ │
│                                                 │ │
│   POST /v1/generate  ────────────────────────── │─┘
│   POST /v1/chat                                 │
│   GET  /health                                  │
│                                                 │
│   ModelService                                  │
│   └── khanh_llm.inference.generator             │
└─────────────────────────────────────────────────┘
```

## API contract

### `GET /health`

**Response:**
```json
{"status": "ok", "model_loaded": true}
```

### `POST /v1/generate`

**Request:**
```json
{
  "prompt": "def fibonacci(n):",
  "max_new_tokens": 200,
  "temperature": 0.6,
  "top_k": 40,
  "top_p": 0.9,
  "repetition_penalty": 1.1,
  "stream": true
}
```

**Response (stream=false):**
```json
{
  "generated_text": "...",
  "tokens_generated": 87,
  "finish_reason": "eos"
}
```

**Response (stream=true):** Server-Sent Events (SSE)
```
data: {"token": "    ", "finish_reason": null}
data: {"token": "if", "finish_reason": null}
data: {"token": " n", "finish_reason": null}
...
data: {"token": "", "finish_reason": "eos"}
```

### `POST /v1/chat`

**Request:**
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "Explain async/await in Python."}
  ],
  "max_new_tokens": 500,
  "temperature": 0.7,
  "stream": true
}
```

**Response**: same SSE format as `/v1/generate`.

## SSE streaming format

```
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive

data: {"token": "def", "finish_reason": null}\n\n
data: {"token": " fib", "finish_reason": null}\n\n
...
data: {"token": "", "finish_reason": "eos"}\n\n
```

Client pseudocode:
```javascript
const evtSource = new EventSource('/v1/generate');
evtSource.onmessage = (e) => {
  const { token, finish_reason } = JSON.parse(e.data);
  appendToken(token);
  if (finish_reason) evtSource.close();
};
```

## Expected latency budget (RTX 5080, khanh_1b, BF16)

| Operation | Expected time |
|---|---|
| Model load (cold) | ~3–5 s |
| Time to first token | ~50–200 ms |
| Subsequent tokens (with KV cache) | ~20–50 ms/token |
| 200 tokens | ~4–10 s |

## CORS configuration

Backend allows only the Vite dev server origin during development:
```python
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
```

Set `VITE_API_URL=http://localhost:8000` in the frontend `.env`.

## Frontend scaffolding (not yet implemented)

When ready, scaffold with:
```bash
cd web/frontend
npm create vite@latest . -- --template react-ts
npm install
npm run dev
```

Key env var: `VITE_API_URL=http://localhost:8000`

Wireframe (future):
```
┌──────────────────────────────────────────────┐
│ KhanhLLM                         [settings]  │
├──────────────────────────────────────────────┤
│                                              │
│  [Chat history / code output area]           │
│                                              │
│                                              │
├──────────────────────────────────────────────┤
│  > Prompt __________________ [Generate]      │
│    Mode: [Chat] [Code] [FIM]                 │
└──────────────────────────────────────────────┘
```

## Future improvements (not in scope now)

- Request batching (vLLM-style continuous batching)
- Authentication / API key
- Rate limiting
- Model hot-swap (load different checkpoints without restart)
- vLLM integration for production serving
- Deployment config (docker-compose, nginx)
