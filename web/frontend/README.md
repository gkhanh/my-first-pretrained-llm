# KhanhLLM Frontend

**Not yet scaffolded.** This directory is a placeholder for the Vite + React SPA.

## Setup (when ready)

```bash
cd web/frontend
npm create vite@latest . -- --template react-ts
npm install
npm run dev
```

## Configuration

Create a `.env` file in this directory:

```
VITE_API_URL=http://localhost:8000
```

## Backend contract

The frontend communicates with the FastAPI backend via:

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check |
| `/v1/generate` | POST | Text generation (SSE streaming) |
| `/v1/chat` | POST | Chat completion (SSE streaming) |

See [`docs/06-web-app-design.md`](../../docs/06-web-app-design.md) for the full SSE format and request schemas.

## SSE client example

```javascript
const response = await fetch(`${import.meta.env.VITE_API_URL}/v1/generate`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ prompt, stream: true, max_new_tokens: 200 }),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  const lines = decoder.decode(value).split("\n\n");
  for (const line of lines) {
    if (line.startsWith("data: ")) {
      const { token, finish_reason } = JSON.parse(line.slice(6));
      appendToken(token);
      if (finish_reason) break;
    }
  }
}
```
