# KhanhLLM Web Interface

## Architecture

```
web/
├── backend/     FastAPI app (Python) — wraps khanh_llm.inference.generator
└── frontend/    Vite + React SPA — SSE streaming chat UI (not yet scaffolded)
```

See [`docs/06-web-app-design.md`](../docs/06-web-app-design.md) for the full API contract and design.

## Backend

```bash
cd web/backend
pip install -e .
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Smoke test:
curl http://localhost:8000/health
# → {"status": "ok", "model_loaded": false}
```

## Frontend (not yet implemented)

When ready:

```bash
cd web/frontend
npm create vite@latest . -- --template react-ts
npm install
npm run dev
```

Set env var: `VITE_API_URL=http://localhost:8000`

The frontend talks to the backend via:
- `GET  /health`
- `POST /v1/generate` (SSE streaming)
- `POST /v1/chat`     (SSE streaming)
