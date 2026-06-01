"""FastAPI application — KhanhLLM inference backend.

Endpoints:
    GET  /health          Liveness check (always returns 200, no model required)
    POST /v1/generate     Text generation with optional SSE streaming
    POST /v1/chat         Chat completion with optional SSE streaming

To run:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

CORS is configured for the Vite dev server (http://localhost:5173).
Set VITE_API_URL=http://localhost:8000 in the frontend .env.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import generate, health

app = FastAPI(
    title="KhanhLLM API",
    description="Local inference API for KhanhLLM (coding + finance LLM)",
    version="0.1.0",
)

# Allow the Vite dev server and localhost variants
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",   # common alternative
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(generate.router, prefix="/v1")
