"""Generate and chat endpoints — stubs pending model_service implementation.

See docs/06-web-app-design.md for the full API contract and SSE format.
"""

from fastapi import APIRouter

from app.schemas import ChatRequest, GenerateRequest, GenerateResponse

router = APIRouter(tags=["generate"])


@router.post("/generate")
async def generate(request: GenerateRequest) -> GenerateResponse:
    """Text generation with optional SSE streaming.

    When request.stream=True, returns a Server-Sent Events stream.
    Each event: data: {"token": "...", "finish_reason": null|"eos"|"max_tokens"}

    Not yet implemented — see docs/06-web-app-design.md.
    """
    raise NotImplementedError("see docs/06-web-app-design.md")


@router.post("/chat")
async def chat(request: ChatRequest) -> GenerateResponse:
    """Chat completion with optional SSE streaming.

    Applies the ChatML template to messages before generation.
    Not yet implemented — see docs/06-web-app-design.md.
    """
    raise NotImplementedError("see docs/06-web-app-design.md")
