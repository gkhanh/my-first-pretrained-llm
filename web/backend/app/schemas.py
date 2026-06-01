"""Pydantic request/response schemas for the KhanhLLM API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Input prompt text")
    max_new_tokens: int = Field(200, ge=1, le=4096)
    temperature: float = Field(0.6, ge=0.01, le=2.0)
    top_k: int = Field(40, ge=0, le=200)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0)
    stream: bool = Field(True, description="If true, returns SSE stream of tokens")


class GenerateResponse(BaseModel):
    generated_text: str
    tokens_generated: int
    finish_reason: Literal["eos", "max_tokens", "error"]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)
    max_new_tokens: int = Field(500, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.01, le=2.0)
    top_k: int = Field(40, ge=0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0)
    stream: bool = Field(True)


class StreamToken(BaseModel):
    token: str
    finish_reason: Literal["eos", "max_tokens"] | None = None
