"""Chat template formatting for SFT datasets.

Supports three formats:
- ChatML (default for KhanhLLM, OpenHermes, etc.)
- Llama-3-instruct
- Qwen-instruct
"""

from __future__ import annotations

from typing import Literal

Message = dict[str, str]  # {"role": "user"|"assistant"|"system", "content": str}
TemplateFormat = Literal["chatml", "llama3", "qwen"]


def format_messages(messages: list[Message], fmt: TemplateFormat = "chatml") -> str:
    """Format a list of chat messages into a training string.

    Args:
        messages: List of {"role": ..., "content": ...} dicts.
        fmt: Template format to use.

    Returns:
        A single string ready for tokenization.
    """
    if fmt == "chatml":
        return _format_chatml(messages)
    elif fmt == "llama3":
        return _format_llama3(messages)
    elif fmt == "qwen":
        return _format_qwen(messages)
    else:
        raise ValueError(f"Unknown template format: {fmt!r}. Choose from: chatml, llama3, qwen")


def _format_chatml(messages: list[Message]) -> str:
    """ChatML format:
        <|im_start|>role\ncontent<|im_end|>\n
    """
    parts = []
    for msg in messages:
        parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
    # Add the trailing generation prompt for the assistant
    if messages and messages[-1]["role"] != "assistant":
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def _format_llama3(messages: list[Message]) -> str:
    """Llama 3 instruct format using <|begin_of_text|> and role headers."""
    parts = ["<|begin_of_text|>"]
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")
    if messages and messages[-1]["role"] != "assistant":
        parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


def _format_qwen(messages: list[Message]) -> str:
    """Qwen instruct format (same as ChatML but with Qwen special tokens)."""
    # Qwen 2.x uses the same ChatML format by default
    return _format_chatml(messages)


def get_response_template(fmt: TemplateFormat = "chatml") -> str:
    """Return the string that marks the start of the assistant response.

    Used for loss masking: only compute loss on tokens AFTER this marker.
    """
    templates = {
        "chatml":  "<|im_start|>assistant\n",
        "llama3":  "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "qwen":    "<|im_start|>assistant\n",
    }
    return templates[fmt]
