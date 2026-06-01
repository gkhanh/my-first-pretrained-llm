"""SwiGLU Feed-Forward Network.

Replaces the GELU FFN used in the legacy model:
    FFN(x) = (xW₁ ⊙ SiLU(xW₃)) W₂

Properties:
- Used by Llama, Mistral, PaLM, Qwen, etc.
- FFN dim is ~2.7× hidden (not 4×) to keep parameter count comparable.
- No bias terms (standard in modern LLMs).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block.

    Args:
        hidden_dim: Model hidden dimension (input/output size).
        ffn_dim: Intermediate dimension. Should be ~2.7× hidden_dim for SwiGLU
                 to match the parameter count of a 4× GELU FFN.
        bias: Whether to include bias in linear layers (default: False, per modern practice).
    """

    def __init__(self, hidden_dim: int, ffn_dim: int, bias: bool = False) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=bias)  # W₃ (gating)
        self.up_proj   = nn.Linear(hidden_dim, ffn_dim, bias=bias)  # W₁ (value)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=bias)  # W₂ (output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: (x W₁ ⊙ SiLU(x W₃)) W₂
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
