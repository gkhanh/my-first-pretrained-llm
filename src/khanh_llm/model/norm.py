"""RMSNorm — Root Mean Square Layer Normalisation.

Replaces nn.LayerNorm throughout KhanhLLM:
- No mean-centering (only RMS scaling).
- Faster than LayerNorm; used by Llama, Mistral, Qwen, DeepSeek, etc.
- Applied in pre-norm style: x = x + sublayer(norm(x))
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square normalisation.

    Args:
        dim: Feature dimension to normalise.
        eps: Small constant for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32 for the norm computation, then back to input dtype.
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
