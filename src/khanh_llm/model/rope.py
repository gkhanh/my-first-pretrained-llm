"""Rotary Position Embeddings (RoPE).

Encodes relative positions directly into the attention QK dot product.
Used by Llama, Mistral, Qwen, Falcon, etc.

Key properties:
- No learned parameters.
- Supports length extrapolation (via RoPE scaling at inference time).
- Applied only to Q and K, not V.

Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Precomputes RoPE cos/sin tables up to max_seq_len.

    Args:
        head_dim: Dimension of each attention head.
        max_seq_len: Maximum sequence length to precompute tables for.
        base: RoPE theta base (default 10000, higher = longer effective context).
    """

    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        self.head_dim = head_dim

        # Precompute inverse frequencies: θᵢ = 1 / (base^(2i/d))
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin tables
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)           # (seq_len, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)         # (seq_len, head_dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin tables for the given sequence length."""
        if seq_len is None:
            seq_len = x.shape[1]
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        cos = self.cos_cached[:seq_len].to(x.dtype)
        sin = self.sin_cached[:seq_len].to(x.dtype)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension to implement RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to Q and K tensors.

    Args:
        q: Query tensor of shape (B, n_heads_q, seq_len, head_dim).
        k: Key tensor of shape (B, n_heads_kv, seq_len, head_dim).
        cos: Cosine table (seq_len, head_dim).
        sin: Sine table (seq_len, head_dim).

    Returns:
        Tuple of (q_rotated, k_rotated) with the same shapes as inputs.
    """
    # Broadcast cos/sin to (1, 1, seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot
