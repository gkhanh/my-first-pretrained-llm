"""Grouped-Query Attention (GQA) with SDPA backend.

Uses torch.nn.functional.scaled_dot_product_attention with is_causal=True, which:
- Automatically dispatches to the FlashAttention-2 kernel when available.
- Needs no manually-built causal mask.
- Is compatible with torch.compile.

Reference: Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from khanh_llm.model.rope import RotaryEmbedding, apply_rotary_emb


class GQAAttention(nn.Module):
    """Grouped-Query Attention.

    Args:
        hidden_dim: Model hidden dimension.
        num_heads_q: Number of query heads.
        num_heads_kv: Number of key/value heads (must divide num_heads_q).
        max_seq_len: Max sequence length (used for RoPE table precomputation).
        rope_base: RoPE theta base.
        bias: Include bias in projections (default: False).
        dropout: Attention dropout probability (default: 0.0).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads_q: int,
        num_heads_kv: int,
        max_seq_len: int = 2048,
        rope_base: float = 10000.0,
        bias: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        assert num_heads_q % num_heads_kv == 0, (
            f"num_heads_q ({num_heads_q}) must be divisible by num_heads_kv ({num_heads_kv})"
        )

        self.num_heads_q = num_heads_q
        self.num_heads_kv = num_heads_kv
        self.groups = num_heads_q // num_heads_kv  # GQA repeat factor
        self.head_dim = hidden_dim // num_heads_q
        self.dropout = dropout

        # Projections (no bias — modern standard)
        self.q_proj = nn.Linear(hidden_dim, num_heads_q  * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_dim, num_heads_kv * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_dim, num_heads_kv * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads_q * self.head_dim, hidden_dim, bias=bias)

        # RoPE
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len, base=rope_base)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, seq_len, hidden_dim).
            past_kv: Optional cached (key, value) tensors from previous steps
                     for KV-cache inference. Shape: (B, n_heads_kv, past_len, head_dim).

        Returns:
            (output, (key, value)) where output is (B, seq_len, hidden_dim)
            and (key, value) are the full accumulated KV cache.
        """
        B, T, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x).view(B, T, self.num_heads_q,  self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads_kv, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        past_len = past_kv[0].shape[2] if past_kv is not None else 0
        cos, sin = self.rope(x, seq_len=past_len + T)
        cos = cos[past_len : past_len + T]
        sin = sin[past_len : past_len + T]
        q, k = apply_rotary_emb(q, k, cos, sin)

        # Concatenate with cached KV (for inference KV-cache)
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        new_kv = (k, v)

        # Expand KV heads to match Q heads (GQA)
        if self.groups > 1:
            k = k.repeat_interleave(self.groups, dim=1)  # (B, n_heads_q, seq, head_dim)
            v = v.repeat_interleave(self.groups, dim=1)

        # Scaled dot-product attention (FlashAttention-2 backend when available)
        attn_dropout = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=attn_dropout,
            is_causal=(past_kv is None),  # causal only during training / prefill
        )

        # Merge heads and project output
        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads_q * self.head_dim)
        return self.o_proj(out), new_kv
