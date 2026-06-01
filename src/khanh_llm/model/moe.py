"""MoE (Mixture of Experts) FFN with correct Switch Transformer balance loss.

This module is NOT used in the default khanh_1b config (dense model).
Set use_moe=True in ModelConfig for a future khanh_700m experiment.

Key fix over the legacy implementation:
- Balance loss uses Switch Transformer formulation (differentiable w.r.t. gate.weight).
- fraction_tokens computed via scatter/one-hot (not argmax — non-differentiable).
- torch.histc replaced with F.one_hot().float().sum(0) (compile-friendly).

Reference: Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models with Simple
and Efficient Sparsity"
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from khanh_llm.model.ffn import SwiGLU


class MoELayer(nn.Module):
    """Switch-Transformer-style sparse MoE FFN.

    Args:
        hidden_dim: Model hidden dimension.
        ffn_dim: Intermediate size per expert.
        num_experts: Total number of experts (E).
        num_experts_active: Experts activated per token (K). Usually 1 or 2.
        aux_loss_weight: Weight for the load-balancing auxiliary loss.
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        num_experts: int = 8,
        num_experts_active: int = 2,
        aux_loss_weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_active = num_experts_active
        self.aux_loss_weight = aux_loss_weight

        # Router
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

        # Experts (SwiGLU, no bias)
        self.experts = nn.ModuleList([
            SwiGLU(hidden_dim, ffn_dim, bias=False)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the MoE layer.

        Args:
            x: Input of shape (B, seq_len, hidden_dim).

        Returns:
            (output, aux_loss) where output has the same shape as x.
        """
        B, T, D = x.shape
        N = B * T
        flat_x = x.view(N, D)  # (N, D)

        # ── Router ──────────────────────────────────────────────────────────
        routing_logits = self.gate(flat_x)           # (N, E)
        routing_probs  = F.softmax(routing_logits, dim=-1)  # (N, E)

        # ── Balance loss (Switch Transformer formulation) ────────────────────
        # prob_mass: average router probability → keeps gradient
        prob_mass = routing_probs.mean(dim=0)        # (E,)

        # fraction_tokens: fraction of tokens routed to each expert via top-k.
        # Detached — this is a count, not a differentiable quantity.
        _, top_indices = torch.topk(routing_probs, self.num_experts_active, dim=-1)  # (N, K)
        one_hot = F.one_hot(top_indices, num_classes=self.num_experts).float()       # (N, K, E)
        fraction_tokens = one_hot.sum(dim=1).mean(dim=0).detach()                   # (E,)

        # aux_loss = E × Σ (fᵢ × pᵢ)  — minimised when routing is uniform
        aux_loss = self.num_experts * (fraction_tokens * prob_mass).sum()

        # ── Top-K dispatch ──────────────────────────────────────────────────
        routing_weights, selected_experts = torch.topk(routing_probs, self.num_experts_active, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)  # re-normalise

        # ── Compute expert outputs ───────────────────────────────────────────
        output = torch.zeros_like(flat_x)

        for expert_idx in range(self.num_experts):
            expert_mask = (selected_experts == expert_idx)          # (N, K) bool
            batch_idx, k_idx = expert_mask.nonzero(as_tuple=True)

            if batch_idx.numel() == 0:
                continue

            tokens_for_expert = flat_x[batch_idx]                  # (n_selected, D)
            expert_out = self.experts[expert_idx](tokens_for_expert)
            weights = routing_weights[batch_idx, k_idx].unsqueeze(-1)  # (n_selected, 1)
            output.index_add_(0, batch_idx, expert_out * weights)

        return output.view(B, T, D), aux_loss
