"""Sampling utilities: top-k, top-p (nucleus), repetition penalty."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def top_k_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Zero out all logits except the top-k."""
    if top_k <= 0:
        return logits
    k = min(top_k, logits.size(-1))
    threshold, _ = torch.topk(logits, k)
    logits[logits < threshold[:, -1:]] = float("-inf")
    return logits


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus (top-p) filtering: keep the smallest set whose cumulative prob ≥ top_p."""
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens once cumulative prob exceeds top_p (shift by 1 to keep the crossing token)
    sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
    sorted_logits[sorted_mask] = float("-inf")

    # Scatter back to original ordering
    logits.scatter_(1, sorted_idx, sorted_logits)
    return logits


def repetition_penalty_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    penalty: float = 1.0,
    window: int = 64,
) -> torch.Tensor:
    """Vectorised repetition penalty (avoids the Python-set loop in the legacy code).

    Args:
        logits: (1, vocab_size) logits for the next token.
        input_ids: (1, seq_len) token IDs seen so far.
        penalty: Values > 1.0 reduce probability of seen tokens.
        window: How many recent tokens to consider.

    Returns:
        Modified logits tensor (in-place).
    """
    if penalty == 1.0:
        return logits
    recent = input_ids[0, -window:]  # (window,)
    score = logits[0].gather(0, recent)
    score = torch.where(score < 0, score * penalty, score / penalty)
    logits[0].scatter_(0, recent, score)
    return logits
