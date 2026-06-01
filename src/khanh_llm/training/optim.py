"""AdamW8bit optimizer and cosine+warmup LR scheduler setup."""

from __future__ import annotations

import math

import bitsandbytes as bnb
import torch
from torch.optim.lr_scheduler import LRScheduler

from khanh_llm.config import TrainConfig


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> bnb.optim.AdamW8bit:
    """Create an AdamW8bit optimizer with weight decay applied only to non-bias/norm params.

    Args:
        model: The model to optimise.
        cfg: TrainConfig with LR, betas, eps, weight_decay.

    Returns:
        A configured AdamW8bit optimizer.
    """
    # Separate parameters: weight decay ON for weight matrices, OFF for biases and norms
    decay_params     = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params  = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]

    param_groups = [
        {"params": decay_params,    "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = bnb.optim.AdamW8bit(
        param_groups,
        lr    = cfg.peak_lr,
        betas = tuple(cfg.lr_betas),
        eps   = cfg.lr_eps,
    )
    return optimizer


class WarmupCosineScheduler(LRScheduler):
    """Linear warmup followed by cosine decay.

    Args:
        optimizer: PyTorch optimizer.
        warmup_steps: Number of linear warmup steps (LR goes from 0 → peak_lr).
        total_steps: Total training steps (warmup + cosine decay).
        min_lr: Minimum LR at the end of cosine decay.
        last_epoch: Step to resume from (default -1 = start fresh).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 3e-5,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.min_lr       = min_lr
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> list[float]:
        step = self.last_epoch

        if step < self.warmup_steps:
            # Linear warmup: 0 → base_lr
            scale = step / max(1, self.warmup_steps)
        else:
            # Cosine decay: base_lr → min_lr
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
            # Rescale so LR never goes below min_lr
            for base_lr in self.base_lrs:
                if base_lr > 0:
                    scale = self.min_lr / base_lr + scale * (1.0 - self.min_lr / base_lr)
                    break

        return [base_lr * scale for base_lr in self.base_lrs]
