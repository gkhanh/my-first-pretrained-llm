"""Checkpoint save/load with EMA support.

Checkpoint layout (single .pt file):
    {
        "step":               int,
        "tokens_processed":   int,
        "model_state_dict":   dict,         # weights-only, stripped of _orig_mod. prefix
        "optimizer_state_dict": dict,
        "scheduler_state_dict": dict,
        "ema_state_dict":     dict | None,
        "rng_state":          dict,         # torch + numpy + cuda RNG states
        "data_cursor":        dict,         # {shard_idx, offset} for exact resume
        "config":             dict,         # ModelConfig + TrainConfig as dicts
    }
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def _strip_compile_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Strip the '_orig_mod.' prefix that torch.compile adds to state_dict keys."""
    return {
        (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
        for k, v in state_dict.items()
    }


class EMA:
    """Exponential Moving Average of model weights for smoother inference checkpoints.

    Args:
        model: The model whose weights to track.
        decay: EMA decay rate (e.g. 0.999).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self._register(model)

    def _register(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                clean = name[len("_orig_mod."):] if name.startswith("_orig_mod.") else name
                self.shadow[clean] = param.data.clone().float()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update EMA weights after each optimizer step."""
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            clean = name[len("_orig_mod."):] if name.startswith("_orig_mod.") else name
            if clean in self.shadow:
                self.shadow[clean] = (
                    self.decay * self.shadow[clean] + (1.0 - self.decay) * param.data.float()
                )

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.clone().float() for k, v in state_dict.items()}

    def copy_to(self, model: nn.Module) -> None:
        """Copy EMA weights into a model (for inference)."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name].to(param.device))


def save_checkpoint(
    run_dir: str | Path,
    step: int,
    loss: float,
    tokens_processed: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    ema: EMA | None,
    data_cursor: dict,
    config_dict: dict,
) -> Path:
    """Save a full training checkpoint.

    Naming: trained_khanh_gpt_step{step:08d}_loss{loss:.3f}.pt

    Args:
        run_dir: Root directory for this run (e.g. "runs/khanh_1b/").
        step: Current training step.
        loss: Current training loss (for filename).
        tokens_processed: Total tokens seen so far.
        model: The model (may be torch.compile-wrapped).
        optimizer: Optimizer instance.
        scheduler: LR scheduler instance.
        ema: EMA tracker (or None if not used).
        data_cursor: {"shard_idx": int, "offset": int} for exact resume.
        config_dict: Serialized ModelConfig + TrainConfig.

    Returns:
        Path to the saved checkpoint file.
    """
    ckpt_dir = Path(run_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    fname = f"trained_khanh_gpt_step{step:08d}_loss{loss:.3f}.pt"
    ckpt_path = ckpt_dir / fname

    # Collect RNG states for exact reproducibility
    rng_state = {
        "torch": torch.get_rng_state(),
        "cuda":  torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

    state = {
        "step":                  step,
        "tokens_processed":      tokens_processed,
        "model_state_dict":      _strip_compile_prefix(model.state_dict()),
        "optimizer_state_dict":  optimizer.state_dict(),
        "scheduler_state_dict":  scheduler.state_dict(),
        "ema_state_dict":        ema.state_dict() if ema is not None else None,
        "rng_state":             rng_state,
        "data_cursor":           data_cursor,
        "config":                config_dict,
    }

    torch.save(state, ckpt_path)

    # Update the 'latest' symlink
    latest = ckpt_dir / "trained_khanh_gpt_latest.pt"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(fname)

    # Save EMA weights separately (weights only — smaller, safe for inference)
    if ema is not None:
        ema_path = ckpt_dir / "trained_khanh_gpt_ema.pt"
        torch.save({"ema_state_dict": ema.state_dict(), "step": step}, ema_path)

    return ckpt_path


def load_checkpoint(
    ckpt_path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    ema: EMA | None = None,
    device: torch.device | str = "cpu",
) -> dict:
    """Load a training checkpoint.

    Args:
        ckpt_path: Path to the .pt checkpoint file.
        model: Model to load weights into.
        optimizer: Optimizer to restore state (optional).
        scheduler: Scheduler to restore state (optional).
        ema: EMA tracker to restore (optional).
        device: Device to map tensors to.

    Returns:
        The full checkpoint dict (for accessing step, tokens_processed, data_cursor, etc.).
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model_sd = _strip_compile_prefix(ckpt["model_state_dict"])
    model.load_state_dict(model_sd, strict=True)

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    if ema is not None and ckpt.get("ema_state_dict") is not None:
        ema.load_state_dict(ckpt["ema_state_dict"])

    if "rng_state" in ckpt:
        torch.set_rng_state(ckpt["rng_state"]["torch"])
        if ckpt["rng_state"].get("cuda") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt["rng_state"]["cuda"])

    return ckpt


def prune_old_checkpoints(run_dir: str | Path, keep_last_n: int = 10) -> None:
    """Delete old checkpoints beyond the rolling window, preserving every 10th.

    Args:
        run_dir: Run directory (contains "checkpoints/" subdirectory).
        keep_last_n: Number of most-recent checkpoints to keep unconditionally.
    """
    ckpt_dir = Path(run_dir) / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("trained_khanh_gpt_step*.pt"))

    if len(ckpts) <= keep_last_n:
        return

    # Keep: last keep_last_n + every 10th (milestone)
    to_keep = set(ckpts[-keep_last_n:])
    for i, ckpt in enumerate(ckpts):
        if i % 10 == 0:
            to_keep.add(ckpt)

    for ckpt in ckpts:
        if ckpt not in to_keep and ckpt.exists():
            ckpt.unlink()
