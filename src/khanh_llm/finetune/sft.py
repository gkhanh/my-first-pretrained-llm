"""Supervised Fine-Tuning (SFT) loop.

Works for both:
- Mode A: KhanhLLM checkpoints (via khanh_llm.model.transformer.KhanhLLM)
- Mode B: External HuggingFace causal-LM (via transformers.AutoModelForCausalLM + QLoRA)

Uses BF16 autocast and gradient accumulation — same recipe as pretraining.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def run_sft(
    model: nn.Module,
    tokenizer: Any,
    dataloader: DataLoader,
    run_dir: str | Path,
    num_epochs: int = 2,
    peak_lr: float = 2e-5,
    min_lr: float = 2e-6,
    warmup_steps: int = 100,
    gradient_accumulation_steps: int = 8,
    max_grad_norm: float = 1.0,
    use_wandb: bool = False,
    pad_token_id: int | None = None,
) -> None:
    """Run a supervised fine-tuning loop.

    Computes loss ONLY on assistant response tokens (prompt tokens are masked).
    Loss masking is applied by setting label = -100 for non-assistant tokens.

    Args:
        model: A KhanhLLM or PEFT-wrapped model (already on the desired device).
        tokenizer: HuggingFace tokenizer with EOS/PAD tokens set.
        dataloader: DataLoader yielding {"input_ids", "labels", "attention_mask"} dicts.
            Labels should have -100 for tokens where loss should NOT be computed.
        run_dir: Directory to save SFT checkpoints and logs.
        num_epochs: Number of passes over the SFT dataset.
        peak_lr: Peak learning rate (much lower than pretrain: 1e-5 to 5e-5).
        min_lr: Minimum LR at end of cosine decay.
        warmup_steps: Linear warmup steps.
        gradient_accumulation_steps: Grad accum factor.
        max_grad_norm: Gradient clipping norm.
        use_wandb: Enable W&B logging.
        pad_token_id: ID to ignore in cross-entropy loss (default: tokenizer.pad_token_id).
    """
    from khanh_llm.training.logging import TrainingLogger
    from khanh_llm.training.optim import WarmupCosineScheduler

    run_dir = Path(run_dir)
    device = next(model.parameters()).device

    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "pad_token_id", -100) or -100

    # Optimizer: use regular AdamW for SFT (model is already small or QLoRA)
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(), lr=peak_lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01
        )
    except ImportError:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=peak_lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01
        )

    total_steps = (len(dataloader) // gradient_accumulation_steps) * num_epochs
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps=warmup_steps, total_steps=total_steps, min_lr=min_lr
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    logger = TrainingLogger(run_dir, use_wandb=use_wandb, run_name=run_dir.name)

    step = 0
    model.train()

    for epoch in range(num_epochs):
        print(f"\n[SFT] Epoch {epoch + 1}/{num_epochs}")
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # KhanhLLM returns (logits, aux_loss, kvs); HF models return a ModelOutput
                output = model(input_ids)
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output.logits

                loss = criterion(
                    logits.view(-1, logits.size(-1)), labels.view(-1)
                ) / gradient_accumulation_steps

            loss.backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm).item()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                logger.log(
                    step=step, tokens=step * input_ids.numel() * gradient_accumulation_steps,
                    loss=loss.item() * gradient_accumulation_steps,
                    lr=scheduler.get_last_lr()[0], tokens_per_sec=0.0, grad_norm=grad_norm,
                )

        # Save epoch checkpoint
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        ckpt_path = run_dir / "checkpoints" / f"sft_epoch_{epoch+1}.pt"
        torch.save({"model_state_dict": model.state_dict(), "step": step, "epoch": epoch+1}, ckpt_path)
        print(f"[SFT] Checkpoint saved: {ckpt_path}")

    logger.close()
    print("[SFT] Training complete.")
