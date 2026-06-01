"""Main training loop for KhanhLLM pretraining.

Orchestrates: data loading → forward (bf16 autocast) → backward →
grad clip → optimizer step → EMA update → logging → checkpointing.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from khanh_llm.config import ModelConfig, TrainConfig
from khanh_llm.model.transformer import KhanhLLM
from khanh_llm.training.checkpoint import (
    EMA,
    load_checkpoint,
    prune_old_checkpoints,
    save_checkpoint,
)
from khanh_llm.training.logging import TrainingLogger
from khanh_llm.training.optim import WarmupCosineScheduler, build_optimizer


class Trainer:
    """BF16-autocast training loop with grad accumulation, EMA, and checkpointing.

    Args:
        model_cfg: ModelConfig instance.
        train_cfg: TrainConfig instance.
        dataloader: DataLoader yielding (input_ids, labels) tensors.
        resume_from: Optional path to a checkpoint to resume from.
    """

    def __init__(
        self,
        model_cfg: ModelConfig,
        train_cfg: TrainConfig,
        dataloader: DataLoader,
        resume_from: str | Path | None = None,
    ) -> None:
        self.model_cfg  = model_cfg
        self.train_cfg  = train_cfg
        self.dataloader = dataloader

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_float32_matmul_precision("high")

        self._build_model()
        self._build_optimizer()
        self._build_scheduler()
        self._build_ema()
        self._build_logger()

        self.step = 0
        self.tokens_processed = 0
        self.data_cursor: dict = {"shard_idx": 0, "offset": 0}

        if resume_from is not None:
            self._resume(resume_from)

    # ── Setup ──────────────────────────────────────────────────────────────────

    def _build_model(self) -> None:
        print(f"[Trainer] Initializing model on {self.device} ...")
        self.model = KhanhLLM(self.model_cfg).to(self.device)
        n_params = self.model.num_parameters()
        print(f"[Trainer] Parameters: {n_params/1e6:.1f}M")

        if self.train_cfg.compile:
            print(f"[Trainer] Compiling model (mode={self.train_cfg.compile_mode}) ...")
            self.model = torch.compile(self.model, mode=self.train_cfg.compile_mode)

    def _build_optimizer(self) -> None:
        self.optimizer = build_optimizer(self.model, self.train_cfg)

    def _build_scheduler(self) -> None:
        total_steps = self._estimate_total_steps()
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps = self.train_cfg.warmup_steps,
            total_steps  = total_steps,
            min_lr       = self.train_cfg.min_lr,
        )

    def _build_ema(self) -> None:
        self.ema = EMA(self.model, decay=self.train_cfg.ema_decay)

    def _build_logger(self) -> None:
        run_dir = Path(self.train_cfg.run_dir) / self.train_cfg.run_name
        self.logger = TrainingLogger(
            run_dir,
            use_wandb      = self.train_cfg.use_wandb,
            run_name       = self.train_cfg.run_name,
        )
        self.run_dir = run_dir

    def _estimate_total_steps(self) -> int:
        tokens_per_step = (
            self.train_cfg.micro_batch_size
            * self.train_cfg.gradient_accumulation_steps
            * self.model_cfg.max_seq_len
        )
        return max(1, self.train_cfg.max_tokens // tokens_per_step)

    def _resume(self, ckpt_path: str | Path) -> None:
        print(f"[Trainer] Resuming from {ckpt_path} ...")
        ckpt = load_checkpoint(
            ckpt_path, self.model, self.optimizer, self.scheduler, self.ema, self.device
        )
        self.step             = ckpt.get("step", 0)
        self.tokens_processed = ckpt.get("tokens_processed", 0)
        self.data_cursor      = ckpt.get("data_cursor", {"shard_idx": 0, "offset": 0})
        print(f"[Trainer] Resumed at step={self.step:,}, tokens={self.tokens_processed/1e9:.3f}B")

    # ── Training loop ──────────────────────────────────────────────────────────

    def train(self) -> None:
        """Run the main training loop until max_tokens is reached."""
        cfg = self.train_cfg
        use_ckpt = cfg.activation_checkpointing != "off"
        use_bf16 = cfg.dtype == "bfloat16"
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        data_iter = iter(self.dataloader)

        tokens_at_last_save = self.tokens_processed
        tokens_at_last_log  = self.tokens_processed
        t_start = time.perf_counter()

        print(f"[Trainer] Starting training. Target: {cfg.max_tokens/1e9:.1f}B tokens.")

        try:
            while self.tokens_processed < cfg.max_tokens:
                self.optimizer.zero_grad(set_to_none=True)
                accum_loss = 0.0

                # ── Gradient accumulation ──────────────────────────────────────────
                for micro_step in range(cfg.gradient_accumulation_steps):
                    try:
                        input_ids, labels = next(data_iter)
                    except StopIteration:
                        print("[Trainer] DataLoader exhausted. Training complete.")
                        self._save_checkpoint(loss=accum_loss)
                        return

                    input_ids = input_ids.to(self.device)
                    labels    = labels.to(self.device)

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                        logits, aux_loss, _ = self.model(
                            input_ids, use_checkpoint=use_ckpt
                        )
                        main_loss = criterion(
                            logits.view(-1, self.model_cfg.vocab_size), labels.view(-1)
                        )
                        loss = (main_loss + self.model_cfg.aux_loss_weight * aux_loss) / cfg.gradient_accumulation_steps

                    loss.backward()
                    accum_loss += loss.item()

                    batch_tokens = input_ids.numel()
                    self.tokens_processed += batch_tokens

                # ── Optimizer step ─────────────────────────────────────────────────
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=cfg.max_grad_norm
                ).item()
                self.optimizer.step()
                self.scheduler.step()
                self.ema.update(self.model)
                self.step += 1

                current_loss = accum_loss * cfg.gradient_accumulation_steps

                # ── Logging ────────────────────────────────────────────────────────
                if self.step % cfg.log_every_steps == 0:
                    t_now = time.perf_counter()
                    dt = max(t_now - t_start, 1e-6)
                    tok_per_sec = (self.tokens_processed - tokens_at_last_log) / dt
                    
                    remaining_tokens = max(0, cfg.max_tokens - self.tokens_processed)
                    eta_seconds = remaining_tokens / tok_per_sec if tok_per_sec > 0 else 0

                    self.logger.log(
                        step          = self.step,
                        tokens        = self.tokens_processed,
                        max_tokens    = cfg.max_tokens,
                        loss          = current_loss,
                        lr            = self.scheduler.get_last_lr()[0],
                        tokens_per_sec= tok_per_sec,
                        grad_norm     = grad_norm,
                        eta_seconds   = eta_seconds,
                    )
                    tokens_at_last_log = self.tokens_processed
                    t_start = t_now

                # ── Checkpointing ──────────────────────────────────────────────────
                tokens_since_save = self.tokens_processed - tokens_at_last_save
                if tokens_since_save >= cfg.save_every_tokens:
                    self._save_checkpoint(loss=current_loss)
                    prune_old_checkpoints(self.run_dir, keep_last_n=cfg.keep_last_n)
                    tokens_at_last_save = self.tokens_processed

            print(f"[Trainer] Training complete. Final step={self.step}, tokens={self.tokens_processed/1e9:.3f}B")
            self._save_checkpoint(loss=current_loss)

        except KeyboardInterrupt:
            print("\n[Trainer] Caught Ctrl+C! Saving graceful checkpoint before exit...")
            # If interrupted mid-accumulation, we don't have a reliable current_loss for this exact step,
            # so we use a generic 0.0 or the last logged loss.
            self._save_checkpoint(loss=0.0)
            return

    def _save_checkpoint(self, loss: float) -> None:
        save_checkpoint(
            run_dir           = self.run_dir,
            step              = self.step,
            loss              = loss,
            tokens_processed  = self.tokens_processed,
            model             = self.model,
            optimizer         = self.optimizer,
            scheduler         = self.scheduler,
            ema               = self.ema,
            data_cursor       = self.data_cursor,
            config_dict       = {
                "model": vars(self.model_cfg),
                "train": vars(self.train_cfg),
            },
        )
