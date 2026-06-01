"""Pretrain entrypoint — thin CLI wrapper over src/khanh_llm/training/trainer.py.

Usage:
    python -m scripts.train.pretrain --config configs/model/khanh_150m.yaml          # smoke test
    python -m scripts.train.pretrain --config configs/model/khanh_1b.yaml            # real run
    python -m scripts.train.pretrain --config configs/model/khanh_1b.yaml --max-steps 10  # quick check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))  # repo root → src/ is on path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, IterableDataset

from khanh_llm.config import ModelConfig, TrainConfig
from khanh_llm.data.shards import ShardedDataset
from khanh_llm.training.trainer import Trainer


class SyntheticDataset(IterableDataset):
    """Infinite random-token dataset for smoke-testing without real shards."""
    def __init__(self, vocab_size: int, seq_len: int, n_batches: int = 200):
        self.vocab_size = vocab_size
        self.seq_len    = seq_len
        self.n_batches  = n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            ids = torch.randint(0, self.vocab_size, (self.seq_len + 1,))
            yield ids[:-1], ids[1:]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KhanhLLM pretraining")
    p.add_argument("--config", required=True, help="Path to model YAML config (configs/model/)")
    p.add_argument("--train-config", default="configs/train/pretrain_5080.yaml", help="Train YAML config")
    p.add_argument("--data-config", default="configs/data/pretrain_mix.yaml", help="Data YAML config")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    p.add_argument("--max-steps", type=int, default=None, help="Stop after N optimizer steps (for smoke testing)")
    p.add_argument("--synthetic", action="store_true",
                   help="Use random token data instead of real shards (for pipeline smoke-testing)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model_cfg_raw = OmegaConf.load(args.config)
    train_cfg_raw = OmegaConf.load(args.train_config)
    data_cfg_raw  = OmegaConf.load(args.data_config)

    model_cfg = ModelConfig(**OmegaConf.to_container(model_cfg_raw, resolve=True))
    train_cfg = TrainConfig(**OmegaConf.to_container(train_cfg_raw, resolve=True))

    if args.max_steps is not None:
        tokens_per_step = (
            train_cfg.micro_batch_size * train_cfg.gradient_accumulation_steps * model_cfg.max_seq_len
        )
        train_cfg.max_tokens = args.max_steps * tokens_per_step

    if args.synthetic:
        n_batches = (args.max_steps or 20) * train_cfg.gradient_accumulation_steps + 10
        dataset = SyntheticDataset(model_cfg.vocab_size, model_cfg.max_seq_len, n_batches=n_batches)
        loader  = DataLoader(dataset, batch_size=train_cfg.micro_batch_size, num_workers=0)
        print("WARNING: using synthetic random data — for pipeline testing only, not real training")
    else:
        shard_dir = OmegaConf.to_container(data_cfg_raw, resolve=True).get("shard_dir", "data/shards/pretrain")
        dataset   = ShardedDataset(shard_dir, seq_len=model_cfg.max_seq_len)
        loader    = DataLoader(dataset, batch_size=train_cfg.micro_batch_size, num_workers=4, pin_memory=True)

    trainer = Trainer(
        model_cfg   = model_cfg,
        train_cfg   = train_cfg,
        dataloader  = loader,
        resume_from = args.resume,
    )
    trainer.train()


if __name__ == "__main__":
    main()
