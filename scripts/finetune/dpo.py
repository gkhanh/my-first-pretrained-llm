"""
DPO (Direct Preference Optimization) fine-tuning entrypoint.

Run AFTER SFT. Takes a preference dataset (chosen vs rejected response pairs)
and further aligns the model toward preferred outputs without a reward model.

Input format (.jsonl, one line per example):
    {
        "prompt": "Write a Python function to ...",
        "chosen": "def foo(x):\n    return x * 2",
        "rejected": "def foo(x):\n    return x + x  # unnecessarily verbose"
    }

Usage:
    python -m scripts.finetune.dpo \
        --model_type khanh \
        --ckpt runs/sft/khanh_code/merged.pt \
        --model_config configs/model/khanh_1b.yaml \
        --train_config configs/train/sft_code.yaml \
        --data data/dpo/code_preferences.jsonl \
        --out_dir runs/dpo/khanh_code
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.khanh_llm.finetune.dpo import run_dpo
from omegaconf import OmegaConf


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", required=True, choices=["khanh", "hf"])
    p.add_argument("--ckpt", default=None, help="KhanhLLM checkpoint (post-SFT recommended)")
    p.add_argument("--model_config", default="configs/model/khanh_1b.yaml")
    p.add_argument("--hf_model", default=None, help="HF model path (post-SFT merged)")
    p.add_argument("--train_config", default="configs/train/sft_code.yaml")
    p.add_argument("--data", required=True, help="Path to DPO preference .jsonl")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--beta", type=float, default=0.1,
                   help="DPO beta — controls how far from SFT policy we drift (0.05-0.5)")
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    train_cfg = OmegaConf.load(args.train_config)
    model_cfg = OmegaConf.load(args.model_config) if args.model_type == "khanh" else None

    run_dpo(
        model_type=args.model_type,
        ckpt_path=args.ckpt,
        model_cfg=model_cfg,
        hf_model_id=args.hf_model,
        train_cfg=train_cfg,
        data_path=args.data,
        out_dir=args.out_dir,
        beta=args.beta,
        use_lora=args.use_lora,
        device=args.device,
    )


if __name__ == "__main__":
    main()
