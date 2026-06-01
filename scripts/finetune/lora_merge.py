"""
Merge LoRA adapter weights back into the base model.

After fine-tuning with --use_lora, the adapter is stored separately.
This script merges the adapter into the base weights, producing a single
deployable model file with no peft dependency at inference time.

Usage — merge KhanhLLM + LoRA adapter:
    python -m scripts.finetune.lora_merge \
        --model_type khanh \
        --ckpt runs/khanh_1b/checkpoints/trained_khanh_gpt_latest.pt \
        --model_config configs/model/khanh_1b.yaml \
        --adapter runs/sft/khanh_code/adapter \
        --out runs/sft/khanh_code/merged.pt

Usage — merge HF base + LoRA adapter:
    python -m scripts.finetune.lora_merge \
        --model_type hf \
        --hf_model Qwen/Qwen2.5-Coder-1.5B-Instruct \
        --adapter runs/sft/qwen_code/adapter \
        --out runs/sft/qwen_code/merged
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.khanh_llm.finetune.adapters import merge_lora_into_base
from omegaconf import OmegaConf


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", required=True, choices=["khanh", "hf"])
    p.add_argument("--ckpt", default=None, help="KhanhLLM base checkpoint path")
    p.add_argument("--model_config", default="configs/model/khanh_1b.yaml")
    p.add_argument("--hf_model", default=None, help="HF base model ID")
    p.add_argument("--adapter", required=True, help="Path to saved peft adapter directory")
    p.add_argument("--out", required=True, help="Where to save merged output")
    p.add_argument("--device", default="cpu", help="cpu is safer for merging (no OOM risk)")
    return p.parse_args()


def main():
    args = parse_args()

    model_cfg = OmegaConf.load(args.model_config) if args.model_type == "khanh" else None

    print(f"Merging adapter {args.adapter} into base model...")
    merge_lora_into_base(
        model_type=args.model_type,
        ckpt_path=args.ckpt,
        model_cfg=model_cfg,
        hf_model_id=args.hf_model,
        adapter_path=args.adapter,
        out_path=args.out,
        device=args.device,
    )
    print(f"Merged model saved to {args.out}")


if __name__ == "__main__":
    main()
