"""
Supervised Fine-Tuning (SFT) entrypoint.

Works for BOTH:
  - KhanhLLM checkpoints (from runs/khanh_1b/checkpoints/)
  - External HuggingFace causal-LM models (Qwen2.5, Llama 3, Mistral, Phi, etc.)

For external models, uses QLoRA (4-bit base + LoRA adapters) so a 7B model fits in 16GB.
For KhanhLLM, uses full fine-tuning or LoRA depending on --use_lora flag.

Usage — fine-tune KhanhLLM:
    python -m scripts.finetune.sft \
        --model_type khanh \
        --ckpt runs/khanh_1b/checkpoints/trained_khanh_gpt_latest.pt \
        --model_config configs/model/khanh_1b.yaml \
        --train_config configs/train/sft_code.yaml \
        --data data/sft/code.jsonl

Usage — QLoRA fine-tune Qwen2.5-Coder-1.5B:
    python -m scripts.finetune.sft \
        --model_type hf \
        --hf_model Qwen/Qwen2.5-Coder-1.5B-Instruct \
        --train_config configs/train/qlora_external.yaml \
        --data data/sft/code.jsonl \
        --use_lora
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.khanh_llm.finetune.sft import run_sft
from omegaconf import OmegaConf


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", required=True, choices=["khanh", "hf"],
                   help="'khanh' = your own checkpoint, 'hf' = HuggingFace model")

    # KhanhLLM-specific
    p.add_argument("--ckpt", default=None, help="Path to KhanhLLM .pt checkpoint")
    p.add_argument("--model_config", default="configs/model/khanh_1b.yaml")

    # HuggingFace-specific
    p.add_argument("--hf_model", default=None,
                   help="HuggingFace model ID, e.g. Qwen/Qwen2.5-Coder-1.5B-Instruct")

    # Shared
    p.add_argument("--train_config", default="configs/train/sft_code.yaml")
    p.add_argument("--data", required=True, help="Path to .jsonl SFT dataset")
    p.add_argument("--out_dir", default=None,
                   help="Where to save adapter/checkpoint. Defaults to runs/sft/<model>/<timestamp>")
    p.add_argument("--use_lora", action="store_true",
                   help="Use LoRA adapters instead of full fine-tuning")
    p.add_argument("--use_4bit", action="store_true",
                   help="Load base model in 4-bit (QLoRA). Only for --model_type hf")
    p.add_argument("--resume", default=None, help="Resume from a previous SFT adapter checkpoint")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    train_cfg = OmegaConf.load(args.train_config)

    if args.model_type == "khanh":
        if not args.ckpt:
            raise ValueError("--ckpt is required for --model_type khanh")
        model_cfg = OmegaConf.load(args.model_config)
    else:
        if not args.hf_model:
            raise ValueError("--hf_model is required for --model_type hf")
        model_cfg = None

    run_sft(
        model_type=args.model_type,
        ckpt_path=args.ckpt,
        model_cfg=model_cfg,
        hf_model_id=args.hf_model,
        train_cfg=train_cfg,
        data_path=args.data,
        out_dir=args.out_dir,
        use_lora=args.use_lora or args.use_4bit,
        use_4bit=args.use_4bit,
        resume_from=args.resume,
        device=args.device,
    )


if __name__ == "__main__":
    main()
