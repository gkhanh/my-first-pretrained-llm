"""
Evaluate a KhanhLLM checkpoint on FinQA and ConvFinQA.

FinQA: financial question answering requiring numeric reasoning over financial reports.
ConvFinQA: multi-turn conversational version of FinQA.

Metric: exact match on numeric answers (after normalizing units, commas, % signs).

For a 1B model after finance SFT: expect ~30-50% exact match on FinQA.
Baseline (no finance training): ~5-15%.

Usage:
    python -m scripts.eval.finance_qa \
        --ckpt runs/khanh_1b/checkpoints/trained_khanh_gpt_latest.pt \
        --config configs/model/khanh_1b.yaml \
        --dataset finqa
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.khanh_llm.eval.finance_eval import run_finqa, run_conv_finqa
from src.khanh_llm.training.checkpoint import load_checkpoint_weights_only
from src.khanh_llm.inference.generator import Generator
from src.khanh_llm.data.tokenizer import load_tokenizer
from omegaconf import OmegaConf


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", default="configs/model/khanh_1b.yaml")
    p.add_argument("--tokenizer", default="bigcode/starcoder2-3b")
    p.add_argument("--dataset", default="finqa", choices=["finqa", "conv_finqa", "both"])
    p.add_argument("--max_examples", type=int, default=200,
                   help="Cap at N examples for a fast estimate")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0.0 = greedy decoding (deterministic, best for QA)")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    print(f"Loading checkpoint: {args.ckpt}")
    model = load_checkpoint_weights_only(args.ckpt, cfg, device=args.device)
    tokenizer = load_tokenizer(args.tokenizer)
    generator = Generator(model=model, tokenizer=tokenizer, device=args.device)

    all_results = {}

    if args.dataset in ("finqa", "both"):
        print("\nRunning FinQA...")
        results = run_finqa(
            generator=generator,
            max_examples=args.max_examples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        all_results["finqa"] = results
        print(f"FinQA exact match: {results['exact_match']:.1%}  ({results['correct']}/{results['total']})")

    if args.dataset in ("conv_finqa", "both"):
        print("\nRunning ConvFinQA...")
        results = run_conv_finqa(
            generator=generator,
            max_examples=args.max_examples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        all_results["conv_finqa"] = results
        print(f"ConvFinQA exact match: {results['exact_match']:.1%}  ({results['correct']}/{results['total']})")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
