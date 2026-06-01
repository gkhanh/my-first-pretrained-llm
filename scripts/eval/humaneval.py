"""
Evaluate a KhanhLLM checkpoint on HumanEval (code generation).

HumanEval: 164 Python programming problems. The model is given a function signature
and docstring, and must complete the function body. We measure pass@1 (fraction of
problems the model solves correctly on the first attempt).

For a 1B model: expect ~10-20% pass@1 early in training, ~25-40% after full pretrain,
~45-60% after code SFT. TinyLlama-1.1B gets ~8-12% pass@1 without SFT.

Usage:
    python -m scripts.eval.humaneval \
        --ckpt runs/khanh_1b/checkpoints/trained_khanh_gpt_latest.pt \
        --config configs/model/khanh_1b.yaml \
        --n_samples 1
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.khanh_llm.eval.code_eval import run_humaneval
from src.khanh_llm.training.checkpoint import load_checkpoint_weights_only
from src.khanh_llm.inference.generator import Generator
from src.khanh_llm.data.tokenizer import load_tokenizer
from omegaconf import OmegaConf


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", default="configs/model/khanh_1b.yaml")
    p.add_argument("--tokenizer", default="bigcode/starcoder2-3b")
    p.add_argument("--n_samples", type=int, default=1,
                   help="Samples per problem for pass@k estimation. 1 = fast, 10+ = accurate")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--out", default=None, help="Save results JSON to this path")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    print(f"Loading checkpoint: {args.ckpt}")
    model = load_checkpoint_weights_only(args.ckpt, cfg, device=args.device)
    tokenizer = load_tokenizer(args.tokenizer)
    generator = Generator(model=model, tokenizer=tokenizer, device=args.device)

    results = run_humaneval(
        generator=generator,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    pass_at_1 = results["pass@1"]
    print(f"\nHumanEval pass@1: {pass_at_1:.1%}  ({results['passed']}/{results['total']} problems)")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Full results saved to {out_path}")


if __name__ == "__main__":
    main()
