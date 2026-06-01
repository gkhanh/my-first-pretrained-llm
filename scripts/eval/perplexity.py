"""
Compute perplexity of a KhanhLLM checkpoint on a held-out token shard.

Perplexity = exp(average cross-entropy loss over all tokens).
Lower is better. Run this at every checkpoint to track whether the model is learning.

Perplexity is evaluated PER SLICE separately (code / finance / general text) so you
can see which domain is improving or regressing.

Usage:
    python -m scripts.eval.perplexity \
        --ckpt runs/khanh_1b/checkpoints/trained_khanh_gpt_latest.pt \
        --data data/shards/eval \
        --config configs/model/khanh_1b.yaml
"""

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.khanh_llm.eval.perplexity import compute_perplexity
from src.khanh_llm.training.checkpoint import load_checkpoint_weights_only
from omegaconf import OmegaConf


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to .pt checkpoint file")
    p.add_argument("--data", required=True,
                   help="Directory with eval .bin shards (one subdir per slice: code/, finance/, text/)")
    p.add_argument("--config", default="configs/model/khanh_1b.yaml")
    p.add_argument("--max_batches", type=int, default=200,
                   help="Cap evaluation at this many batches per slice (fast estimate)")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    data_dir = Path(args.data)

    print(f"Loading checkpoint: {args.ckpt}")
    model = load_checkpoint_weights_only(args.ckpt, cfg, device=args.device)
    model.eval()

    slices = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    if not slices:
        slices = [""]  # flat dir — evaluate as a single slice

    results = {}
    for slice_name in slices:
        shard_dir = data_dir / slice_name if slice_name else data_dir
        print(f"\nEvaluating slice: {slice_name or 'all'}  ({shard_dir})")
        loss = compute_perplexity(
            model=model,
            shard_dir=shard_dir,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            device=args.device,
        )
        ppl = math.exp(loss)
        results[slice_name or "all"] = ppl
        print(f"  loss={loss:.4f}  perplexity={ppl:.2f}")

    print("\n=== Summary ===")
    for name, ppl in results.items():
        print(f"  {name:20s}  perplexity={ppl:.2f}")


if __name__ == "__main__":
    main()
