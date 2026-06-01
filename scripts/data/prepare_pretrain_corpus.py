"""
Auto-download, tokenize, and pack the pretraining corpus into .bin shards.

Everything streams directly from HuggingFace — no manual downloads needed.
Run this once before training. It resumes automatically if interrupted.

Data mix (configs/data/pretrain_mix.yaml):
  50% The Stack v2 code (Python/JS/TS/Go/Rust)
   8% StackExchange programming Q&A
   7% Markdown/docs from The Stack
   8% SEC EDGAR financial filings      ← local only, skip if not downloaded
   5% Financial news (FNSPID)          ← local only, skip if not downloaded
   2% StackExchange finance Q&A
  10% Wikipedia English
  10% C4 general web text

Usage:
    # Full corpus (~390B tokens, takes days — streams while training is possible):
    python -m scripts.data.prepare_pretrain_corpus

    # Quick smoke-test (50M tokens, ~10 min):
    python3 -m scripts.data.prepare_pretrain_corpus --max-tokens 500_000_000

    # After smoke-test passes, run the full corpus in the background:
    nohup python -m scripts.data.prepare_pretrain_corpus &
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.khanh_llm.data.streaming import DocumentPacker, build_pretrain_stream
from src.khanh_llm.data.shards import ShardWriter
from src.khanh_llm.data.tokenizer import load_tokenizer
from omegaconf import OmegaConf


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data/pretrain_mix.yaml")
    p.add_argument("--out-dir", default="data/shards/pretrain")
    p.add_argument("--tokenizer", default="bigcode/starcoder2-3b",
                   help="HF tokenizer name or local path. Downloaded automatically on first run.")
    p.add_argument("--shard-size", type=int, default=500_000_000,
                   help="Tokens per shard file (default 500M)")
    p.add_argument("--max-tokens", type=int, default=None,
                   help="Stop after this many tokens total (omit for full corpus)")
    p.add_argument("--seq-len", type=int, default=2048)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-download tokenizer from HuggingFace if not cached locally
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = load_tokenizer(args.tokenizer)
    print(f"Vocab size: {tokenizer.vocab_size}")

    packer = DocumentPacker(seq_len=args.seq_len, eos_id=tokenizer.eos_token_id)

    # Count existing shards so we can resume
    existing = sorted(out_dir.glob("shard_*.bin"))
    shard_idx = len(existing)
    total_tokens = shard_idx * args.shard_size
    print(f"Resuming from shard {shard_idx} ({total_tokens:,} tokens already written)")

    tokens_to_skip = total_tokens  # skip already-written tokens from the stream
    tokens_written = total_tokens

    print(f"Streaming data from HuggingFace (auto-download on first run)...")
    print(f"Output: {out_dir}")
    if args.max_tokens:
        print(f"Stopping at {args.max_tokens:,} tokens")

    doc_stream = build_pretrain_stream(cfg, tokenizer)
    packed_stream = packer.pack(doc_stream)

    current_shard_tokens: list[int] = []
    skipped = 0

    for chunk in packed_stream:
        # Resume: skip chunks already written to disk
        if skipped < tokens_to_skip:
            skipped += args.seq_len
            continue

        current_shard_tokens.extend(chunk)

        if len(current_shard_tokens) >= args.shard_size:
            _write_shard(out_dir, shard_idx, current_shard_tokens[:args.shard_size])
            tokens_written += args.shard_size
            shard_idx += 1
            current_shard_tokens = current_shard_tokens[args.shard_size:]
            print(f"  Shard {shard_idx-1} written — {tokens_written:,} tokens total")

        if args.max_tokens and tokens_written >= args.max_tokens:
            break

    # Write final partial shard if it has content
    if current_shard_tokens:
        _write_shard(out_dir, shard_idx, current_shard_tokens)
        tokens_written += len(current_shard_tokens)
        print(f"  Shard {shard_idx} written (partial) — {tokens_written:,} tokens total")

    print(f"\nDone. {tokens_written:,} tokens in {shard_idx+1} shards at {out_dir}")
    print("Now run: python -m scripts.train.pretrain --config configs/model/khanh_1b.yaml")


def _write_shard(out_dir: Path, idx: int, tokens: list[int]) -> None:
    import numpy as np
    path = out_dir / f"shard_{idx:04d}.bin"
    arr = np.array(tokens, dtype=np.uint16)
    arr.tofile(path)


if __name__ == "__main__":
    main()
