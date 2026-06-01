"""Download the StarCoder2 tokenizer from HuggingFace and save locally.

Usage:
    python scripts/data/download_starcoder_tokenizer.py
    python scripts/data/download_starcoder_tokenizer.py --output data/tokenizers/starcoder2
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Download StarCoder2 tokenizer")
    p.add_argument(
        "--output", default="data/tokenizers/starcoder2",
        help="Local directory to save the tokenizer (default: data/tokenizers/starcoder2)"
    )
    p.add_argument(
        "--model", default="bigcode/starcoder2-3b",
        help="HuggingFace model to pull the tokenizer from (default: bigcode/starcoder2-3b)"
    )
    args = p.parse_args()

    from transformers import AutoTokenizer

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading tokenizer from {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.save_pretrained(str(out))

    print(f"Tokenizer saved to {out}")
    print(f"  Vocab size:   {tok.vocab_size}")
    print(f"  EOS token:    {tok.eos_token!r} (id={tok.eos_token_id})")

    # Verify FIM tokens are present
    vocab = tok.get_vocab()
    fim_tokens = ["<fim_prefix>", "<fim_suffix>", "<fim_middle>"]
    missing = [t for t in fim_tokens if t not in vocab]
    if missing:
        print(f"WARNING: FIM tokens missing: {missing}")
        print("  The FIM transform will not work correctly. Verify you are using a StarCoder2 tokenizer.")
    else:
        print("  FIM tokens:   present ✓")


if __name__ == "__main__":
    main()
