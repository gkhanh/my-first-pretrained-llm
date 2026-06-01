"""
Apply FIM (Fill-in-the-Middle) SPM transform to pre-tokenized code shards.

Takes .bin shards produced by prepare_pretrain_corpus.py and re-writes them
with ~50% of code documents randomly permuted into SPM format:
    <fim_prefix> prefix_tokens <fim_suffix> suffix_tokens <fim_middle> middle_tokens

Uses the StarCoder2 tokenizer's built-in FIM special tokens:
  - <fim_prefix>  (token ID looked up from tokenizer)
  - <fim_middle>
  - <fim_suffix>
  - <fim_pad>     (used to pad short splits, not for masking)

Only applied to the code slice. Finance/text shards are passed through unchanged.

Usage:
    python -m scripts.data.pack_fim \
        --in_dir  data/shards/pretrain/code \
        --out_dir data/shards/pretrain_fim/code \
        --fim_rate 0.5
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.khanh_llm.data.shards import ShardReader, ShardWriter
from src.khanh_llm.data.tokenizer import load_tokenizer
from src.khanh_llm.data.fim import fim_permute_sequence


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--fim_rate", type=float, default=0.5,
                   help="Fraction of documents to transform with FIM")
    p.add_argument("--tokenizer", default="bigcode/starcoder2-3b")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.tokenizer)

    fim_prefix_id = tokenizer.convert_tokens_to_ids("<fim_prefix>")
    fim_middle_id = tokenizer.convert_tokens_to_ids("<fim_middle>")
    fim_suffix_id = tokenizer.convert_tokens_to_ids("<fim_suffix>")

    if any(x <= 0 for x in [fim_prefix_id, fim_middle_id, fim_suffix_id]):
        raise ValueError(
            "FIM special tokens not found in tokenizer. "
            "Make sure you are using the StarCoder2 tokenizer."
        )

    fim_tokens = (fim_prefix_id, fim_middle_id, fim_suffix_id)

    reader = ShardReader(in_dir)
    writer = ShardWriter(out_dir)

    total_docs = 0
    fim_docs = 0

    for doc_tokens in reader.iter_documents():
        total_docs += 1
        if random.random() < args.fim_rate:
            doc_tokens = fim_permute_sequence(doc_tokens, fim_tokens)
            fim_docs += 1
        writer.write_document(doc_tokens)

    writer.flush()
    print(f"Processed {total_docs} documents, applied FIM to {fim_docs} ({fim_docs/max(1,total_docs)*100:.1f}%)")
    print(f"Output written to {out_dir}")


if __name__ == "__main__":
    main()
