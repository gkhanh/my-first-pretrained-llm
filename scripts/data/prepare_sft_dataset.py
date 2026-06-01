"""
Build ChatML-formatted SFT datasets for code and finance fine-tuning.

Supported datasets (all locally cached after first HuggingFace download):
  - code:    OSS-Instruct (Magicoder), CodeAlpaca-20k
  - finance: FinQA, ConvFinQA, FiQA-2018
  - chat:    UltraChat-200k subset, OpenHermes-2.5 (filtered)

Output: .jsonl files under data/sft/, one ChatML conversation per line:
    {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

Usage:
    python -m scripts.data.prepare_sft_dataset --split code --out data/sft/code.jsonl
    python -m scripts.data.prepare_sft_dataset --split finance --out data/sft/finance.jsonl
    python -m scripts.data.prepare_sft_dataset --split chat --out data/sft/chat.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.khanh_llm.finetune.chat_templates import apply_chatml_template


DATASET_REGISTRY = {
    "code": [
        {"hf_path": "ise-uiuc/Magicoder-OSS-Instruct-75K", "split": "train"},
        {"hf_path": "sahil2801/CodeAlpaca-20k", "split": "train"},
    ],
    "finance": [
        {"hf_path": "ibm/finqa", "split": "train"},
        {"hf_path": "ibm/conv_finqa", "split": "train"},
        {"hf_path": "BeIR/fiqa", "split": "train"},
    ],
    "chat": [
        {"hf_path": "HuggingFaceH4/ultrachat_200k", "split": "train_sft", "max_examples": 50000},
        {"hf_path": "teknium/OpenHermes-2.5", "split": "train", "max_examples": 50000},
    ],
}


def convert_to_chatml(example: dict, source: str) -> dict | None:
    """Convert a raw HuggingFace example to ChatML message list."""
    if source in ("ise-uiuc/Magicoder-OSS-Instruct-75K", "sahil2801/CodeAlpaca-20k"):
        instruction = example.get("instruction", "") or example.get("problem", "")
        response = example.get("output", "") or example.get("solution", "")
        if not instruction or not response:
            return None
        return {"messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]}
    if source == "HuggingFaceH4/ultrachat_200k":
        return {"messages": example.get("messages", [])}
    if source == "teknium/OpenHermes-2.5":
        convs = example.get("conversations", [])
        messages = []
        for turn in convs:
            role = "user" if turn.get("from") == "human" else "assistant"
            messages.append({"role": role, "content": turn.get("value", "")})
        return {"messages": messages} if messages else None
    return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, choices=["code", "finance", "chat"])
    p.add_argument("--out", default=None)
    p.add_argument("--max_examples", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out or f"data/sft/{args.split}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets: pip install datasets")
        sys.exit(1)

    written = 0
    with open(out_path, "w") as f:
        for spec in DATASET_REGISTRY[args.split]:
            ds_kwargs = {"path": spec["hf_path"], "split": spec["split"]}
            print(f"Loading {spec['hf_path']}...")
            ds = load_dataset(**ds_kwargs, trust_remote_code=True)
            cap = spec.get("max_examples") or args.max_examples
            if cap:
                ds = ds.select(range(min(cap, len(ds))))
            for example in ds:
                record = convert_to_chatml(example, spec["hf_path"])
                if record:
                    f.write(json.dumps(record) + "\n")
                    written += 1

    print(f"Wrote {written} examples to {out_path}")


if __name__ == "__main__":
    main()
