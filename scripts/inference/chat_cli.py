"""chat_cli.py — Interactive text generation CLI.

Supports both the new KhanhLLM model (with KV cache) and the legacy 692M MoE
checkpoint (via --legacy flag).

Usage:
    # New model:
    python scripts/inference/chat_cli.py \
        --ckpt runs/khanh_1b/checkpoints/trained_khanh_gpt_latest.pt \
        --tokenizer data/tokenizers/starcoder2

    # Legacy 692M checkpoint:
    python scripts/inference/chat_cli.py --legacy \
        --ckpt runs/_archive_legacy_692m/checkpoint_latest.pth.tar \
        --tokenizer runs/_archive_legacy_692m/khanh_tokenizer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))  # repo root

import torch
from khanh_llm.inference.generator import Generator

torch.set_float32_matmul_precision("high")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KhanhLLM interactive text generator")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint file")
    p.add_argument("--tokenizer", required=True, help="Path to tokenizer directory")
    p.add_argument("--legacy", action="store_true", help="Load legacy 692M MoE checkpoint")
    p.add_argument("--device", default="cuda", help="Torch device (default: cuda)")
    p.add_argument("--no-ema", action="store_true", help="Load non-EMA weights (default: use EMA)")
    p.add_argument("--no-compile", action="store_true", help="Skip torch.compile")
    return p.parse_args()


def run_interactive(gen: Generator, args: argparse.Namespace) -> None:
    # Generation settings (mutable)
    settings = {
        "max_new_tokens": 200,
        "temperature": 0.6,
        "top_k": 40,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
    }

    print("=" * 60)
    print("  KhanhLLM Text Generator" + (" [LEGACY 692M]" if args.legacy else ""))
    print("=" * 60)
    print("Commands: 'settings' to adjust, 'q' to quit")
    print("=" * 60)

    while True:
        try:
            prompt = input("\n> Prompt: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if prompt.lower() in {"q", "quit", "exit"}:
            print("Goodbye!")
            break

        if prompt.lower() == "settings":
            _update_settings(settings)
            continue

        if not prompt:
            continue

        print("\nGenerating... ", end="", flush=True)
        print("\r" + " " * 20 + "\r", end="")

        generated_tokens = []
        try:
            for tok in gen.stream(prompt, **settings):
                print(tok, end="", flush=True)
                generated_tokens.append(tok)
        except Exception as e:
            print(f"\n[Error during generation: {e}]")
            continue

        print()  # newline after streamed output
        print(f"\n[{len(generated_tokens)} tokens generated]")


def _update_settings(settings: dict) -> None:
    print(f"\nCurrent settings: {settings}")
    try:
        for key, cast in [
            ("max_new_tokens", int), ("temperature", float),
            ("top_k", int), ("top_p", float), ("repetition_penalty", float)
        ]:
            val = input(f"  {key} [{settings[key]}] (Enter to keep): ").strip()
            if val:
                settings[key] = cast(val)
        print("Settings updated.")
    except ValueError:
        print("Invalid input, keeping current settings.")


def main() -> None:
    args = parse_args()

    if args.legacy:
        print(f"Loading legacy checkpoint from {args.ckpt} ...")
        gen = Generator.from_legacy_checkpoint(args.ckpt, args.tokenizer, device=args.device)
    else:
        print(f"Loading checkpoint from {args.ckpt} ...")
        gen = Generator.from_checkpoint(
            args.ckpt,
            args.tokenizer,
            device=args.device,
            use_ema=not args.no_ema,
            compile_model=not args.no_compile,
        )

    run_interactive(gen, args)


if __name__ == "__main__":
    main()
