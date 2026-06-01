"""Tokenizer wrapper for KhanhLLM.

Wraps the HuggingFace tokenizer (default: StarCoder2) with convenience methods
for FIM token handling and encoding/decoding utilities.
"""

from __future__ import annotations

from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerFast

# StarCoder2 FIM special token strings
FIM_PREFIX = "<fim_prefix>"
FIM_SUFFIX = "<fim_suffix>"
FIM_MIDDLE = "<fim_middle>"
FIM_PAD    = "<fim_pad>"


def load_tokenizer(path: str | Path) -> PreTrainedTokenizerFast:
    """Load tokenizer from a local directory or HuggingFace Hub (auto-downloads).

    Args:
        path: Local directory path OR HuggingFace model ID (e.g. "bigcode/starcoder2-3b").
              If a local path doesn't exist, falls back to downloading from HuggingFace.
    """
    local = Path(path)
    if local.exists():
        tok = AutoTokenizer.from_pretrained(str(local))
    else:
        # Auto-download from HuggingFace Hub (cached in ~/.cache/huggingface/)
        print(f"Tokenizer not found locally, downloading from HuggingFace: {path}")
        tok = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
        # Cache locally for future runs
        cache_dir = Path("data/tokenizers") / Path(str(path)).name
        cache_dir.mkdir(parents=True, exist_ok=True)
        tok.save_pretrained(str(cache_dir))
        print(f"Tokenizer cached at {cache_dir}")
    return tok


def get_fim_token_ids(tok: PreTrainedTokenizerFast) -> dict[str, int]:
    """Return a dict mapping FIM token names to their IDs.

    Raises:
        KeyError: If the tokenizer does not contain FIM special tokens.
    """
    vocab = tok.get_vocab()
    missing = [t for t in [FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE] if t not in vocab]
    if missing:
        raise KeyError(
            f"Tokenizer is missing FIM tokens: {missing}. "
            "Make sure you are using the StarCoder2 tokenizer."
        )
    return {
        "prefix": vocab[FIM_PREFIX],
        "suffix": vocab[FIM_SUFFIX],
        "middle": vocab[FIM_MIDDLE],
        "pad":    vocab.get(FIM_PAD, tok.eos_token_id),
    }
