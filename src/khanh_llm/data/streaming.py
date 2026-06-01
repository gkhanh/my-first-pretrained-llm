"""Streaming download + tokenize + pack pipeline.

Each source is streamed from HuggingFace datasets (never fully downloaded to RAM).
Documents are tokenized on the fly and packed into fixed-length seq_len chunks.
"""

from __future__ import annotations

from collections.abc import Iterator


# ── Document packer ──────────────────────────────────────────────────────────

class DocumentPacker:
    """Packs tokenized documents into fixed-length chunks with no padding."""

    def __init__(self, seq_len: int = 2048, eos_id: int = 0) -> None:
        self.seq_len = seq_len
        self.eos_id  = eos_id

    def pack(self, documents: Iterator[list[int]]) -> Iterator[list[int]]:
        buf: list[int] = []
        for doc_tokens in documents:
            buf.extend(doc_tokens)
            buf.append(self.eos_id)
            while len(buf) >= self.seq_len:
                yield buf[: self.seq_len]
                buf = buf[self.seq_len :]


# ── Per-source streaming ──────────────────────────────────────────────────────

def _text_from_example(example: dict, source_name: str) -> str | None:
    """Extract the text field from a HuggingFace dataset example."""
    # Most datasets use "text" or "content"
    for key in ("text", "content", "document", "passage", "body"):
        if key in example and example[key]:
            return str(example[key])
    # StackExchange preferences: combine question + best answer
    if "question" in example:
        q = example.get("question", "")
        answers = example.get("answers", [])
        best = max(answers, key=lambda a: a.get("pm_score", 0), default={}) if answers else {}
        ans = best.get("text", "")
        return f"{q}\n\n{ans}" if ans else q
    return None


def _stream_hf_source(
    hf_path: str,
    split: str,
    language_filter: list[str] | None,
    tokenizer,
    max_tokens: int | None,
) -> Iterator[list[int]]:
    """Stream and tokenize one HuggingFace dataset source."""
    from datasets import load_dataset
    from datasets.exceptions import DatasetNotFoundError

    load_kwargs: dict = dict(streaming=True)
    ds = None
    # Try several load-signatures because HF datasets are inconsistent:
    attempts = [
        lambda: load_dataset(hf_path, split=split, **load_kwargs),
        lambda: load_dataset(hf_path, split, split="train", **load_kwargs),
        lambda: load_dataset(hf_path, split="train", **load_kwargs),
    ]
    last_err = None
    for attempt in attempts:
        try:
            ds = attempt()
            break
        except DatasetNotFoundError as e:
            print(f"  SKIPPING {hf_path}: {e}")
            return
        except Exception as e:
            last_err = e
            continue
    if ds is None:
        print(f"  SKIPPING {hf_path}: {last_err}")
        return

    tokens_yielded = 0
    for example in ds:
        # Language filter for The Stack v2
        if language_filter:
            lang = example.get("lang") or example.get("language") or ""
            if lang not in language_filter:
                continue

        text = _text_from_example(example, hf_path)
        if not text or len(text) < 50:
            continue

        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue

        tokens_yielded += len(ids)
        yield ids

        if max_tokens and tokens_yielded >= max_tokens:
            break


def _stream_local_source(
    local_path: str,
    tokenizer,
    max_tokens: int | None,
) -> Iterator[list[int]]:
    """Stream and tokenize local .jsonl or .txt files."""
    import json
    from pathlib import Path

    p = Path(local_path)
    if not p.exists():
        print(f"  WARNING: local path {local_path} not found, skipping.")
        return

    files = sorted(p.rglob("*.jsonl")) + sorted(p.rglob("*.txt"))
    tokens_yielded = 0

    for f in files:
        with open(f, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get("text") or obj.get("content") or ""
                except json.JSONDecodeError:
                    text = line

                if len(text) < 50:
                    continue

                ids = tokenizer.encode(text, add_special_tokens=False)
                if not ids:
                    continue

                tokens_yielded += len(ids)
                yield ids

                if max_tokens and tokens_yielded >= max_tokens:
                    return


# ── Main entry point ──────────────────────────────────────────────────────────

def build_pretrain_stream(cfg, tokenizer) -> Iterator[list[int]]:
    """Build a weighted interleaved token stream from all sources in cfg.

    Uses reservoir-style interleaving: cycles through sources proportionally
    to their weights so the mix is maintained throughout the corpus.
    """
    import random
    from omegaconf import OmegaConf

    sources = OmegaConf.to_container(cfg.sources, resolve=True)
    total_weight = sum(s["weight"] for s in sources)

    # Build generators for each source
    def make_gen(s: dict):
        hf_path    = s.get("hf_path")
        local_path = s.get("local_path")
        split      = s.get("split", "train")
        lang_filter = s.get("language_filter")

        if hf_path:
            return _stream_hf_source(hf_path, split, lang_filter, tokenizer, max_tokens=None)
        elif local_path:
            return _stream_local_source(local_path, tokenizer, max_tokens=None)
        else:
            raise ValueError(f"Source {s['name']} has neither hf_path nor local_path")

    # Weighted random interleaving
    gens = [(s["weight"] / total_weight, make_gen(s), s["name"]) for s in sources]

    weights  = [w for w, _, _ in gens]
    iterators = [g for _, g, _ in gens]
    names    = [n for _, _, n in gens]
    active   = list(range(len(gens)))

    while active:
        # Pick a source proportional to its weight
        active_weights = [weights[i] for i in active]
        idx = active[random.choices(range(len(active)), weights=active_weights, k=1)[0]]
        try:
            yield next(iterators[idx])
        except StopIteration:
            print(f"  Source exhausted: {names[idx]}")
            active.remove(idx)
