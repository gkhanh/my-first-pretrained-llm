# Data Pipeline

## Overview

```
Source datasets (HuggingFace / web download)
        │
        ▼
  Filtering + deduplication
        │
        ▼
  Tokenization (StarCoder2 tokenizer)
        │
        ▼
  FIM transform (code slice, p=0.5)
        │
        ▼
  Sequence packing → 2048-token chunks
        │
        ▼
  Binary shard files (.bin + .idx)
        │
        ▼
  Training loop (mmap-loaded, no tokenizer overhead)
```

## Data mix

| Source | Weight | Notes |
|---|---|---|
| The Stack v2 (Python/JS/TS/Go/Rust) | 50% | Permissively licensed code |
| StackExchange — programming | 8% | Stack Overflow, Code Review — Q&A format |
| Markdown/docs from The Stack | 7% | READMEs, API docs |
| SEC EDGAR 10-K/10-Q/8-K filings | 8% | Finance domain text |
| Financial news (FNSPID or similar) | 5% | Current-events finance language |
| StackExchange — Money/Quant/Finance | 2% | Finance Q&A format |
| Wikipedia (English) | 10% | General world knowledge |
| C4 (small slice) | 10% | General web English |

Total: ~65% code, ~15% finance, ~20% general text. Tunable via `configs/data/pretrain_mix.yaml`.

## Tokenizer

**StarCoder2 tokenizer** (Apache-2.0, ~49K vocab, code-aware byte-level BPE).

Download once:
```bash
python scripts/data/download_starcoder_tokenizer.py
# Saves to: data/tokenizers/starcoder2/
```

Why StarCoder2 instead of the custom 50K BPE on C4:
- Purpose-built for code: identifiers, indentation, operators are tokenized efficiently
- Includes FIM special tokens: `<fim_prefix>`, `<fim_suffix>`, `<fim_middle>`, `<fim_pad>`
- Byte-level BPE handles any unicode without `[UNK]`

The legacy custom tokenizer is archived at `runs/_archive_legacy_692m/khanh_tokenizer/`. It is only needed to interpret the legacy training CSV.

## FIM Transform (Fill-in-the-Middle)

Applied to code sequences only, with probability `p_fim = 0.5`.

**SPM format** (Suffix-Prefix-Middle — used by StarCoder2):
```
<fim_prefix>PREFIX_TOKENS<fim_suffix>SUFFIX_TOKENS<fim_middle>MIDDLE_TOKENS
```

Algorithm per document:
1. With prob `1 - p_fim`: leave as causal (prefix → completion).
2. With prob `p_fim`: pick a random split point for prefix/suffix/middle, reorder into SPM format.

The model then learns to predict the middle given prefix+suffix — this is what makes it useful for editor autocomplete (tab completion, insert-at-cursor).

Implementation: `src/khanh_llm/data/fim.py`
Config key: `data.fim_rate: 0.5`

## Sequence packing

Documents are concatenated into 2048-token chunks with document boundary markers:

```python
# Pseudocode
chunk = []
for doc_tokens in stream:
    chunk.extend(doc_tokens)
    chunk.append(EOS)       # document boundary
    while len(chunk) >= 2048:
        yield chunk[:2048]
        chunk = chunk[2048:]
```

**Attention boundary reset**: position IDs are reset at each `EOS` token so that attention does not leak across documents. In practice, with packing=2048 and typical doc lengths of 300–2000 tokens, most sequences contain 2–8 documents.

Implementation: `src/khanh_llm/data/streaming.py`

## Pre-tokenized shards (`.bin` format)

Each shard is a flat array of `uint16` token IDs (49K vocab fits in uint16).

```
shard_0000.bin    # ~500M tokens = ~1 GB on disk
shard_0000.idx    # byte offsets for each document
shard_0001.bin
...
```

The training `DataLoader` memory-maps shards via `np.memmap` — no tokenizer in the hot path, no CPU tokenization bottleneck.

Build shards:
```bash
python scripts/data/prepare_pretrain_corpus.py \
    --config configs/data/pretrain_mix.yaml \
    --output-dir data/shards/pretrain/ \
    --shard-size 500_000_000   # tokens per shard
```

Implementation: `src/khanh_llm/data/shards.py`

## Finance corpus preparation

```bash
python scripts/data/prepare_finance_corpus.py \
    --config configs/data/finance_only.yaml \
    --output-dir data/shards/finance/
```

Sources:
- **SEC EDGAR**: Public API (`https://data.sec.gov/submissions/`). Filter to 10-K, 10-Q, 8-K. Strip HTML, keep text sections.
- **FNSPID** (or similar permissive news dataset): filter to finance-tagged articles.
- **StackExchange Money/Quant/Personal Finance**: available as HuggingFace dataset dump.

All sources are publicly available and permissively licensed. No paywalled or non-redistributable content.

## SFT dataset preparation

```bash
python scripts/data/prepare_sft_dataset.py \
    --output data/sft/code_sft.jsonl   # ChatML format
```

Output format (ChatML):
```json
{"messages": [
  {"role": "user", "content": "Write a Python function that..."},
  {"role": "assistant", "content": "```python\ndef foo():\n    ..."}
]}
```

## Reproducibility

- Shard ordering determined by `configs/data/pretrain_mix.yaml` seeds.
- Each shard has a deterministic filename that encodes source + split.
- Training resumes from a `data_shard_cursor` saved in each checkpoint — exact byte offset into the current shard.
- Document deduplication uses MinHash LSH (via `datasketch`) before tokenization.
