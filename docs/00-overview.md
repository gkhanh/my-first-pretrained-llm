# KhanhLLM — Project Overview

**KhanhLLM** is a from-scratch language model project targeting a 350M-parameter, coding-focused decoder-only Transformer, trained on a mixed code/finance/text corpus on a single NVIDIA RTX 5080 (16 GB VRAM) over approximately 2 years (~6,000 GPU-hours).

## What this repo is

| Track | Description |
|---|---|
| **From-scratch pretraining** | Training `khanh_1b` (Llama-style, dense, 24 layers) from zero on ~700 B tokens of code + finance + general text. This is the primary deliverable. |
| **Local fine-tuning pipeline** | A fully-local LoRA/QLoRA pipeline that works on KhanhLLM checkpoints **and** any HuggingFace causal-LM (Qwen 2.5, Llama 3, Mistral, Phi). Provides a usable model while pretrain bakes. |
| **Web interface skeleton** | FastAPI backend + Vite/React frontend stubs. Not implemented yet — see `web/README.md`. |

## Two domain capabilities in one model

- **Coding** (primary): Code completion, Fill-in-the-Middle (FIM), Python/JS/TS/Go/Rust
- **Finance chatbot** (secondary): Reads and summarises SEC filings, answers basic finance Q&A

## Folder map

```
building-llm/
├── src/khanh_llm/        ← Python package (model, training, inference, finetune, eval)
├── configs/              ← YAML configs for model, train, and data
├── scripts/              ← Thin CLI entrypoints (data prep, training, eval, inference)
├── docs/                 ← Design documents (you are here)
├── web/                  ← FastAPI + Vite/React skeleton
├── tests/                ← Smoke tests
└── runs/                 ← Gitignored: checkpoints, logs, samples
```

## Key design decisions (locked-in)

1. **Architecture**: Llama-style dense decoder (RoPE, RMSNorm, SwiGLU, GQA). MoE available but off by default.
2. **Tokenizer**: StarCoder2 (~49K vocab, code-aware BPE, FIM tokens included).
3. **Training precision**: BF16 autocast — no GradScaler needed on RTX 5080.
4. **Checkpoints**: `trained_khanh_gpt_step{N}_loss{X.XX}.pt` format, plus rolling `latest` symlink and EMA weights.
5. **Library policy**: Latest stable, pinned via `pyproject.toml` + lockfile.
6. **All local**: No cloud dependency. One-time HuggingFace download for datasets/tokenizer.

## Where to go next

- **Architecture details**: [`01-architecture.md`](01-architecture.md)
- **Training recipe**: [`02-training-recipe.md`](02-training-recipe.md)
- **Data pipeline**: [`03-data-pipeline.md`](03-data-pipeline.md)
- **RTX 5080 VRAM tuning**: [`04-rtx5080-tuning.md`](04-rtx5080-tuning.md)
- **2-year roadmap**: [`05-roadmap.md`](05-roadmap.md)
- **Web app design**: [`06-web-app-design.md`](06-web-app-design.md)
- **Evaluation strategy**: [`07-evaluation.md`](07-evaluation.md)
- **Fine-tuning pipeline**: [`08-finetuning.md`](08-finetuning.md)
- **Finance domain**: [`09-finance-domain.md`](09-finance-domain.md)
