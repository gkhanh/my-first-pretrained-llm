# Local Fine-Tuning Pipeline

## Overview

A fully-local pipeline at `src/khanh_llm/finetune/` that works in two modes:

| Mode | Base model | Use case |
|---|---|---|
| **A: KhanhLLM** | Your `trained_khanh_gpt_*.pt` checkpoint | Fine-tune your own pretrained model |
| **B: External HF model** | Any `AutoModelForCausalLM` (Qwen 2.5, Llama 3, Mistral, Phi) | QLoRA on 7B models while KhanhLLM is pretraining |

## Mode A: Fine-tuning KhanhLLM

```bash
# Full fine-tune (small model, lots of VRAM)
python scripts/finetune/sft.py \
    --config configs/train/sft_code.yaml \
    --ckpt runs/khanh_1b/checkpoints/trained_khanh_gpt_latest.pt \
    --mode full

# LoRA fine-tune (recommended for iteration speed)
python scripts/finetune/sft.py \
    --config configs/train/lora_local.yaml \
    --ckpt runs/khanh_1b/checkpoints/trained_khanh_gpt_latest.pt \
    --mode lora
```

- Uses the same model/tokenizer code as pretraining.
- LoRA adapters saved to `runs/<run>/adapters/`.
- Merge adapter into weights: `python scripts/finetune/lora_merge.py --ckpt ... --adapter ...`

## Mode B: Fine-tuning external HuggingFace models (QLoRA)

```bash
# Default: Qwen2.5-1.5B (recommended first test)
python scripts/finetune/sft.py \
    --config configs/train/qlora_external.yaml \
    --base-model Qwen/Qwen2.5-1.5B \
    --mode qlora

# 7B model (fits in 16 GB with 4-bit base + LoRA adapters)
python scripts/finetune/sft.py \
    --config configs/train/qlora_external.yaml \
    --base-model Qwen/Qwen2.5-7B \
    --mode qlora
```

VRAM requirements for QLoRA:

| Base model | 4-bit base | + LoRA adapters | Total VRAM |
|---|---|---|---|
| 0.5B | ~0.5 GB | ~0.2 GB | ~2–3 GB |
| 1.5B | ~1 GB | ~0.3 GB | ~3–5 GB |
| 7B | ~4.5 GB | ~0.5 GB | ~7–10 GB |
| 13B | ~8 GB | ~0.8 GB | ~13–15 GB |

## Chat templating

`src/khanh_llm/finetune/chat_templates.py` supports three formats:

| Format | Used by |
|---|---|
| **ChatML** | KhanhLLM (default), OpenHermes, many open models |
| **Llama-3-instruct** | Meta Llama 3.x family |
| **Qwen-instruct** | Qwen 2.x family |

ChatML format:
```
<|im_start|>system
You are a helpful coding assistant.<|im_end|>
<|im_start|>user
Write a binary search function in Python.<|im_end|>
<|im_start|>assistant
```

## SFT datasets (locally cached after first download)

| Capability | Dataset | Size | HF path |
|---|---|---|---|
| Code instruction | CodeAlpaca-20k | ~20k examples | `sahil2801/CodeAlpaca-20k` |
| Code instruction | Magicoder OSS-Instruct | ~74k examples | `ise-uiuc/Magicoder-OSS-Instruct-75K` |
| Finance Q&A | FinQA | ~8k examples | `ibm/finqa` |
| Finance Q&A | ConvFinQA | ~3k examples | `ibm/convfinqa` |
| Finance Q&A | FiQA-2018 | ~6k examples | `BeIR/fiqa` |
| General chat | UltraChat-200k (subset) | ~200k examples | `HuggingFaceH4/ultrachat_200k` |
| General chat | OpenHermes-2.5 (filtered) | ~1M examples | `teknium/OpenHermes-2.5` |

Prepare SFT datasets in ChatML JSONL format:
```bash
python scripts/data/prepare_sft_dataset.py \
    --datasets code_alpaca magicoder finqa ultrachat \
    --output data/sft/code_finance_chat.jsonl
```

## Two-stage SFT

To prevent catastrophic forgetting of general capabilities:

**Stage 1 — General chat** (~1 epoch, UltraChat + OpenHermes subset):
- Teaches instruction following and chat format.
- Does NOT introduce domain-specific data yet.

**Stage 2 — Code + Finance** (~2 epochs, CodeAlpaca + Magicoder + FinQA, with 10% Stage-1 replay):
- Specializes for the target domains.
- Replay prevents forgetting of Stage-1 behavior.

## DPO (optional, later)

```bash
python scripts/finetune/dpo.py \
    --config configs/train/dpo.yaml \
    --ckpt runs/khanh_1b_sft_code/checkpoints/trained_khanh_gpt_latest.pt \
    --preference-data data/sft/dpo_pairs.jsonl
```

Preference data format:
```json
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

## LoRA adapter management

```bash
# Save adapter after SFT
# (done automatically by sft.py — adapters saved to runs/<run>/adapters/)

# Merge adapter into base weights (creates a standalone model)
python scripts/finetune/lora_merge.py \
    --ckpt runs/khanh_1b/checkpoints/trained_khanh_gpt_latest.pt \
    --adapter runs/khanh_1b_lora_code/adapters/final/ \
    --output runs/khanh_1b_merged/

# Load a merged external HF model
python scripts/inference/load_external.py \
    --model Qwen/Qwen2.5-1.5B \
    --adapter runs/qlora_qwen15b_code/adapters/final/
```

## Why build this even before pretrain finishes

1. **Immediate usable model** — QLoRA on Qwen2.5-1.5B gives you a capable coding assistant in hours, not months.
2. **Pipeline validation** — proves the SFT flow works before you need it on your own weights.
3. **Future-proof** — after pretrain finishes, the same pipeline applies without changes.
4. **LoRA top-ups** — add a new language (Rust, Swift, etc.) with ~10 GPU-hours and a focused dataset.
