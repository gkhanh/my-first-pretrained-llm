# Plan: Professionalize KhanhLLM into a Small Coding-Focused LLM

## Context

You have a working 692M-parameter hybrid Transformer + MoE (KhanhLLM), pretrained on ~11.5% of C4 (1.15M rows / 730M tokens, loss 10.35 → 4.66) on what is presumably your current GPU. You want to:

1. Honestly assess the downsides of the current setup.
2. Restructure the repo into something professional.
3. Optimize training to fit an **RTX 5080 / 16 GB VRAM** target.
4. Add a **web-app skeleton** (no implementation yet) so the existing `generate_text.py` interactive tester can later become a real product.
5. Put a written design plan into a new `docs/` folder.
6. Long-term: build something useful for **coding tasks AND a simple finance chatbot**, plus a **local fine-tuning pipeline** that works on KhanhLLM and external open models.

This plan is organized so it can be executed in phases. Phase 0 is honest framing — read it before anything else, because it changes what "success" should look like.

---

## Phase 0 — Reality Check with Your Actual Compute Budget

**Your budget:** 2 full days/week (48 h) + 5 weekday evenings × 2 h (10 h) ≈ **58 h/week**, sustained for **2 years** = **~6,000 GPU-hours** on a single RTX 5080 (16 GB).

That changes the picture. It's still not Claude/GPT territory — those are 100B–2T params trained on trillions of tokens with thousands of accelerators — but it is **enough compute to train a genuinely useful small model from scratch**, not just a toy.

**Token budget at realistic throughput** (350M dense, bf16, SDPA, packed sequences, ~40k tokens/sec on a 5080):
- 6,000 h × 3,600 s × 40,000 tok/s ≈ **~860 B tokens** over 2 years.
- For reference: SmolLM2-360M was trained on ~4 T tokens, Qwen2.5-0.5B on similar. So you'd be at ~20–25 % of their token budgets — not parity, but in the same order of magnitude.
- For a 692M model, throughput halves → ~430 B tokens.
- For a hypothetical ~1 B model, ~250 B tokens — still real training, not a tech-demo.

**Recalibrated goals (all realistic in your budget):**

| Goal | Realistic? | Notes |
|---|---|---|
| Match Claude/GPT general ability | No | Hardware-bound. |
| A 350M–700M dense model that handles **code completion + FIM** on Python/JS/TS at the level of "useful intern, not yet senior" | **Yes** | Core deliverable. |
| The same model also being a **simple finance chatbot** (reads SEC-style text, answers basic questions, summarizes filings) | **Yes, with the right data mix** | Not a Bloomberg-GPT competitor, but a credible domain assistant. |
| Beating same-size open models (SmolLM2-360M, Qwen2.5-0.5B) from scratch | **Possibly within ~30 %**, with good data | Their main edge is more tokens, not better tricks. |
| A **local fine-tuning pipeline** that works on KhanhLLM **and** external open bases (Llama 3.x, Qwen 2.5, Mistral 7B with QLoRA) | **Yes, easily** | LoRA/QLoRA is exactly what 16 GB is good at. |

**Decisions locked in:**

1. **Pretraining: from-scratch only — TOP PRIORITY.** This is the main deliverable. Maximum quality architecture, maximum compute allocation.
2. **Existing 11.5 %-trained 692M checkpoint: dropped.** No `--legacy` resume path. Clean break — the new architecture is too different from the old one for resume to be meaningful, and you'd rather have a great model than a continuation of a compromised one.
3. **Fine-tuning pipeline: built but secondary.** Generic LoRA/QLoRA pipeline runs locally on the 5080 and works on KhanhLLM checkpoints **and** any HuggingFace-format causal-LM (Llama 3, Qwen 2.5, Mistral, Phi, etc.). It exists so (a) you have a usable model while KhanhLLM bakes, and (b) **after** pretrain finishes, you can keep improving your KhanhLLM with task-specific SFT/LoRA without retraining.
4. **Two domain capabilities in one model:** coding (primary) and finance chatbot (secondary). Both come from the data mix and SFT, not separate models.
5. **Tokenizer:** StarCoder2 (~49K vocab, code-aware, FIM tokens).
6. **Frontend:** Vite + React (skeleton only).
7. **Library policy:** every library pinned to its latest stable as of project start. PyTorch 2.x latest, transformers latest, bitsandbytes latest, peft latest, tokenizers latest, datasets latest, FlashAttention latest, triton latest. Pin via `pyproject.toml` with `>=` floors and a lockfile (`uv.lock` or `pip-tools`).
8. **Checkpoint naming:** `trained_khanh_gpt_step{N}_loss{X.XX}.pt` with periodic snapshots (every ~5 B tokens) plus a rolling `trained_khanh_gpt_latest.pt` symlink and a separate `trained_khanh_gpt_ema.pt` (EMA weights for inference). All under `runs/<run-name>/checkpoints/`.

**Smart staging at the 6,000-hour budget:**
- **Months 1–2 (~290 h):** repo restructure, modern architecture rebuild, tokenizer + data pipeline, 150M debug runs, fine-tuning pipeline (so you have an immediate-use model via QLoRA on Qwen2.5-Coder-1.5B while you wait).
- **Months 3–18 (~4,650 h):** full pretrain of `khanh_350m` on the code+finance+text mix. Targets ~700 B tokens — comfortably above the SmolLM2-360M / Qwen2.5-0.5B "minimum useful" threshold.
- **Months 19–22 (~750 h):** post-training: SFT (chat → code+finance staged), then DPO. Uses the same fine-tuning pipeline.
- **Months 23–24 (~290 h):** eval, polish, periodic LoRA top-ups for new domains as you find them.

**Contingency: only 3,000 hours available?**

| Tactic | Effect |
|---|---|
| Switch default to **`khanh_250m.yaml`** instead of 350M | More tokens-per-param. ~600 B tokens fit in 3,000 h, which is genuinely competitive at the 250M scale. |
| Cut SFT to single-stage (chat + code + finance mixed in one pass) | Saves ~150 h |
| Drop DPO | Saves ~100 h |
| Keep the fine-tuning pipeline | Cheap and high-leverage — don't drop this |

A 250M model trained on 600B tokens of a code/finance/text mix on the modern architecture is a real model — usable as a personal-use coder + simple finance assistant, fine-tunable for narrower tasks. Not "Claude," but **not a toy.**

The configs/ folder will ship both `khanh_350m.yaml` (6,000 h target) and `khanh_250m.yaml` (3,000 h target) so you can pick on the day you commit to a long run.

---

## Phase 1 — Downsides of the Current Implementation

### 1.1 Architecture problems (`models/khanh_llm.py`)

| # | Issue | Impact | Fix |
|---|---|---|---|
| 1 | **Learned positional embeddings**, capped at 2048 | No length extrapolation; can't grow context window without retraining | Switch to **RoPE** (Rotary Position Embedding) |
| 2 | **`nn.LayerNorm`** | Slower; not the modern standard | Switch to **RMSNorm** (used by Llama, Mistral, DeepSeek, Qwen) |
| 3 | **GELU FFN** | Lower quality vs modern activations | Switch to **SwiGLU** |
| 4 | **Post-norm residual**: `x = norm(x + sublayer(x))` at [models/khanh_llm.py:133](models/khanh_llm.py#L133) and [models/khanh_llm.py:144](models/khanh_llm.py#L144) | Known to be unstable at depth | Switch to **pre-norm**: `x = x + sublayer(norm(x))` |
| 5 | **`nn.MultiheadAttention`** with no GQA | Large KV cache → slow inference, bigger memory | Use **Grouped-Query Attention** (e.g. 16 Q-heads, 4 KV-heads) and **F.scaled_dot_product_attention** (FlashAttention-2 backend) |
| 6 | **MoE balance loss uses `argmax`** at [models/khanh_llm.py:52-58](models/khanh_llm.py#L52-L58) | `argmax` is non-differentiable → balance loss has weak/no gradient on the routing weights → the load balance signal is mostly cosmetic | Use the **Switch Transformer formulation**: aux_loss = N · Σ (fraction_tokens · prob_mass), where `fraction_tokens` comes from the *top-k* selection (not argmax) and is detached for the count but `prob_mass` keeps gradient |
| 7 | **`torch.histc` in the MoE forward** at [models/khanh_llm.py:55](models/khanh_llm.py#L55) | Slow on GPU, breaks `torch.compile` graph | Use `F.one_hot(...).sum(0)` or `scatter_add_` |
| 8 | **Per-expert Python loop** at [models/khanh_llm.py:78](models/khanh_llm.py#L78) | Eight serial CUDA dispatches per layer | Acceptable at E=8 for now; later move to grouped GEMM (Megablocks-style) |
| 9 | **No tied embeddings** between input/output | Wastes ~51 M parameters on the 50K vocab | Tie `output_layer.weight = token_embedding.weight` |
| 10 | **Activation checkpointing on every layer** at [models/khanh_llm.py:192](models/khanh_llm.py#L192) | Up to 30 % slowdown for memory you may not need under bf16 | Make checkpointing **selective** (every other layer or off entirely if VRAM allows after bf16 + GQA) |
| 11 | **Causal mask passed manually** as additive `-inf` matrix | Prevents PyTorch from picking the FlashAttention kernel automatically | Pass `is_causal=True` to SDPA instead of building the mask |
| 12 | **Mask cache is stored on `self`** at [models/khanh_llm.py:165-182](models/khanh_llm.py#L165-L182) | Becomes part of state_dict accidentally; brittle | Drop the cache once SDPA `is_causal=True` is used |

### 1.2 Training problems (`scripts/train.py`)

| # | Issue | Impact | Fix |
|---|---|---|---|
| 1 | **No mixed precision** | Training in fp32 on RTX 5080 leaves ~2× speed and ~50 % memory on the table | Use `torch.autocast(device_type='cuda', dtype=torch.bfloat16)` (RTX 5080 supports BF16 natively, no GradScaler needed) |
| 2 | **Batch size = 2, grad-accum = 64** at [scripts/train.py:25-61](scripts/train.py#L25-L61) | Effective batch is fine, but micro-batch wastes the 5080. With bf16+GQA you can fit 4–8 | Retune to micro-batch 4–8, accum 16–32, keep effective ≈128 |
| 3 | **No sequence packing** | Padding wastes compute on every batch; C4 docs vary wildly in length | Pack streamed tokens into fixed-length 2048 chunks; emit document-boundary IDs to reset attention if needed |
| 4 | **Tokenization happens inside the training loop** | Tokenizer becomes the bottleneck at high throughput | Pre-tokenize the corpus once into memory-mapped shards (`.bin` + index) — same approach as nanoGPT/litGPT |
| 5 | **Aux loss weight 0.01** | Switch Transformer paper uses 0.01 but with a *correctly-differentiable* loss. With the bug above, this number is meaningless | Re-tune **after** fixing the loss; start at 0.001–0.01 |
| 6 | **`torch.compile()` enabled but interaction with `checkpoint(use_reentrant=False)` and the `histc` op is fragile** | Possible recompiles or graph breaks; you may not be getting the speedup you think | Verify with `TORCH_LOGS=recompiles` and `torch.compile(mode='max-autotune')` after the architecture is fixed |
| 7 | **No LR sanity** | `5e-5` with cosine to `5e-6` is conservative; 690M models are usually trained at `3e-4` peak | Re-tune to ~`3e-4` peak, 2 k warmup steps, cosine to ~10 % |
| 8 | **Single optimizer.zero_grad()** placement assumes accum loop never errors mid-stream | Hard to debug, easy to leak grads | Use `optimizer.zero_grad(set_to_none=True)` and wrap the accum block defensively |
| 9 | **Vocab = 50 K, BPE on plain text C4** | Code-hostile: identifiers, indentation, symbols get split badly | **Adopt an existing battle-tested tokenizer** (StarCoder2 chosen as default — Apache-2.0, ~49K vocab, code-aware byte-level BPE, ships with FIM special tokens). Your custom 50K BPE is kept only for backward-compat with the legacy 692M checkpoint. |
| 10 | **No FIM (Fill-in-the-Middle) training objective** | FIM is what makes a model useful for editor autocomplete (PSM/SPM format) | Add FIM permutation in the data pipeline once base pretraining is healthy |
| 11 | **No eval during training** | You can't tell if loss-down means quality-up | Add periodic perplexity-on-held-out-set, plus HumanEval/MBPP small-N eval for the coding track |
| 12 | **No EMA / weight averaging** | Final checkpoint is noisier than it needs to be | Add EMA of weights for inference checkpoints |

### 1.3 Inference problems (`scripts/generate_text.py`)

| # | Issue | Impact | Fix |
|---|---|---|---|
| 1 | **No KV cache** | Every generated token re-runs the full forward over the whole prefix → O(n²) cost per token, 10–50× slower than necessary | Add a per-layer KV cache (`past_kv`); refactor attention to accept and update it |
| 2 | **Repetition penalty hand-rolled with a Python set** at [scripts/generate_text.py:145-153](scripts/generate_text.py#L145-L153) | Slow and not standard | Standard logits processor (HF `RepetitionPenaltyLogitsProcessor` or vectorized scatter) |
| 3 | **Triple-repeat stop heuristic** | Will misfire on legitimate code (`)))`, `]]]`, `===`, etc.) — actively harmful for a coding LLM | Drop it; rely on EOS + max length + nucleus sampling |
| 4 | **No streaming, no batching** | Limits future web-app perf | Generator design should yield tokens lazily |

### 1.4 Project structure problems

| # | Issue | Fix |
|---|---|---|
| 1 | One-file `models/khanh_llm.py` mixed with global constants | Make a real Python package under `src/khanh_llm/` |
| 2 | `scripts/` mixes data prep, training, eval, and an interactive tester | Split into `scripts/{data,train,eval,inference}/` |
| 3 | All hyperparameters in a Python `Config` class | Move to YAML configs (Hydra or plain `OmegaConf`) so you can run experiments without editing code |
| 4 | No `pyproject.toml`, no installable package | Add `pyproject.toml`, make `khanh_llm` pip-installable in editable mode |
| 5 | No tests beyond `test_causal_mask.py` | Add `tests/` with smoke tests for: forward pass, MoE balance loss, KV cache equivalence vs no-cache |
| 6 | No CI | Add a minimal GitHub Actions workflow (lint + smoke test) |
| 7 | `requirements.txt` lists only torch + bitsandbytes; everything else is implicit | Pin a real dependency set |
| 8 | No `docs/` | Created in this plan |
| 9 | Generated artifacts (`checkpoint_latest.pth.tar`, `training_log.csv`, `training_progress.png`) live at repo root | Move under `runs/` or `artifacts/`, gitignored |
| 10 | No data preparation pipeline | New `scripts/data/` with download → dedupe → tokenize → pack |

---

## Phase 2 — Target Repository Structure

```
building-llm/
├── pyproject.toml                  # installable package, deps, ruff/black/mypy config
├── README.md                       # rewritten: short overview + link to docs/
├── LICENSE
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml                  # lint + pytest on push
│
├── src/
│   └── khanh_llm/
│       ├── __init__.py
│       ├── config.py               # dataclasses for ModelConfig, TrainConfig, DataConfig
│       ├── model/
│       │   ├── __init__.py
│       │   ├── transformer.py      # KhanhLLM main class
│       │   ├── attention.py        # GQA + SDPA / FlashAttention
│       │   ├── moe.py              # fixed MoE w/ correct balance loss
│       │   ├── ffn.py              # SwiGLU
│       │   ├── norm.py             # RMSNorm
│       │   └── rope.py             # Rotary embeddings
│       ├── data/
│       │   ├── __init__.py
│       │   ├── tokenizer.py        # wraps tokenizers / HF tokenizer
│       │   ├── streaming.py        # streaming + sequence packing
│       │   ├── fim.py              # Fill-in-the-middle transform
│       │   └── shards.py           # memory-mapped pre-tokenized shards
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py          # main loop (bf16 autocast, grad accum, EMA)
│       │   ├── optim.py            # AdamW8bit, cosine+warmup
│       │   ├── checkpoint.py       # save/load, EMA weights
│       │   └── logging.py          # CSV + optional W&B
│       ├── inference/
│       │   ├── __init__.py
│       │   ├── generator.py        # KV-cache-backed sampler, streaming
│       │   └── samplers.py         # top-k, top-p, repetition penalty
│       ├── finetune/
│       │   ├── __init__.py
│       │   ├── lora.py             # PEFT/LoRA + QLoRA setup
│       │   ├── sft.py              # supervised fine-tuning loop (works for KhanhLLM AND external HF models)
│       │   ├── dpo.py              # preference tuning (later)
│       │   ├── chat_templates.py   # ChatML / Llama / Qwen template handling
│       │   └── adapters.py         # save/load/merge LoRA adapters
│       └── eval/
│           ├── __init__.py
│           ├── perplexity.py
│           ├── code_eval.py        # HumanEval / MBPP harness
│           └── finance_eval.py     # FinQA / ConvFinQA accuracy + qualitative prompts
│
├── configs/
│   ├── model/
│   │   ├── khanh_350m.yaml         # default for the 6,000-hour budget
│   │   ├── khanh_250m.yaml         # contingency for the 3,000-hour budget
│   │   └── khanh_150m.yaml         # debug / fast-iteration config
│   │   # NOTE: legacy 692M MoE config dropped per user decision — clean break
│   ├── train/
│   │   ├── pretrain_5080.yaml      # bf16, micro-batch 4-8, accum 16-32
│   │   ├── sft_code.yaml           # instruction tuning on code (CodeAlpaca, OSS-Instruct)
│   │   ├── sft_finance.yaml        # SFT on finance Q&A (FinQA, ConvFinQA)
│   │   ├── sft_chat.yaml           # general chat SFT (UltraChat / OpenHermes subset)
│   │   ├── lora_local.yaml         # LoRA on khanh_llm checkpoint
│   │   └── qlora_external.yaml     # QLoRA on external HF base (default: Qwen2.5-1.5B)
│   └── data/
│       ├── c4_text.yaml            # legacy run reproducibility
│       ├── pretrain_mix.yaml       # full pretrain mix: code + finance + general text
│       ├── code_only.yaml
│       └── finance_only.yaml       # SEC EDGAR + financial news + StackExchange Money/Quant
│
├── scripts/                        # thin entrypoints — logic lives in src/
│   ├── data/
│   │   ├── build_tokenizer.py      # legacy BPE rebuild (kept for reproducibility)
│   │   ├── download_starcoder_tokenizer.py
│   │   ├── prepare_pretrain_corpus.py    # download → dedupe → pack into .bin shards
│   │   ├── prepare_finance_corpus.py     # SEC EDGAR / news → cleaned text
│   │   ├── prepare_sft_dataset.py        # builds ChatML-formatted SFT jsonl
│   │   └── pack_fim.py                   # FIM SPM-format transform
│   ├── train/
│   │   └── pretrain.py             # `python -m scripts.train.pretrain --config ...`
│   ├── finetune/
│   │   ├── sft.py                  # full or LoRA SFT (works on KhanhLLM or external HF model)
│   │   ├── lora_merge.py           # merge LoRA adapter back into base weights
│   │   └── dpo.py
│   ├── eval/
│   │   ├── perplexity.py
│   │   ├── humaneval.py
│   │   └── finance_qa.py
│   └── inference/
│       ├── chat_cli.py             # current generate_text.py, cleaned up; supports KhanhLLM + external HF models
│       └── load_external.py        # helper to load Qwen/Llama/Mistral checkpoints into the same generator
│
├── web/                            # SKELETON ONLY — see Phase 4
│   ├── README.md
│   ├── backend/
│   │   ├── pyproject.toml
│   │   ├── app/
│   │   │   ├── __init__.py
│   │   │   ├── main.py             # FastAPI app stub
│   │   │   ├── routes/
│   │   │   │   ├── generate.py     # POST /v1/generate (stub)
│   │   │   │   └── health.py
│   │   │   ├── schemas.py          # pydantic request/response
│   │   │   └── model_service.py    # wraps khanh_llm.inference.generator
│   │   └── Dockerfile
│   └── frontend/
│       ├── README.md               # instructions: scaffold Vite + React app here
│       └── .gitkeep                # empty for now — `npm create vite@latest .` later
│
├── tests/
│   ├── test_model_forward.py
│   ├── test_moe_balance.py
│   ├── test_kv_cache.py
│   └── test_data_packing.py
│
├── docs/                           # NEW — see Phase 5
│   ├── 00-overview.md
│   ├── 01-architecture.md
│   ├── 02-training-recipe.md
│   ├── 03-data-pipeline.md
│   ├── 04-rtx5080-tuning.md
│   ├── 05-coding-llm-roadmap.md
│   ├── 06-web-app-design.md
│   └── 07-evaluation.md
│
├── runs/                           # gitignored — checkpoints, logs, plots
└── notebooks/                      # optional, for experimentation
```

**Migration of existing files:**

| Current location | New location |
|---|---|
| `models/khanh_llm.py` | rewritten into `src/khanh_llm/model/{transformer,attention,moe,ffn,norm,rope}.py` (clean rebuild, not a port — modern architecture) |
| `scripts/train.py` | logic → `src/khanh_llm/training/trainer.py`; entrypoint → `scripts/train/pretrain.py` |
| `scripts/build_tokenizer.py` | kept under `scripts/data/build_tokenizer.py` for reproducibility, but **not used** by the new pipeline (StarCoder2 tokenizer instead) |
| `scripts/generate_text.py` | logic → `src/khanh_llm/inference/generator.py`; CLI → `scripts/inference/chat_cli.py` |
| `scripts/plot_loss.py` | `scripts/eval/plot_loss.py` |
| `scripts/test_causal_mask.py` | `tests/test_causal_mask.py` |
| `checkpoint_latest.pth.tar` (3.9 GB) | **archived** to `runs/_archive_legacy_692m/` (gitignored), then deleted from repo root. Not used by new pipeline. |
| `training_log.csv` | archived to `runs/_archive_legacy_692m/` |
| `training_progress.png` | archived to `runs/_archive_legacy_692m/` |
| `khanh_tokenizer/` | archived to `runs/_archive_legacy_692m/` (kept on disk so the legacy CSV is interpretable) |

The existing 692M checkpoint is **preserved**. A small loader shim in `scripts/inference/chat_cli.py` should still be able to load it (with a `--legacy` flag) so your 11.5 %-of-C4 work doesn't get orphaned.

---

## Phase 3 — Optimized Training Recipe for RTX 5080 (16 GB)

### 3.1 Architecture (new default — from-scratch)

A **350M-param Llama-style decoder** is the right size to actually finish training on a single 5080 in reasonable time. Plus a **150M debug config** for fast iteration before committing to a long run.

**`khanh_350m.yaml` (new default)**

| Hyperparameter | Value | Rationale |
|---|---|---|
| Hidden dim | 1024 | Same as current — good ratio to heads |
| Layers | 24 | Deeper-narrower beats wider-shallower at this size |
| Attention heads (Q) | 16 | 64-dim per head |
| KV heads (GQA) | 4 | 4× smaller KV cache, ~free quality |
| FFN dim (SwiGLU) | 2752 | ~2.7× hidden, standard for SwiGLU |
| Norm | RMSNorm (pre-norm) | Modern standard |
| Position | RoPE (base=10000) | Length extrapolation |
| Vocab | **49,152 (StarCoder2 tokenizer)** | Code-aware, FIM tokens included |
| Tied embeddings | yes | Save ~50 M params |
| Context length | 2048 (train), extendable later via RoPE scaling | |
| **MoE** | **Off in v1** | MoE adds memory and instability you don't need at 350M; keep the implementation but disable for the new default config |

Total: ~330M dense params. Fits comfortably in 16 GB with bf16 and a meaningful batch.

**`khanh_150m.yaml` (debug / fast iteration)** — same shape, hidden=768, layers=12. Trains a usable checkpoint in hours, not days. Use it to validate every code change before kicking off the real 350M run.

**`khanh_692m_legacy.yaml`** — your current MoE architecture, frozen. Lets you resume the existing 11.5%-trained checkpoint with the legacy tokenizer + legacy data path.

### 3.2 Memory + speed tactics for 16 GB VRAM

| Tactic | Setting |
|---|---|
| Mixed precision | **bf16 autocast** (no GradScaler needed) |
| Optimizer | **AdamW8bit** (already in use) |
| Attention | `F.scaled_dot_product_attention` with `is_causal=True` → FlashAttention-2 backend |
| GQA | 4 KV heads vs 16 Q heads |
| Activation checkpointing | **Selective**: every other transformer block, or off if VRAM allows |
| `torch.compile` | `mode='reduce-overhead'` for inference, default for train; verify no graph breaks |
| TF32 matmul | already enabled |
| Micro-batch | start 4, push to 8 if VRAM allows (after bf16 + GQA gains) |
| Grad accum | 32 → effective batch 128 sequences × 2048 tokens = 262 K tokens/step |
| Sequence packing | yes — concatenate documents into 2048 chunks |
| Pre-tokenized shards | yes — `.bin` files mmap-loaded |

### 3.3 Optimizer + schedule

| Field | Value |
|---|---|
| Peak LR | 3e-4 |
| Min LR | 3e-5 (~10 % of peak) |
| Warmup | 2000 steps (linear from 0) |
| Decay | Cosine to min |
| Betas | (0.9, 0.95) |
| Eps | 1e-8 |
| Weight decay | 0.1 (decoupled, AdamW) |
| Grad clip | 1.0 |
| EMA | decay 0.999 over weights, used only for eval/inference checkpoints |

### 3.3a Checkpointing strategy

- **Naming:** `trained_khanh_gpt_step{step:08d}_loss{loss:.3f}.pt`
- **Snapshot frequency:** every 5 B tokens of training (≈ every ~35 h on 5080) — gives ~140 snapshots over the 6,000 h budget. Old snapshots beyond a rolling 10 are auto-pruned to save disk; every 10th snapshot is kept permanently as a "milestone."
- **Always-current pointer:** `runs/<run>/checkpoints/trained_khanh_gpt_latest.pt` is a symlink to the most recent.
- **EMA inference weights:** `runs/<run>/checkpoints/trained_khanh_gpt_ema.pt` (separate, smaller — just weights, no optimizer state).
- **What's inside each checkpoint:** model weights, optimizer state, scheduler state, RNG state (torch + numpy + cuda), data shard cursor, step count, EMA state, run config — i.e. fully resumable + reproducible.
- **Format:** raw PyTorch `.pt` (`torch.save`) — simple, durable, no special framework dep. Plus a `safetensors` export (`trained_khanh_gpt_step{N}.safetensors`) of weights-only at every milestone, for safer sharing.
- **Post-pretrain compatibility:** weights are stored in HuggingFace-compatible `state_dict` shape (clean key names, no `_orig_mod.` torch.compile prefix — strip on save). This means after pretrain finishes, the same checkpoints load directly into the LoRA / SFT pipeline without conversion.

### 3.3b Library / framework versions

Pin to **latest stable as of project start** for everything. Concrete approach:
- Use `pyproject.toml` with `>=` floors for direct deps + a `uv.lock` (or `pip-tools`-generated `requirements.lock`) for fully reproducible installs.
- Direct dependencies (latest stable): `torch`, `transformers`, `tokenizers`, `datasets`, `bitsandbytes`, `peft`, `accelerate`, `safetensors`, `flash-attn`, `triton`, `sse-starlette`, `fastapi`, `uvicorn`, `pydantic`, `omegaconf`, `pyyaml`, `pandas`, `matplotlib`, `pytest`, `ruff`, `mypy`.
- CI runs `uv pip install --resolution=highest` weekly to surface upgrade opportunities.
- One sanity rule: do not chase bleeding-edge nightlies for `torch` mid-pretrain — pin to a specific stable for the duration of any long run, then upgrade between runs.

### 3.4 Pretraining data mix (code + finance + general text)

For a model that must do **both** code and finance with simple chat ability, the corpus must reflect both. Suggested mix:

| Source | Weight | Notes |
|---|---|---|
| The Stack v2 (filtered: Python / JS / TS / Go / Rust) | 50 % | Permissively-licensed code; core of the coding ability |
| StackExchange — programming (Stack Overflow, Code Review) | 8 % | Q&A format, very high signal |
| Markdown / docs from The Stack | 7 % | Teaches the model to write READMEs and explanations |
| **SEC EDGAR 10-K / 10-Q / 8-K filings** (filtered subset) | **8 %** | Core finance domain text |
| **Financial news** (FNSPID or similar permissively-licensed dataset) | **5 %** | Current-events finance language |
| **StackExchange — Money / Quant / Personal Finance** | **2 %** | Finance Q&A format |
| Wikipedia (English) | 10 % | General world knowledge so the model can read instructions |
| C4 (small slice) | 10 % | General web English |

Total: 100 %. Approximately **65 % code, 15 % finance, 20 % general text**. Tunable via `configs/data/pretrain_mix.yaml` — if early eval shows finance is too weak, raise the finance weight to 25 %.

**FIM transform**: applied with probability ~0.5 to the code slice in the SPM (suffix-prefix-middle) format. Use the StarCoder2 tokenizer's existing FIM special tokens (`<fim_prefix>`, `<fim_middle>`, `<fim_suffix>`) — no need to invent your own. Required for editor autocomplete.

**Document boundaries**: every packed sequence resets attention at document boundaries (no leakage across documents) — important for the finance slice especially, since 10-K filings shouldn't blend into each other.

### 3.5 Evaluation

Add to every checkpoint:
- Held-out perplexity **per data slice separately** (code / finance / general) — so you can see if finance is improving without code regressing, and vice versa.
- **HumanEval** / **MBPP** pass@1 for code (small, runs in minutes on the 5080).
- **FinQA** / **ConvFinQA** accuracy for finance (small numeric-reasoning sets over financial reports).
- A fixed suite of ~10 qualitative prompts (5 code, 5 finance) saved to `runs/<run>/samples/<step>.md`.

You currently have *no* feedback signal beyond loss — adding even small evals will tell you whether the model is actually getting smarter, and which capability is leading or lagging.

---

## Phase 3.6 — Local Fine-Tuning Pipeline (works for KhanhLLM AND external models)

This is the second deliverable: a generic, **fully-local** fine-tuning pipeline that lives at `src/khanh_llm/finetune/` and `scripts/finetune/`. It supports two modes:

### Mode A: Fine-tune KhanhLLM (your own checkpoints)
- Loads from `runs/<run>/checkpoints/`.
- Uses the same model/tokenizer code as pretraining.
- Supports full fine-tuning (small models) or LoRA (larger configs).

### Mode B: Fine-tune external HuggingFace base models
- Loads any HF causal-LM (Llama 3.x, Qwen 2.5, Mistral 7B, Phi-3, etc.) via `transformers.AutoModelForCausalLM`.
- Uses **QLoRA** (4-bit base + LoRA adapters) so a 7B model fits in 16 GB during training.
- Adapters saved separately from base weights — `scripts/finetune/lora_merge.py` can later merge them into a single deployable model.

### Shared infrastructure (used by both modes)
- **ChatML chat templating** in `src/khanh_llm/finetune/chat_templates.py` — also supports Llama-3-instruct and Qwen-instruct templates so external models keep their native formatting.
- **SFT loop** in `src/khanh_llm/finetune/sft.py` — bf16 autocast, gradient accumulation, EMA, eval hooks. Same shape as the pretrain trainer; deliberately reuses the optimizer + scheduler code.
- **DPO** in `dpo.py` — added later, after SFT works. Reads preference pairs from a JSONL file.
- **Adapter management** in `adapters.py` — load/save/merge LoRA via `peft`.

### Dependencies it adds
- `peft>=0.10` (LoRA, QLoRA)
- `bitsandbytes` (already present, used for 4-bit base model loading too)
- `transformers` (already implicit, made explicit)
- `trl` (optional — only if you want their `SFTTrainer` / `DPOTrainer` shortcuts; the plan otherwise uses our own loop)

### Recommended SFT datasets (all locally cached after first download)

| Capability | Dataset | Size |
|---|---|---|
| Code instruction | OSS-Instruct (Magicoder), CodeAlpaca-20k | small, ~75 MB |
| Code FIM (already covered in pretrain) | — | — |
| Finance Q&A | FinQA, ConvFinQA, FiQA-2018 | small, ~50 MB |
| General chat | UltraChat-200k subset, OpenHermes-2.5 (filtered) | medium, ~1 GB |
| Refusals / safety basics | A small hand-curated jsonl of "I don't know" examples | tiny |

Mix and SFT in **two stages** to avoid forgetting: (1) general chat first (~1 epoch), (2) code + finance second (~2 epochs), with a small replay of stage-1 data interleaved.

### Why "all local" works fine
- 7B QLoRA training fits in 16 GB.
- Datasets above are all <2 GB total.
- Inference of a fine-tuned 7B in 4-bit fits in ~5 GB.
- No cloud, no API key, no telemetry. The only network call is the one-time HuggingFace dataset / model download.

### Why this matters even after pretrain finishes
The same pipeline applied to your finished `trained_khanh_gpt` lets you:
- Add a new programming language (LoRA on a focused corpus, ~10 GPU-hours).
- Sharpen finance behavior with a few hundred curated Q&A pairs.
- Add a new chat persona without retraining.
- Experiment with DPO / RLAIF on top of SFT.

This is the "your model can keep getting better forever" path. The pretrain run is the foundation; the LoRA pipeline is how you keep building on it.

---

## Phase 4 — Web-App Skeleton (no implementation)

Goal: lay out folders + minimal stubs so a future contributor (or you, later) can pick up `web/` as a standalone project without touching the training code.

### 4.1 What gets created in `web/`

- **`web/README.md`** — explains the architecture: FastAPI backend that loads `khanh_llm.inference.generator`, exposes `POST /v1/generate` and `POST /v1/chat` with SSE streaming. Frontend is a Vite + React SPA that talks to the backend over HTTP/SSE.
- **`web/backend/pyproject.toml`** — separate package, depends on `khanh_llm` + FastAPI + uvicorn + sse-starlette.
- **`web/backend/app/main.py`** — FastAPI app with routes wired and CORS configured for the Vite dev server (`http://localhost:5173`); endpoint bodies stub out with `raise NotImplementedError("see docs/06-web-app-design.md")`.
- **`web/backend/app/routes/generate.py`** — endpoint signature + pydantic schemas, no logic.
- **`web/backend/app/routes/health.py`** — actual `/health` endpoint (trivial, useful as a smoke test).
- **`web/backend/app/schemas.py`** — `GenerateRequest`, `GenerateResponse`, `ChatMessage` pydantic models.
- **`web/backend/app/model_service.py`** — class with `load()`, `generate()`, `stream_generate()` method *signatures only*.
- **`web/backend/Dockerfile`** — multi-stage, CUDA base image, installs `khanh_llm` + backend.
- **`web/frontend/README.md`** — explicit setup instructions: `npm create vite@latest . -- --template react-ts`, then `npm install` and `npm run dev`. Notes the expected backend URL (`VITE_API_URL=http://localhost:8000`) and points at the `/v1/generate` SSE contract documented in `docs/06-web-app-design.md`.
- **`web/frontend/.gitkeep`**.

### 4.2 What does NOT happen in this phase

- No real generation logic.
- No frontend code.
- No streaming implementation.
- No auth / rate limiting.
- No deployment config beyond the Dockerfile stub.

The goal is purely structural — when you (or the future "separate application") starts, the wiring is already designed.

---

## Phase 5 — Contents of `docs/`

All documents live in `/media/giakhanh/Storage1/projects/building-llm/docs/`. Each is a single Markdown file kept under ~300 lines so it's actually readable.

| File | Contents |
|---|---|
| **00-overview.md** | One-page intro: what this repo is, two tracks (from-scratch KhanhLLM, continued-pretrain coder), how the folders map to the work |
| **01-architecture.md** | Detailed model spec (the new 350M Llama-style + the legacy 692M MoE), diagrams, `state_dict` shape table |
| **02-training-recipe.md** | The full Phase-3.3 recipe: bf16, AdamW8bit, LR schedule, grad accum math, expected throughput, checkpointing strategy |
| **03-data-pipeline.md** | Streaming → tokenizing → packing → FIM transform → sharded `.bin` format; reproducibility notes (seed, shard ordering) |
| **04-rtx5080-tuning.md** | VRAM budget worksheet (params + grads + optimizer state + activations + KV), how to dial micro-batch / accum / checkpointing for the 16 GB ceiling, profiler walkthrough |
| **05-roadmap.md** | Phase-by-phase plan keyed to your **2-year / ~6,000 GPU-hour budget**. Months 1–2 setup; 3–14 pretrain `khanh_350m`; 15–18 SFT (chat + code + finance); 19–24 optional larger run + DPO. Explicit milestones with token targets. From-scratch only for pretrain; finetuning supports external bases too. |
| **06-web-app-design.md** | API contract for `/v1/generate` and `/v1/chat`, SSE streaming format, request validation, expected latency budget, future ideas (batching, vLLM integration). Vite + React frontend wireframe. |
| **07-evaluation.md** | What "good" means: perplexity targets per data slice, HumanEval/MBPP target ranges for our size class, FinQA/ConvFinQA targets, qualitative prompt suite |
| **08-finetuning.md** | The local fine-tuning pipeline (Phase 3.6): LoRA / QLoRA, ChatML templating, dataset format, how to point it at KhanhLLM checkpoints, how to point it at an external HuggingFace base (Qwen / Llama / Mistral), adapter merging, expected VRAM for 0.5B–7B bases, full local-only workflow |
| **09-finance-domain.md** | Finance data sources (SEC EDGAR, FNSPID, StackExchange Money/Quant), filtering and licensing notes, FinQA / ConvFinQA eval setup, how the chatbot is expected to behave (and what it should refuse — no investment advice) |

The README.md at the repo root gets shortened to a 1-screen overview and links into `docs/`.

---

## Phase 6 — Suggested Implementation Order

The user explicitly asked for: **(a)** restructure first, **(b)** web skeleton, **(c)** algorithm + training improvements. That order is preserved.

| Step | Output | Approx effort |
|---|---|---|
| 1. Create `docs/` with all seven docs from Phase 5 (plan the work in writing first) | docs land | small |
| 2. Add `pyproject.toml`, real `requirements.txt`, `.github/workflows/ci.yml` | installable repo | small |
| 3. Create `src/khanh_llm/` package layout; **move** existing model + scripts in (no behavior change yet, just relocation + import-path fixes) | Phase-2 layout exists, old code still runs | medium |
| 4. Create `web/` skeleton per Phase 4 | scaffold ready for future work | small |
| 5. Add `tests/` with smoke tests against the *moved-but-unchanged* code | green CI on legacy behavior | small |
| 6. Refactor model: RMSNorm, RoPE, SwiGLU, GQA, SDPA `is_causal=True`, pre-norm, tied embeddings, fixed MoE balance loss | new `khanh_350m.yaml` config trains | large |
| 7. Refactor training loop: bf16 autocast, sequence packing, pre-tokenized shards, EMA, eval hooks | actual 5080-tuned recipe runs | large |
| 8. Add KV-cache inference + streaming generator | `chat_cli.py` ~10–50× faster | medium |
| 9. Adopt StarCoder2 tokenizer; build code+finance+text data pipeline (Stack v2 + SEC EDGAR + Wikipedia + C4, FIM transform, sharded `.bin` files) | full pretrain corpus ready | large |
| 10. Smoke-train `khanh_150m.yaml` for ~1 hour on the new corpus to validate the whole pipeline end-to-end before kicking off `khanh_350m.yaml` for real | confidence to start the long run | small |
| 11. Build the local fine-tuning pipeline (`src/khanh_llm/finetune/`): LoRA + QLoRA, ChatML templating, SFT loop reusing the pretrain trainer; smoke-test by QLoRA-fine-tuning Qwen2.5-0.5B on a tiny SFT mix | reusable finetune pipeline that handles both KhanhLLM and external HF models | medium |
| 12. Kick off the long pretrain (`khanh_350m.yaml`) — the months-3-to-14 run | the actual model | the run itself |
| 13. Once pretrain is done, run staged SFT (general chat → code+finance) with the pipeline from step 11 | usable chatbot + coder | medium |

Each step should be a separate PR / commit so the existing 11.5 %-trained checkpoint remains usable until the architecture changes (step 6 breaks compat — handled by the `--legacy` loader).

---

## Critical Files to Modify or Create

**Heavy modifications (rewrite):**
- [models/khanh_llm.py](models/khanh_llm.py) — split into the `src/khanh_llm/model/` modules
- [scripts/train.py](scripts/train.py) — split between `src/khanh_llm/training/` and `scripts/train/pretrain.py`
- [scripts/generate_text.py](scripts/generate_text.py) — split between `src/khanh_llm/inference/` and `scripts/inference/chat_cli.py`

**Light modifications:**
- [README.md](README.md) — shorten, link to `docs/`
- [requirements.txt](requirements.txt) — add transformers, datasets, tokenizers, pyyaml/omegaconf, fastapi, uvicorn, pytest, ruff
- [.gitignore](.gitignore) — add `runs/`, `artifacts/`, `wandb/`, `.venv/`, `dist/`, `*.egg-info/`

**New files (representative, not exhaustive):**
- `pyproject.toml`
- `docs/00-overview.md` … `docs/07-evaluation.md`
- `src/khanh_llm/__init__.py` and the full module tree under Phase 2
- `configs/model/khanh_350m.yaml`, `configs/train/pretrain_5080.yaml`, `configs/data/code_mix.yaml`
- `web/backend/app/main.py` + stubs
- `tests/test_model_forward.py`, `tests/test_moe_balance.py`, `tests/test_kv_cache.py`
- `.github/workflows/ci.yml`

**Existing utilities to reuse (do not reimplement):**
- `bitsandbytes.optim.AdamW8bit` — already used at [scripts/train.py:580-585](scripts/train.py)
- The CSV logging structure in `scripts/train.py` — keep, just move
- The interactive REPL in `scripts/generate_text.py` — keep the UX, swap the engine for the KV-cache one
- The BPE training in `scripts/build_tokenizer.py` — keep as a fallback option even if you adopt an external tokenizer

---

## Verification Plan

After each phase, run:

| Check | Command | Expected |
|---|---|---|
| Package installs | `pip install -e .` | success |
| Lint + tests | `ruff check . && pytest -q` | green |
| Legacy checkpoint still loads | `python scripts/inference/chat_cli.py --legacy --ckpt runs/khanh_692m/checkpoints/latest.pth.tar` | generates text (badly, as expected) |
| Forward-pass smoke | `pytest tests/test_model_forward.py` | passes for both 350M and 692M configs |
| MoE balance gradient flows | `pytest tests/test_moe_balance.py` | non-zero gradient on `gate.weight` from aux_loss |
| KV cache equivalence | `pytest tests/test_kv_cache.py` | KV-cache output matches no-cache to within 1e-4 |
| 5080 training step fits | `python scripts/train/pretrain.py --config configs/train/pretrain_5080.yaml --max-steps 10` | 10 steps complete, `nvidia-smi` shows < 15 GB used |
| Web stub serves | `cd web/backend && uvicorn app.main:app` then `curl localhost:8000/health` | `{"status":"ok"}` |
| Eval harness runs | `python scripts/eval/perplexity.py --ckpt ... --data ...` | prints number |
| HumanEval (later) | `python scripts/eval/humaneval.py --ckpt ...` | reports pass@1 |

End-to-end "did this actually work" test: run `scripts/train/pretrain.py` for ~1 hour on the 5080, confirm bf16 autocast is on (`torch.cuda.memory_allocated()` ≈ half of fp32 baseline), check `nvidia-smi` for VRAM headroom, and check `runs/.../samples/` for generated text from the EMA checkpoint.

---

## Decisions Locked In (final)

1. **Pretraining: from-scratch only — TOP PRIORITY.** Maximum-quality modern architecture.
2. **Existing 11.5%-trained 692M checkpoint: dropped.** Clean break. Old artifacts archived under `runs/_archive_legacy_692m/`, deleted from repo root.
3. **Fine-tuning pipeline: built early, secondary priority.** Lets you (a) have a usable model immediately via QLoRA on an external base while KhanhLLM pretrains, (b) keep improving your KhanhLLM after pretrain finishes.
4. **Two domain capabilities in one model:** coding (primary) + finance chatbot (secondary).
5. **Compute budget:** ~6,000 GPU-hours over 2 years (with `khanh_350m` config). Contingency: ~3,000 hours → switch to `khanh_250m` config, still produces a real (not toy) model.
6. **Tokenizer:** StarCoder2 (~49K vocab, code-aware, FIM tokens).
7. **Library policy:** latest stable across the board, pinned via `pyproject.toml` + lockfile.
8. **Checkpoint naming:** `trained_khanh_gpt_step{N}_loss{X.XX}.pt` with rolling latest + permanent milestones + EMA + safetensors export.
9. **Web frontend:** Vite + React (skeleton only).

## Still Open (decide during execution, not blockers)

- W&B integration in the trainer, or stay CSV-only? (Default: CSV-only — keeps "all local" promise; W&B is opt-in.)
- Which subset of languages to filter The Stack v2 down to? (Default: Python / JS / TS / Go / Rust — easy to change in `configs/data/pretrain_mix.yaml`.)
- Whether to enable MoE in the new `khanh_350m.yaml` default. (Default: off; dense 350M is the recommended starting point. The fixed MoE implementation stays available for `khanh_700m.yaml` later.)
- Default external base for the QLoRA pipeline smoke test: Qwen2.5-1.5B vs Llama-3.2-1B. (Default: Qwen2.5-1.5B — smaller-vocab tokenizer, well-trained on code+text.)
- Whether finance content should include any non-permissive sources. (Default: no — stick to SEC EDGAR public filings + permissively-licensed news, to keep the model legally redistributable.)
