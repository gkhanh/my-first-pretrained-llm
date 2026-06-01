# Training Recipe — RTX 5080 Optimized

## Optimizer + schedule

| Hyperparameter | Value | Rationale |
|---|---|---|
| Optimizer | AdamW8bit (`bitsandbytes`) | 8-bit states halve optimizer memory |
| Peak LR | `3e-4` | Standard for 300M-scale models (Llama, Chinchilla) |
| Min LR | `3e-5` (10% of peak) | Cosine tail, not zero |
| Warmup | 2000 steps, linear from 0 | Stable start for large LR |
| Decay | Cosine to min LR | Smooth, well-tested |
| Betas | `(0.9, 0.95)` | Standard for LLMs |
| Eps | `1e-8` | |
| Weight decay | `0.1` (decoupled AdamW) | Regularization |
| Grad clip | `1.0` | Prevents gradient explosions |
| EMA decay | `0.999` | Smoother inference weights |

## Mixed precision

- `torch.autocast(device_type='cuda', dtype=torch.bfloat16)` wraps the forward pass.
- RTX 5080 has native BF16 tensor cores → no `GradScaler` needed (BF16 doesn't suffer from FP16 underflow).
- Optimizer states are kept in FP32 by AdamW8bit internally.
- `torch.set_float32_matmul_precision('high')` enables TF32 on matmuls.

## Batch size and gradient accumulation

| Setting | Value |
|---|---|
| Micro-batch | 4 (push to 8 after VRAM profiling) |
| Gradient accumulation steps | 32 |
| Effective batch size | 128 sequences |
| Tokens per step | 128 × 2048 = **262,144 tokens** |

> Start at micro-batch=4. After validating VRAM headroom with `nvidia-smi`, try 8. Target: < 14 GB used during the forward+backward of one accumulation step.

## Sequence packing

Instead of padding variable-length documents to 2048, tokens from consecutive documents are packed into fixed-length 2048-token chunks:

```
[doc1_tok...] [EOS] [doc2_tok...] [EOS] [doc3_tok...]
                                        ^^^^ chunk boundary if needed
```

- No padding tokens → 100% of compute goes to real tokens.
- Document boundaries tracked with a `position_ids` reset (or an attention bias) so attention does not leak across documents.
- Implemented in `src/khanh_llm/data/streaming.py`.

## Pre-tokenized shards

- Corpus is tokenized once, offline, into binary `.bin` shards.
- Each shard: a flat array of uint16 token IDs + index file.
- Training loop memory-maps shards (`np.memmap`) → no tokenizer in the hot path.
- Shard size: ~500M tokens each (configurable).
- Implemented in `src/khanh_llm/data/shards.py`.

## FIM (Fill-in-the-Middle)

- Applied with probability 0.5 to the code slice only.
- SPM format: `<fim_prefix>PREFIX<fim_suffix>SUFFIX<fim_middle>MIDDLE`.
- Uses StarCoder2 tokenizer's built-in FIM special tokens.
- Implemented in `src/khanh_llm/data/fim.py`.

## MoE aux loss (new formulation)

When MoE layers are active, the balance loss uses the Switch Transformer formulation:

```
aux_loss = N × Σᵢ (fᵢ × pᵢ)
```

Where:
- `fᵢ` = fraction of tokens routed to expert `i` (from top-k selection, **detached**)
- `pᵢ` = average router probability for expert `i` (**keeps gradient**)
- `N` = number of experts

Weight: `aux_loss_weight = 0.01` (re-tune if experts collapse).

The old formulation used `argmax` for `fᵢ`, which had no gradient signal on the router weights. The new one uses the top-k mask (scatter/one-hot), which is differentiable w.r.t. `pᵢ`.

## torch.compile

```python
model = torch.compile(model, mode='reduce-overhead')  # inference
model = torch.compile(model)                           # training (default mode)
```

Verify no graph breaks with:
```bash
TORCH_LOGS=recompiles python scripts/train/pretrain.py --config configs/train/pretrain_5080.yaml --max-steps 5
```

## Checkpointing strategy

See [`01-architecture.md`](01-architecture.md) for `state_dict` key conventions.

| Artifact | Location | Frequency |
|---|---|---|
| Full checkpoint (weights + optimizer + scheduler + RNG + data cursor) | `runs/<run>/checkpoints/trained_khanh_gpt_step{N:08d}_loss{loss:.3f}.pt` | Every ~5B tokens (~35h on 5080) |
| EMA weights only | `runs/<run>/checkpoints/trained_khanh_gpt_ema.pt` | Updated every step, saved every snapshot |
| Latest symlink | `runs/<run>/checkpoints/trained_khanh_gpt_latest.pt` | Always points to newest full checkpoint |
| Safetensors export | `runs/<run>/checkpoints/trained_khanh_gpt_step{N}.safetensors` | Every 10th snapshot (permanent milestone) |

Rolling pruning: keep the 10 most recent full checkpoints; every 10th is kept permanently. This caps disk usage at roughly 10 × checkpoint_size.

## Expected throughput on RTX 5080

| Model | Batch setup | Approx tok/s |
|---|---|---|
| khanh_150m (debug) | micro-batch 8, accum 16 | ~60,000 |
| khanh_1b | micro-batch 4, accum 32 | ~40,000 |
| khanh_1b | micro-batch 8, accum 16 | ~45,000 |

At 40,000 tok/s: **~144B tokens/year** → 390B tokens ≈ 4.9 years at this rate with 24/7 usage. Realistically with ~58h/week uptime: **390B tokens in ~18 months** (Months 3–22 of the roadmap).

## Key signals to watch during training

```
loss          → should fall from ~10 (random) toward 2.5–3.0 by 50B tokens
aux_loss      → should stay near 0 (not spike); if experts collapse, raise aux_loss_weight
lr            → verify warmup ramp and cosine tail in training_log.csv
VRAM used     → target < 14 GB; check with nvidia-smi or torch.cuda.memory_allocated()
tok/s         → should be stable; a sudden drop = graph recompile or OOM recovery
```
