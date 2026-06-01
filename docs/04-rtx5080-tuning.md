# RTX 5080 VRAM Tuning Guide

## VRAM budget breakdown

For `khanh_1b` (~330M params) in BF16 training:

| Component | Formula | Value |
|---|---|---|
| Model weights (BF16) | 330M × 2 bytes | ~660 MB |
| Gradients (BF16) | 330M × 2 bytes | ~660 MB |
| AdamW8bit states (INT8) | 330M × 1 byte × 2 | ~660 MB |
| Activations (micro-batch=4, seq=2048, BF16) | depends on checkpointing | ~2–6 GB |
| KV cache (inference only) | — | — |
| CUDA kernels + fragmentation overhead | — | ~1–2 GB |
| **Total (no activation checkpointing)** | | **~5–10 GB** |
| **Total (with selective checkpointing)** | | **~4–8 GB** |

At micro-batch=4: typically **8–12 GB** during training. At micro-batch=8: **12–15 GB**. The 5080's 16 GB gives ~1–4 GB headroom.

## How to dial settings

### Step 1: Start conservative
```yaml
# configs/train/pretrain_5080.yaml
micro_batch_size: 4
gradient_accumulation_steps: 32
activation_checkpointing: "selective"  # every other block
```

### Step 2: Profile peak VRAM
```python
# Add to pretrain.py --max-steps 10
torch.cuda.reset_peak_memory_stats()
# ... training steps ...
peak = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak VRAM: {peak:.2f} GB")
```

Or watch with:
```bash
watch -n 0.5 nvidia-smi
```

### Step 3: Tune micro-batch
- If peak < 12 GB → try micro-batch=8 (halve accum steps to keep effective batch=128).
- If peak > 15 GB → drop to micro-batch=2 or turn on full activation checkpointing.

### Step 4: Activation checkpointing options
| Mode | VRAM saved | Speed cost |
|---|---|---|
| `off` | 0 | 0% |
| `selective` (every other block) | ~30% | ~10–15% |
| `full` (every block) | ~50% | ~25–30% |

## Common VRAM issues

### Out of Memory (OOM) during backward
- Cause: activations accumulate across grad-accum steps.
- Fix: reduce micro-batch first; if still OOM, enable `selective` checkpointing.

### Sudden OOM after stable training
- Cause: `torch.compile` graph recompile triggered by variable sequence length.
- Fix: ensure all batches are exactly `seq_len=2048` (packing handles this).
- Verify: `TORCH_LOGS=recompiles python scripts/train/pretrain.py --max-steps 5`

### Reserved vs allocated memory gap
- `torch.cuda.memory_reserved()` >> `torch.cuda.memory_allocated()` means fragmentation.
- Fix: add `torch.cuda.empty_cache()` between grad-accum steps (small cost, big headroom gain in some configs).

## torch.compile interaction

```python
# Training — default mode (good balance)
model = torch.compile(model)

# Inference — reduce-overhead (better for fixed-size batches)
model = torch.compile(model, mode="reduce-overhead")
```

Do NOT use `mode='max-autotune'` until architecture is fully stable — it has a very long warm-up that burns VRAM.

## BF16 specifics on RTX 5080

- BF16 tensor cores are natively supported (no emulation).
- BF16 has a larger dynamic range than FP16 → no gradient underflow → no `GradScaler`.
- Matmul and attention run in BF16; layer norm accumulation is FP32 (handled by autocast automatically).
- `torch.set_float32_matmul_precision('high')` enables TF32 for the FP32 fallback matmuls in BF16 training → free speed.

## Quick profiler walkthrough

```python
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=tensorboard_trace_handler("./runs/profile"),
    record_shapes=True,
    with_stack=True,
) as prof:
    for step in range(10):
        loss = train_step(...)
        prof.step()
```

Then: `tensorboard --logdir ./runs/profile`

Look for:
- Long CUDA idle gaps → CPU bottleneck (tokenization, data loading)
- Large memory spikes → activation accumulation
- Frequent recompiles → compile graph instability

## Target numbers

| Metric | Target |
|---|---|
| Peak VRAM (micro-batch 4) | < 12 GB |
| Peak VRAM (micro-batch 8) | < 15 GB |
| Throughput (khanh_1b, micro-batch 4) | ≥ 35,000 tok/s |
| GPU utilization | ≥ 85% |
| Time to 5B tokens | ≤ 40 hours |
