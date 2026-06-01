# Architecture — KhanhLLM

## New default: `khanh_1b` (Llama-style dense decoder)

The legacy 692M hybrid Transformer+MoE (trained on 11.5% of C4) has been archived. The new default is a clean, modern architecture from scratch.

### Model specification

| Hyperparameter | `khanh_1b` (default) | `khanh_15b_moe` (stretch) | `khanh_700m` (contingency) | `khanh_150m` (debug) |
|---|---|---|---|---|
| Hidden dim | 2048 | 2048 | 1536 | 768 |
| Layers | 22 | 22 | 20 | 12 |
| Q heads | 16 | 16 | 16 | 12 |
| KV heads (GQA) | 4 | 4 | 4 | 4 |
| FFN dim (SwiGLU) | 5504 | 5504 | 4096 | 2048 |
| Vocab | 49,152 (StarCoder2) | 49,152 | 49,152 | 49,152 |
| Context length | 2048 | 2048 | 2048 | 2048 |
| Norm | RMSNorm (pre-norm) | RMSNorm | RMSNorm | RMSNorm |
| Position | RoPE (base=500000) | RoPE (base=500000) | RoPE (base=500000) | RoPE |
| Tied embeddings | Yes | Yes | Yes | Yes |
| MoE | Off | Every other FFN (E=8, K=2) | Off | Off |
| Approx params | ~1B | ~1.5B total / ~700M active | ~700M | ~160M |
| VRAM (train, bf16) | ~10.5 GB | ~13.5 GB | ~8 GB | ~3 GB |

> **Why dense 1B as default?** Maximum parameters that fit comfortably in 16 GB with room to breathe. MoE stretches to 1.5B total params but is harder to train stably — switch to `khanh_15b_moe.yaml` only after smoke tests confirm training is healthy.

### Key architectural choices and rationale

#### RoPE (Rotary Position Embedding)
- Replaces the old learned positional embeddings (capped at 2048, no extrapolation).
- Encodes relative position directly in the attention computation.
- Allows context-length extension via RoPE scaling at inference time.
- Implementation: `src/khanh_llm/model/rope.py`.

#### RMSNorm (pre-norm)
- Replaces `nn.LayerNorm`. Faster, simpler (no recentering), fewer parameters.
- **Pre-norm** placement: `x = x + sublayer(norm(x))` — more stable at depth than post-norm.
- Implementation: `src/khanh_llm/model/norm.py`.

#### SwiGLU FFN
- Replaces GELU FFN. Used by Llama, Mistral, Qwen, PaLM.
- `FFN(x) = (xW₁ ⊙ SiLU(xW₃)) W₂`
- FFN dim is `~2.7×` hidden (2752 for hidden=1024) to keep param count comparable to the 4× GELU baseline.
- Implementation: `src/khanh_llm/model/ffn.py`.

#### Grouped-Query Attention (GQA)
- 16 Q-heads, 4 KV-heads → 4× smaller KV cache, almost no quality loss.
- Uses `F.scaled_dot_product_attention(is_causal=True)` to automatically dispatch to the FlashAttention-2 backend.
- No manually-built additive causal mask — PyTorch handles it.
- Implementation: `src/khanh_llm/model/attention.py`.

#### Tied embeddings
- `output_layer.weight = token_embedding.weight` — saves ~51M parameters at 49K vocab.
- Parameter count above already includes this saving.

### Module layout

```
src/khanh_llm/model/
├── transformer.py   ← KhanhLLM: full model (embedding → blocks → output)
├── attention.py     ← GQAAttention using SDPA
├── ffn.py           ← SwiGLU feed-forward
├── moe.py           ← MoE FFN (disabled in default config, available for later)
├── norm.py          ← RMSNorm
└── rope.py          ← Rotary position embeddings
```

### `state_dict` key names

Keys are stored **without** the `_orig_mod.` prefix that `torch.compile` adds. The checkpoint save routine strips this prefix so checkpoints load cleanly into both compiled and uncompiled models.

Example key shapes for `khanh_1b`:

| Key | Shape |
|---|---|
| `token_embedding.weight` | `[49152, 2048]` |
| `layers.0.attn.q_proj.weight` | `[2048, 2048]` (16 heads × 128) |
| `layers.0.attn.k_proj.weight` | `[512, 2048]` (GQA: 4 heads × 128) |
| `layers.0.attn.v_proj.weight` | `[512, 2048]` |
| `layers.0.attn.o_proj.weight` | `[2048, 2048]` |
| `layers.0.ffn.gate_proj.weight` | `[5504, 2048]` |
| `layers.0.ffn.up_proj.weight` | `[5504, 2048]` |
| `layers.0.ffn.down_proj.weight` | `[2048, 5504]` |
| `layers.0.norm1.weight` | `[2048]` |
| `layers.0.norm2.weight` | `[2048]` |
| `norm_f.weight` | `[2048]` |

> `output_layer` has no stored weight — it shares `token_embedding.weight`.

### Legacy 692M MoE architecture

Archived at `runs/_archive_legacy_692m/`. Config preserved at `configs/model/khanh_692m_legacy.yaml` for reference. The checkpoint is loadable via `scripts/inference/chat_cli.py --legacy`.

Key differences from new architecture:

| Feature | Legacy 692M | New khanh_1b |
|---|---|---|
| Params | ~692M | ~1B |
| Position | Learned PE (capped 2048) | RoPE |
| Norm | LayerNorm, post-norm | RMSNorm, pre-norm |
| FFN | GELU | SwiGLU |
| Attention | `nn.MultiheadAttention` | GQA + SDPA |
| MoE | 7 MoE layers, `argmax` routing | Dense (MoE disabled) |
| Tokenizer | Custom 50K BPE on C4 | StarCoder2 49K code-aware BPE |
| Tied embed | No | Yes |
| KV cache | No | Yes (generator.py) |
