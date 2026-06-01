"""KhanhLLM — the main Transformer model.

A Llama-style dense decoder-only Transformer:
- Pre-norm (RMSNorm)
- RoPE positional embeddings
- GQA + SDPA (FlashAttention-2 backend)
- SwiGLU FFN
- Tied input/output embeddings

Config is loaded from ModelConfig or a YAML file (via OmegaConf).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from khanh_llm.config import ModelConfig
from khanh_llm.model.attention import GQAAttention
from khanh_llm.model.ffn import SwiGLU
from khanh_llm.model.moe import MoELayer
from khanh_llm.model.norm import RMSNorm


class TransformerBlock(nn.Module):
    """Single Transformer decoder block (pre-norm).

    Structure:
        x = x + attn(norm1(x))
        x = x + ffn(norm2(x))
    """

    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        self.norm1 = RMSNorm(cfg.hidden_dim)
        self.attn  = GQAAttention(
            hidden_dim    = cfg.hidden_dim,
            num_heads_q   = cfg.num_heads_q,
            num_heads_kv  = cfg.num_heads_kv,
            max_seq_len   = cfg.max_seq_len,
            rope_base     = cfg.rope_base,
            dropout       = cfg.dropout,
        )

        self.norm2 = RMSNorm(cfg.hidden_dim)
        if cfg.use_moe:
            self.ffn: nn.Module = MoELayer(
                hidden_dim         = cfg.hidden_dim,
                ffn_dim            = cfg.ffn_dim,
                num_experts        = cfg.num_experts,
                num_experts_active = cfg.num_experts_active,
                aux_loss_weight    = cfg.aux_loss_weight,
            )
        else:
            self.ffn = SwiGLU(cfg.hidden_dim, cfg.ffn_dim)

        self.use_moe = cfg.use_moe

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Returns:
            (x, new_kv, aux_loss)
        """
        # ── Attention (pre-norm) ─────────────────────────────────────────────
        attn_out, new_kv = self.attn(self.norm1(x), past_kv=past_kv)
        x = x + attn_out

        # ── FFN (pre-norm) ───────────────────────────────────────────────────
        normed = self.norm2(x)
        if self.use_moe:
            ffn_out, aux_loss = self.ffn(normed)
        else:
            ffn_out  = self.ffn(normed)
            aux_loss = x.new_zeros(1).squeeze()

        x = x + ffn_out
        return x, new_kv, aux_loss


class KhanhLLM(nn.Module):
    """KhanhLLM: Llama-style dense decoder-only LLM.

    Usage:
        cfg = ModelConfig()            # or load from YAML
        model = KhanhLLM(cfg)
        logits, aux_loss = model(tokens)

    Inference with KV cache:
        logits, aux_loss, new_kvs = model(tokens, past_kvs=past_kvs)
    """

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = ModelConfig()
        self.cfg = cfg

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)

        self.layers = nn.ModuleList([
            TransformerBlock(cfg, layer_idx=i)
            for i in range(cfg.num_layers)
        ])

        self.norm_f = RMSNorm(cfg.hidden_dim)

        # Output projection (tied to token_embedding by default)
        self.output = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)
        if cfg.tied_embeddings:
            self.output.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Apply initialisation scheme from GPT-NeoX / Llama."""
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self,
        tokens: torch.Tensor,
        past_kvs: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_checkpoint: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass.

        Args:
            tokens: Input token IDs of shape (B, seq_len).
            past_kvs: List of (key, value) pairs for each layer (KV-cache inference).
                      None means training / full-sequence prefill.
            use_checkpoint: If True, uses activation checkpointing on each block.
                            Controlled by TrainConfig.activation_checkpointing in the trainer.

        Returns:
            (logits, total_aux_loss, new_kvs)
            - logits: (B, seq_len, vocab_size)
            - total_aux_loss: scalar (0 when MoE is disabled)
            - new_kvs: list of (key, value) tuples, one per layer
        """
        x = self.token_embedding(tokens)

        total_aux_loss = x.new_zeros(1).squeeze()
        new_kvs: list[tuple[torch.Tensor, torch.Tensor]] = []

        for i, layer in enumerate(self.layers):
            past_kv = past_kvs[i] if past_kvs is not None else None

            if use_checkpoint and self.training:
                # Activation checkpointing: only supported without KV cache
                def create_custom_forward(layer_fn):
                    def custom_forward(*inputs):
                        out, kv, al = layer_fn(inputs[0], past_kv=None)
                        return out, kv[0], kv[1], al
                    return custom_forward

                x, k, v, aux = checkpoint(
                    create_custom_forward(layer), x, use_reentrant=False
                )
                new_kvs.append((k, v))
            else:
                x, new_kv, aux = layer(x, past_kv=past_kv)
                new_kvs.append(new_kv)

            total_aux_loss = total_aux_loss + aux

        x = self.norm_f(x)
        logits = self.output(x)
        return logits, total_aux_loss, new_kvs

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        """Count trainable parameters."""
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if exclude_embeddings:
            params -= self.token_embedding.weight.numel()
        return params
