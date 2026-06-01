"""Configuration dataclasses for model, training, and data pipelines.

All configs are serialisable to/from YAML via OmegaConf:

    from omegaconf import OmegaConf
    from khanh_llm.config import ModelConfig

    cfg = OmegaConf.structured(ModelConfig)
    OmegaConf.save(cfg, "configs/model/khanh_1b.yaml")
    loaded = OmegaConf.load("configs/model/khanh_1b.yaml")
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ── Model config ──────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Architecture hyperparameters for KhanhLLM."""

    # Identification
    name: str = "khanh_1b"

    # Dimensions
    hidden_dim: int = 2048
    num_layers: int = 22
    num_heads_q: int = 16        # Query heads
    num_heads_kv: int = 4        # Key/Value heads (GQA)
    ffn_dim: int = 5504          # SwiGLU intermediate (≈2.69× hidden)

    # Vocabulary / sequence
    vocab_size: int = 49152      # StarCoder2 tokenizer
    max_seq_len: int = 2048
    rope_base: float = 500000.0  # High RoPE base for better long-context extrapolation

    # Regularisation
    dropout: float = 0.0         # 0 is standard for large-scale pretrain

    # Efficiency
    tied_embeddings: bool = True  # Output layer shares input embedding weights

    # MoE (disabled by default; set use_moe=True for khanh_15b_moe)
    use_moe: bool = False
    num_experts: int = 8
    num_experts_active: int = 2
    aux_loss_weight: float = 0.01


# ── Training config ───────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    """Hyperparameters for the training loop."""

    # Run identification
    run_name: str = "khanh_1b_run1"
    run_dir: str = "runs"

    # Batch / accumulation
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 32
    # Effective batch = micro_batch_size × gradient_accumulation_steps × seq_len tokens
    # = 4 × 32 × 2048 = 262,144 tokens/step

    # Learning rate schedule
    peak_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 2000
    lr_betas: tuple = (0.9, 0.95)
    lr_eps: float = 1e-8
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0

    # Mixed precision
    dtype: str = "bfloat16"  # "bfloat16" | "float32"

    # Activation checkpointing
    activation_checkpointing: str = "selective"  # "off" | "selective" | "full"

    # EMA
    ema_decay: float = 0.999

    # Checkpointing
    save_every_tokens: int = 5_000_000_000   # 5B tokens
    keep_last_n: int = 10                    # rolling window; every 10th is kept permanently

    # Logging
    log_every_steps: int = 10
    eval_every_tokens: int = 2_000_000_000  # eval checkpoint every 2B tokens
    use_wandb: bool = False                  # CSV-only by default; set True to enable W&B

    # torch.compile
    compile: bool = True
    compile_mode: str = "default"  # "default" | "reduce-overhead" | "max-autotune"

    # Max tokens to train for (set to 0 for unlimited)
    max_tokens: int = 390_000_000_000  # 390B tokens for khanh_1b


# ── Data config ───────────────────────────────────────────────────────────────

@dataclass
class DataSource:
    name: str
    hf_path: str | None = None
    local_path: str | None = None
    weight: float = 1.0
    split: str = "train"
    language_filter: list | None = None  # for code datasets


@dataclass
class DataConfig:
    """Configuration for the data pipeline."""

    tokenizer_path: str = "data/tokenizers/starcoder2"
    shard_dir: str = "data/shards/pretrain"
    shard_size_tokens: int = 500_000_000  # 500M tokens per shard
    seq_len: int = 2048
    num_workers: int = 4

    # FIM
    fim_rate: float = 0.5   # fraction of code sequences to apply FIM to

    # Data sources with weights (must sum to 1.0)
    sources: list = field(default_factory=lambda: [
        DataSource("stack_v2_code",    hf_path="bigcode/the-stack-v2-train-smol-ids", weight=0.50,
                   language_filter=["Python", "JavaScript", "TypeScript", "Go", "Rust"]),
        DataSource("stackexchange_prog", hf_path="HuggingFaceH4/stack-exchange-preferences", weight=0.08),
        DataSource("stack_v2_docs",    hf_path="bigcode/the-stack-v2-train-smol-ids", weight=0.07,
                   language_filter=["Markdown"]),
        DataSource("sec_edgar",         local_path="data/raw/finance/edgar", weight=0.08),
        DataSource("finance_news",      local_path="data/raw/finance/news", weight=0.05),
        DataSource("stackexchange_fin", hf_path="HuggingFaceH4/stack-exchange-preferences", weight=0.02),
        DataSource("wikipedia",         hf_path="wikimedia/wikipedia", weight=0.10),
        DataSource("c4",               hf_path="allenai/c4", weight=0.10),
    ])
