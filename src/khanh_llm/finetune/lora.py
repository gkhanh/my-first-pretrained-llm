"""LoRA and QLoRA setup using the PEFT library.

Supports two modes:
- Mode A: LoRA fine-tuning on KhanhLLM checkpoints (full precision or BF16).
- Mode B: QLoRA on external HuggingFace causal-LM models (4-bit base + LoRA adapters).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def add_lora(
    model: nn.Module,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> nn.Module:
    """Attach LoRA adapters to a KhanhLLM model using PEFT.

    Args:
        model: The KhanhLLM model to fine-tune.
        r: LoRA rank (higher = more capacity, more params).
        lora_alpha: LoRA scaling factor.
        lora_dropout: Dropout on the LoRA adapter layers.
        target_modules: List of module names to apply LoRA to.
            Defaults to all Q/K/V/O projection layers.

    Returns:
        A PeftModel wrapping the original model.
    """
    from peft import LoraConfig, TaskType, get_peft_model  # type: ignore

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    lora_cfg = LoraConfig(
        task_type      = TaskType.CAUSAL_LM,
        r              = r,
        lora_alpha     = lora_alpha,
        lora_dropout   = lora_dropout,
        target_modules = target_modules,
        bias           = "none",
    )
    return get_peft_model(model, lora_cfg)


def add_qlora(
    model_name_or_path: str,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
    device_map: str = "auto",
) -> tuple:
    """Load an external HuggingFace model in 4-bit and attach LoRA adapters (QLoRA).

    Args:
        model_name_or_path: HuggingFace model name (e.g. "Qwen/Qwen2.5-1.5B").
        r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        lora_dropout: Adapter dropout.
        target_modules: Module names to apply LoRA to (defaults to attention + FFN projections).
        device_map: HuggingFace device map for model loading.

    Returns:
        (peft_model, tokenizer) tuple, ready for SFT training.
    """
    from peft import (  # type: ignore
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_compute_dtype    = torch.bfloat16,
        bnb_4bit_use_double_quant = True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config = bnb_cfg,
        device_map          = device_map,
        trust_remote_code   = True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    lora_cfg = LoraConfig(
        task_type      = TaskType.CAUSAL_LM,
        r              = r,
        lora_alpha     = lora_alpha,
        lora_dropout   = lora_dropout,
        target_modules = target_modules,
        bias           = "none",
    )
    peft_model = get_peft_model(base_model, lora_cfg)
    peft_model.print_trainable_parameters()

    return peft_model, tokenizer
