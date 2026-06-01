"""LoRA adapter save/load/merge utilities."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def save_adapter(peft_model: nn.Module, output_dir: str | Path) -> None:
    """Save LoRA adapter weights and config to a directory.

    Args:
        peft_model: A PEFT-wrapped model (returned by add_lora or add_qlora).
        output_dir: Directory to save adapter files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(output_dir))
    print(f"[Adapters] Saved adapter to {output_dir}")


def load_adapter(base_model: nn.Module, adapter_dir: str | Path) -> nn.Module:
    """Load LoRA adapter weights into a base model.

    Args:
        base_model: The un-adapted base model.
        adapter_dir: Directory containing adapter_config.json and adapter_model.bin.

    Returns:
        A PEFT-wrapped model with loaded adapters.
    """
    from peft import PeftModel  # type: ignore

    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    print(f"[Adapters] Loaded adapter from {adapter_dir}")
    return peft_model


def merge_adapter(peft_model: nn.Module, output_dir: str | Path) -> nn.Module:
    """Merge LoRA adapter weights into the base model and save.

    The resulting model has no adapter — it's a standalone model with the
    fine-tuned weights baked in. Useful for deployment or further fine-tuning.

    Args:
        peft_model: A PEFT-wrapped model.
        output_dir: Directory to save the merged model.

    Returns:
        The merged model (in float16 for storage efficiency).
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged = peft_model.merge_and_unload()
    merged = merged.half()  # save in fp16

    # Try to save via HF save_pretrained if available
    if hasattr(merged, "save_pretrained"):
        merged.save_pretrained(str(output_dir))
    else:
        torch.save({"model_state_dict": merged.state_dict()}, output_dir / "merged_model.pt")

    print(f"[Adapters] Merged model saved to {output_dir}")
    return merged
