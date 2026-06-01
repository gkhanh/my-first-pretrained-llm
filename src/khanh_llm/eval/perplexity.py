"""Per-slice perplexity evaluation on pre-tokenized shard files."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from khanh_llm.data.shards import ShardedDataset
from khanh_llm.model.transformer import KhanhLLM


@torch.inference_mode()
def compute_perplexity(
    model: KhanhLLM,
    shard_path: str | Path,
    seq_len: int = 2048,
    max_batches: int = 50,
    device: torch.device | str = "cuda",
    use_bf16: bool = True,
) -> float:
    """Compute perplexity on a held-out shard file.

    Args:
        model: A KhanhLLM instance in eval mode.
        shard_path: Path to a .bin shard file (e.g. data/shards/eval/code_eval.bin).
        seq_len: Sequence length for each evaluation chunk.
        max_batches: Maximum number of chunks to evaluate (for speed).
        device: Evaluation device.
        use_bf16: Whether to use BF16 autocast.

    Returns:
        Perplexity (float). Lower is better.
    """
    device = torch.device(device)
    model = model.to(device).eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    # Create a single-shard dataset
    shard_dir = Path(shard_path).parent
    ds = ShardedDataset(shard_dir, seq_len=seq_len)

    total_loss   = 0.0
    total_tokens = 0
    batches_seen = 0

    for input_ids, labels in ds:
        if batches_seen >= max_batches:
            break

        input_ids = input_ids.unsqueeze(0).to(device)
        labels    = labels.unsqueeze(0).to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            logits, _, _ = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        total_loss   += loss.item()
        total_tokens += labels.numel()
        batches_seen += 1

    if total_tokens == 0:
        return float("inf")

    avg_loss    = total_loss / total_tokens
    perplexity  = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity
