"""DPO (Direct Preference Optimisation) — placeholder for future implementation.

This module will implement DPO training on top of a fine-tuned KhanhLLM or external HF model.
See docs/08-finetuning.md and the Phase 3.6 / Months 19-22 roadmap section.

Preference data format (JSONL):
    {"prompt": "...", "chosen": "...", "rejected": "..."}

Reference: Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
"""

from __future__ import annotations


def run_dpo(*args, **kwargs) -> None:
    """DPO training loop — not yet implemented.

    See docs/08-finetuning.md for the planned implementation.
    This will be filled in during Months 19-22 of the roadmap (post-SFT phase).
    """
    raise NotImplementedError(
        "DPO training is not yet implemented. "
        "See docs/08-finetuning.md for the planned design. "
        "This module will be implemented after the SFT pipeline is validated."
    )
