"""FinQA / ConvFinQA financial evaluation harness stub.

See docs/07-evaluation.md and docs/09-finance-domain.md for context.
"""

from __future__ import annotations


def evaluate_finqa(
    model,
    tokenizer,
    split: str = "test",
    device: str = "cuda",
    max_new_tokens: int = 128,
) -> dict:
    """Run FinQA exact-match accuracy evaluation.

    Args:
        model: KhanhLLM or HF causal-LM in eval mode.
        tokenizer: Matching tokenizer.
        split: Dataset split ("train", "dev", "test").
        device: Evaluation device.
        max_new_tokens: Max tokens to generate for each answer.

    Returns:
        {"execution_accuracy": float}
    """
    raise NotImplementedError(
        "FinQA harness not yet implemented. "
        "Dataset: ibm/finqa — load via `datasets.load_dataset('ibm/finqa')`."
        " See docs/07-evaluation.md for expected accuracy ranges."
    )


def evaluate_convfinqa(
    model,
    tokenizer,
    split: str = "test",
    device: str = "cuda",
    max_new_tokens: int = 128,
) -> dict:
    """Run ConvFinQA multi-turn accuracy evaluation.

    Returns:
        {"execution_accuracy": float}
    """
    raise NotImplementedError(
        "ConvFinQA harness not yet implemented. "
        "Dataset: ibm/convfinqa — load via `datasets.load_dataset('ibm/convfinqa')`."
    )
