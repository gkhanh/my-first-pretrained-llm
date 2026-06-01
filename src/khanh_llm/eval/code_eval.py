"""HumanEval / MBPP code evaluation harness stub.

Full implementation depends on the `human-eval` and `evaluate` packages.
Run after pretrain or SFT to measure pass@1 on Python coding problems.

Install extras:
    pip install human-eval   # for HumanEval
    pip install evaluate     # for MBPP via HuggingFace evaluate

See docs/07-evaluation.md for expected pass@1 ranges.
"""

from __future__ import annotations


def evaluate_humaneval(
    model,
    tokenizer,
    device: str = "cuda",
    n_samples: int = 1,
    temperature: float = 0.2,
    max_new_tokens: int = 512,
) -> dict:
    """Run HumanEval pass@1 evaluation.

    Args:
        model: KhanhLLM or HF causal-LM in eval mode.
        tokenizer: Matching tokenizer.
        device: Evaluation device.
        n_samples: Samples per problem (1 for pass@1, 10 for pass@10).
        temperature: Sampling temperature (lower for pass@1).
        max_new_tokens: Max tokens to generate per solution.

    Returns:
        {"pass@1": float, "pass@10": float (if n_samples>=10)}
    """
    # TODO: implement using the human-eval package
    # https://github.com/openai/human-eval
    raise NotImplementedError(
        "HumanEval harness not yet implemented. "
        "See docs/07-evaluation.md for the planned setup. "
        "Will be added alongside the full pretrain pipeline (Month 1-2 work)."
    )


def evaluate_mbpp(
    model,
    tokenizer,
    device: str = "cuda",
    n_samples: int = 1,
    temperature: float = 0.2,
    max_new_tokens: int = 512,
) -> dict:
    """Run MBPP pass@1 evaluation via HuggingFace evaluate.

    Returns:
        {"pass@1": float}
    """
    raise NotImplementedError(
        "MBPP harness not yet implemented. "
        "See docs/07-evaluation.md for the planned setup."
    )
