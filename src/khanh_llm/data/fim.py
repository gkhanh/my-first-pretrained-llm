"""Fill-in-the-Middle (FIM) transform for code sequences.

Applies the SPM (Suffix-Prefix-Middle) format used by StarCoder2:
    <fim_prefix>PREFIX<fim_suffix>SUFFIX<fim_middle>MIDDLE

Applied with probability `fim_rate` to code sequences during data preparation.
"""

from __future__ import annotations

import random


def apply_fim_transform(
    token_ids: list[int],
    fim_prefix_id: int,
    fim_suffix_id: int,
    fim_middle_id: int,
    eos_id: int,
    fim_rate: float = 0.5,
    rng: random.Random | None = None,
) -> list[int]:
    """Apply FIM (SPM format) to a tokenized code sequence.

    With probability `fim_rate`, splits the sequence into prefix/middle/suffix
    and reorders to: [FIM_PREFIX] prefix [FIM_SUFFIX] suffix [FIM_MIDDLE] middle.

    With probability `1 - fim_rate`, returns the sequence unchanged (causal mode).

    Args:
        token_ids: Flat list of token IDs for one document.
        fim_prefix_id: Token ID for <fim_prefix>.
        fim_suffix_id: Token ID for <fim_suffix>.
        fim_middle_id: Token ID for <fim_middle>.
        eos_id: Token ID for EOS (appended at the end).
        fim_rate: Probability of applying FIM (default 0.5).
        rng: Optional random.Random instance for reproducibility.

    Returns:
        Transformed (or unchanged) list of token IDs.
    """
    if rng is None:
        rng = random.Random()

    n = len(token_ids)
    if n < 3 or rng.random() >= fim_rate:
        return token_ids  # causal mode — no FIM applied

    # Pick two split points
    split1 = rng.randint(1, n - 2)
    split2 = rng.randint(split1 + 1, n - 1)

    prefix = token_ids[:split1]
    middle = token_ids[split1:split2]
    suffix = token_ids[split2:]

    # SPM: <fim_prefix> PREFIX <fim_suffix> SUFFIX <fim_middle> MIDDLE <eos>
    return [fim_prefix_id] + prefix + [fim_suffix_id] + suffix + [fim_middle_id] + middle + [eos_id]


def fim_transform_batch(
    documents: list[list[int]],
    fim_prefix_id: int,
    fim_suffix_id: int,
    fim_middle_id: int,
    eos_id: int,
    fim_rate: float = 0.5,
    seed: int = 42,
) -> list[list[int]]:
    """Apply FIM transform to a batch of tokenized documents.

    Args:
        documents: List of per-document token ID lists.
        fim_prefix_id: Token ID for <fim_prefix>.
        fim_suffix_id: Token ID for <fim_suffix>.
        fim_middle_id: Token ID for <fim_middle>.
        eos_id: EOS token ID appended after each FIM middle.
        fim_rate: Probability of applying FIM per document.
        seed: Random seed for reproducibility.

    Returns:
        List of transformed token ID lists (same length as input).
    """
    rng = random.Random(seed)
    return [
        apply_fim_transform(doc, fim_prefix_id, fim_suffix_id, fim_middle_id, eos_id, fim_rate, rng)
        for doc in documents
    ]
