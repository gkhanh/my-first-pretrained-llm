"""KV cache equivalence test.

Verifies that incremental generation with KV cache produces outputs
identical to a full forward pass (within floating point tolerance).

This is the key correctness check before relying on the KV cache for inference.
"""

import pytest
import torch

from khanh_llm.config import ModelConfig
from khanh_llm.model.transformer import KhanhLLM


@pytest.fixture
def small_model() -> KhanhLLM:
    """A tiny model for fast CPU testing."""
    cfg = ModelConfig(
        hidden_dim=256, num_layers=2, num_heads_q=4, num_heads_kv=2,
        ffn_dim=512, vocab_size=1000, max_seq_len=64
    )
    return KhanhLLM(cfg).eval()


def test_kv_cache_matches_full_forward(small_model: KhanhLLM) -> None:
    """KV-cache forward matches full-sequence forward for last token logits."""
    torch.manual_seed(0)
    tokens = torch.randint(0, small_model.cfg.vocab_size, (1, 16))

    with torch.no_grad():
        # Full forward: feed entire sequence
        logits_full, _, _ = small_model(tokens)

        # KV-cache: prefill then step
        _, _, past_kvs = small_model(tokens[:, :-1])
        logits_cached, _, _ = small_model(tokens[:, -1:], past_kvs=past_kvs)

    # Last position logits should match between the two approaches
    assert torch.allclose(
        logits_full[:, -1, :],
        logits_cached[:, 0, :],
        atol=1e-4,
        rtol=1e-4,
    ), "KV-cache output diverges from full forward at tolerance 1e-4"


def test_kv_cache_multi_step(small_model: KhanhLLM) -> None:
    """Multi-step KV-cache generation is consistent with the full forward."""
    torch.manual_seed(42)
    prefix_len = 8
    tokens = torch.randint(0, small_model.cfg.vocab_size, (1, prefix_len + 3))

    with torch.no_grad():
        # Full forward over all tokens at once
        logits_full, _, _ = small_model(tokens)

        # Step-by-step with KV cache
        _, _, past_kvs = small_model(tokens[:, :prefix_len])
        for step in range(3):
            pos = prefix_len + step
            logits_step, _, past_kvs = small_model(tokens[:, pos:pos+1], past_kvs=past_kvs)
            assert torch.allclose(
                logits_full[:, pos, :], logits_step[:, 0, :], atol=1e-4, rtol=1e-4
            ), f"KV-cache diverges at step {step} (position {pos})"
