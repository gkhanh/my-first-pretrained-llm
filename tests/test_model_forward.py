"""Smoke test: instantiate KhanhLLM and run one forward pass.

Tests both the 150m (debug) and 1b configs to catch shape/dtype errors
before starting a long training run.
"""

import pytest
import torch

from khanh_llm.config import ModelConfig
from khanh_llm.model.transformer import KhanhLLM


def make_model(name: str) -> KhanhLLM:
    configs = {
        "150m": ModelConfig(name="khanh_150m", hidden_dim=768, num_layers=2, num_heads_q=12,
                            num_heads_kv=4, ffn_dim=2048),
        "1b": ModelConfig(name="khanh_1b", hidden_dim=2048, num_layers=2, num_heads_q=16,
                            num_heads_kv=4, ffn_dim=5504),
    }
    return KhanhLLM(configs[name])


@pytest.mark.parametrize("model_name", ["150m", "1b"])
def test_forward_pass_shapes(model_name: str) -> None:
    """Forward pass returns correct output shapes."""
    model = make_model(model_name).eval()
    B, T = 2, 64
    tokens = torch.randint(0, model.cfg.vocab_size, (B, T))

    with torch.no_grad():
        logits, aux_loss, kvs = model(tokens)

    assert logits.shape == (B, T, model.cfg.vocab_size), f"Bad logits shape: {logits.shape}"
    assert aux_loss.item() == 0.0, "aux_loss should be 0 when MoE is disabled"
    assert len(kvs) == model.cfg.num_layers, "Should have one KV pair per layer"


@pytest.mark.parametrize("model_name", ["150m", "1b"])
def test_tied_embeddings(model_name: str) -> None:
    """Output projection weight is shared with token embedding."""
    model = make_model(model_name)
    assert model.output.weight is model.token_embedding.weight, \
        "output.weight should be tied to token_embedding.weight"


def test_param_count_reasonable() -> None:
    """1B config (22 layers, full depth) has roughly the expected parameter count."""
    model = KhanhLLM(ModelConfig(hidden_dim=2048, num_layers=22, num_heads_q=16,
                                  num_heads_kv=4, ffn_dim=5504))
    n = model.num_parameters()
    # With tied embeddings and GQA, should be roughly 900M-1.1B
    assert 900_000_000 < n < 1_100_000_000, f"Unexpected param count: {n:,}"


def test_kv_cache_forward() -> None:
    """KV cache forward produces same logits as full forward for the last token."""
    model = make_model("150m").eval()
    tokens = torch.randint(0, model.cfg.vocab_size, (1, 10))

    with torch.no_grad():
        # Full forward
        logits_full, _, _ = model(tokens)

        # KV-cache: prefill all but last token, then forward last token
        _, _, past_kvs = model(tokens[:, :-1])
        logits_kv, _, _ = model(tokens[:, -1:], past_kvs=past_kvs)

    # Last token logits should match
    assert torch.allclose(logits_full[:, -1, :], logits_kv[:, 0, :], atol=1e-4), \
        "KV-cache output does not match full forward pass output"
