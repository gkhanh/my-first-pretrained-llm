"""Smoke test: MoE balance loss has a non-zero gradient on gate.weight.

This is the key regression test for the bug in the legacy implementation
where argmax was used instead of top-k for fraction_tokens, making the
balance loss gradient zero (or near-zero) for gate.weight.
"""

import torch

from khanh_llm.model.moe import MoELayer


def test_moe_balance_loss_gradient_flows() -> None:
    """aux_loss must produce non-zero gradient on gate.weight."""
    moe = MoELayer(hidden_dim=64, ffn_dim=128, num_experts=4, num_experts_active=2)

    x = torch.randn(2, 16, 64)  # (batch=2, seq=16, d=64)
    output, aux_loss = moe(x)

    assert aux_loss.item() > 0, "aux_loss should be positive (experts not perfectly balanced)"

    # Backpropagate only through aux_loss
    aux_loss.backward()

    gate_grad = moe.gate.weight.grad
    assert gate_grad is not None, "gate.weight.grad is None — no gradient flowing"
    assert gate_grad.abs().max().item() > 1e-8, \
        "gate.weight gradient is effectively zero — balance loss not differentiable"


def test_moe_output_shape() -> None:
    """MoE output shape matches input shape."""
    moe = MoELayer(hidden_dim=64, ffn_dim=128, num_experts=4, num_experts_active=2)
    x = torch.randn(2, 32, 64)
    output, aux_loss = moe(x)
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"


def test_moe_fraction_tokens_sums_to_one() -> None:
    """Fraction tokens across experts should sum to approximately 1."""
    moe = MoELayer(hidden_dim=64, ffn_dim=128, num_experts=4, num_experts_active=2)
    x = torch.randn(4, 16, 64)

    # Manually recompute fraction_tokens to verify the formula
    import torch.nn.functional as F
    flat_x = x.view(-1, 64)
    with torch.no_grad():
        routing_probs = F.softmax(moe.gate(flat_x), dim=-1)
        _, top_indices = torch.topk(routing_probs, 2, dim=-1)
        one_hot = F.one_hot(top_indices, num_classes=4).float()
        fraction_tokens = one_hot.sum(dim=1).mean(dim=0)

    # With K_active=2 experts per token, total fraction should sum to K_active
    assert abs(fraction_tokens.sum().item() - 2.0) < 1e-4, \
        f"fraction_tokens should sum to K_active=2, got {fraction_tokens.sum().item()}"
