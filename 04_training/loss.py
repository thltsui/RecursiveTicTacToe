"""Episode 13 — The Loss Function — Teaching the Network Five Things at Once

Concept taught: Multi-task learning. Why each loss component matters.
How loss weights control what the network prioritizes learning.
How to interpret each loss component during training.

The Complete Loss:
    L_total = L_policy
            + lambda_value     * L_value
            + lambda_score     * L_score
            + lambda_ownership * L_ownership
            + lambda_opp       * L_opp_policy
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@dataclass
class LossBreakdown:
    """Breakdown of all loss components for logging.

    Attributes:
        total: Scalar — call .backward() on this.
        policy: Scalar — policy cross-entropy.
        value: Scalar — value MSE.
        score: Scalar — score margin MSE.
        ownership: Scalar — ownership BCE.
        opp_policy: Scalar — opponent policy cross-entropy.
    """
    total:          torch.Tensor  # scalar
    policy:         torch.Tensor  # scalar
    value:          torch.Tensor  # scalar
    score:          torch.Tensor  # scalar
    ownership:      torch.Tensor  # scalar
    opp_policy:     torch.Tensor  # scalar


def policy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    legal_masks: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy loss between network policy and MCTS policy target.

    Formula: L = -sum_a pi_mcts(a) * log(softmax(logits)[a])

    Important: Apply legal mask before computing softmax.
    The MCTS target already has 0.0 for illegal moves, so they don't
    contribute to the loss numerically, but the logit masking ensures
    the softmax denominator is correct.

    Args:
        logits: Raw policy logits, shape (B, 81).
        targets: MCTS probability distribution, shape (B, 81).
        legal_masks: Binary mask, shape (B, 81), 1.0=legal.

    Returns:
        Scalar mean loss over batch.
    """
    # Mask illegal moves to -inf so softmax gives them 0 probability
    masked_logits = logits.masked_fill(~legal_masks.bool(), float('-inf'))
    log_probs = F.log_softmax(masked_logits, dim=-1)         # (B, 81)

    # Replace -inf and NaN with -100 before multiplying by targets
    # to avoid 0 * (-inf) = nan and propagation of NaN from degenerate masks
    log_probs = torch.nan_to_num(log_probs, nan=-100.0, posinf=0.0, neginf=-100.0)

    # Cross-entropy: -sum(target * log_prob)
    loss_per_sample = -(targets * log_probs).sum(dim=-1)      # (B,)
    return loss_per_sample.mean()


def value_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error between predicted and actual game outcome.

    Formula: L = mean((z - v)^2)

    Note: MSE is used (not cross-entropy) because value is continuous in [-1, 1].

    Args:
        predictions: Network output, shape (B, 1), tanh output.
        targets: Actual outcomes, shape (B, 1), values in {-1, 0, 1}.

    Returns:
        Scalar mean loss over batch.
    """
    return F.mse_loss(predictions, targets)


def score_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """MSE between predicted score margin and actual score margin.

    Score margin = (sub_boards_won - sub_boards_lost) / 9, normalized to [-1, 1].

    Args:
        predictions: Network output, shape (B, 1), tanh output.
        targets: Normalized score margin, shape (B, 1).

    Returns:
        Scalar mean loss over batch.
    """
    return F.mse_loss(predictions, targets)


def ownership_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Binary cross-entropy between predicted and actual sub-board ownership.

    Formula: L = -mean(sum_i [o_i*log(o_hat_i) + (1-o_i)*log(1-o_hat_i)])

    Args:
        predictions: Network output, shape (B, 9), sigmoid output.
        targets: Binary targets, shape (B, 9). Did current player win each sub-board?

    Returns:
        Scalar mean loss over batch.
    """
    return F.binary_cross_entropy(predictions, targets)


def compute_total_loss(
    network_output: 'NetworkOutput',
    batch: 'TrainingBatch',
    lambda_value: float = 3.0,
    lambda_score: float = 1.0,
    lambda_ownership: float = 1.0,
    lambda_opp: float = 0.15,
) -> LossBreakdown:
    """Compute the complete weighted multi-task loss.

    L_total = L_policy
            + lambda_value     * L_value
            + lambda_score     * L_score
            + lambda_ownership * L_ownership
            + lambda_opp       * L_opp_policy

    Args:
        network_output: Forward pass output from the network.
        batch: Training batch with targets.
        lambda_value: Weight for value loss. Default: 1.0.
        lambda_score: Weight for score loss. Default: 0.5.
        lambda_ownership: Weight for ownership loss. Default: 0.5.
        lambda_opp: Weight for opponent policy loss. Default: 0.15.

    Returns:
        LossBreakdown — use .total for .backward(), components for logging.
    """
    l_policy = policy_loss(
        network_output.policy_logits, batch.policy_targets, batch.legal_masks
    )
    l_value = value_loss(network_output.win_value, batch.value_targets)
    l_score = score_loss(network_output.score_margin, batch.score_targets)
    l_ownership = ownership_loss(network_output.ownership, batch.ownership_targets)
    # Opponent policy targets come from the next ply state, so use its legal mask.
    opp_mask = batch.opp_legal_masks
    l_opp = policy_loss(
        network_output.opp_policy_logits, batch.opp_policy_targets, opp_mask
    )

    total = (l_policy
             + lambda_value * l_value
             + lambda_score * l_score
             + lambda_ownership * l_ownership
             + lambda_opp * l_opp)

    return LossBreakdown(
        total=total,
        policy=l_policy.detach(),
        value=l_value.detach(),
        score=l_score.detach(),
        ownership=l_ownership.detach(),
        opp_policy=l_opp.detach(),
    )


if __name__ == "__main__":
    print("=== Episode 13: Loss Function ===\n")

    B = 8

    # Test policy loss with perfect prediction
    targets = torch.zeros(B, 81)
    targets[:, 0] = 1.0  # all mass on move 0
    logits_perfect = torch.full((B, 81), -100.0)
    logits_perfect[:, 0] = 100.0
    masks = torch.ones(B, 81)
    loss_p = policy_loss(logits_perfect, targets, masks)
    print(f"Policy loss (perfect): {loss_p.item():.6f} (should be ~0)")
    assert loss_p.item() < 0.01

    # Test policy loss with uniform prediction
    logits_uniform = torch.zeros(B, 81)
    loss_u = policy_loss(logits_uniform, targets, masks)
    print(f"Policy loss (uniform): {loss_u.item():.4f} (should be ~log(81)={torch.tensor(81.0).log().item():.4f})")

    # Test value loss with perfect prediction
    v_pred = torch.ones(B, 1)
    v_target = torch.ones(B, 1)
    loss_v = value_loss(v_pred, v_target)
    print(f"Value loss (perfect): {loss_v.item():.6f} (should be 0)")
    assert loss_v.item() < 1e-6

    # Test compute_total_loss
    from importlib import import_module
    network_mod = import_module('02_network.network')

    net = network_mod.UltimateTTTNetwork(channels=32, num_blocks=2)
    x = torch.randn(B, 7, 9, 9)
    out = net(x)

    from .replay_buffer import TrainingBatch
    batch = TrainingBatch(
        state_tensors=x,
        policy_targets=torch.softmax(torch.randn(B, 81), dim=-1),
        opp_policy_targets=torch.softmax(torch.randn(B, 81), dim=-1),
        value_targets=torch.zeros(B, 1),
        score_targets=torch.zeros(B, 1),
        ownership_targets=torch.rand(B, 9).round(),
        legal_masks=torch.ones(B, 81),
        opp_legal_masks=torch.ones(B, 81),
    )

    breakdown = compute_total_loss(out, batch)
    print(f"\nTotal loss: {breakdown.total.item():.4f}")
    print(f"  Policy:    {breakdown.policy.item():.4f}")
    print(f"  Value:     {breakdown.value.item():.4f}")
    print(f"  Score:     {breakdown.score.item():.4f}")
    print(f"  Ownership: {breakdown.ownership.item():.4f}")
    print(f"  Opp Policy:{breakdown.opp_policy.item():.4f}")

    assert breakdown.total.item() > 0, "Total loss should be positive"
    assert all(getattr(breakdown, f).item() >= 0 for f in
               ['policy', 'value', 'score', 'ownership', 'opp_policy'])

    # Test backward
    breakdown2 = compute_total_loss(net(x), batch)
    breakdown2.total.backward()
    print("\nBackward pass: PASSED")

    print("\n=== Episode 13 PASSED ===")
