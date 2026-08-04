"""Episode 5 — The Policy Head — Teaching the Network to Choose Moves

Concept taught: How to convert a spatial feature map into a probability distribution
over moves. The auxiliary opponent policy head for regularization (KataGo improvement).
How illegal move masking works and why it's applied AFTER the logits, not before.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyHead(nn.Module):
    """Dual policy head: current player moves + opponent move prediction (auxiliary).

    The main policy output is used for move selection during play and MCTS.
    The auxiliary opponent policy is used only during training as a regularizer
    (KataGo domain-independent improvement: predicts opponent's likely next move
    to force the network to model both perspectives).

    During inference, only call forward() and use policy_logits.
    During training, use both policy_logits and opp_policy_logits.

    Architecture (each head):
        input (B, C, 9, 9)
            -> Conv2d(C, 2, kernel=1) -> BN -> ReLU  # (B, 2, 9, 9)
            -> Flatten                                 # (B, 162)
            -> Linear(162, 81)                         # (B, 81)

    Args:
        in_channels: Number of channels from trunk (C).

    Input:  Tensor of shape (B, C, 9, 9)
    Output: Tuple of (policy_logits, opp_policy_logits), each shape (B, 81).
            These are RAW LOGITS. Apply legal move masking and softmax externally.
    """

    def __init__(self, in_channels: int):
        super().__init__()

        # Main policy head
        self.main_conv = nn.Conv2d(in_channels, 2, kernel_size=1, bias=False)
        self.main_bn = nn.BatchNorm2d(2)
        self.main_fc = nn.Linear(2 * 9 * 9, 81)  # 162 -> 81

        # Auxiliary opponent policy head (separate weights)
        self.opp_conv = nn.Conv2d(in_channels, 2, kernel_size=1, bias=False)
        self.opp_bn = nn.BatchNorm2d(2)
        self.opp_fc = nn.Linear(2 * 9 * 9, 81)  # 162 -> 81

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning raw logits for both policy heads.

        Args:
            x: Trunk features, shape (B, C, 9, 9).

        Returns:
            (policy_logits, opp_policy_logits), each shape (B, 81).
        """
        # Main policy
        p = self.main_conv(x)          # (B, 2, 9, 9)
        p = self.main_bn(p)            # (B, 2, 9, 9)
        p = F.relu(p)                  # (B, 2, 9, 9)
        p = p.flatten(start_dim=1)     # (B, 162)
        policy_logits = self.main_fc(p)  # (B, 81)

        # Opponent policy
        o = self.opp_conv(x)           # (B, 2, 9, 9)
        o = self.opp_bn(o)             # (B, 2, 9, 9)
        o = F.relu(o)                  # (B, 2, 9, 9)
        o = o.flatten(start_dim=1)     # (B, 162)
        opp_policy_logits = self.opp_fc(o)  # (B, 81)

        return policy_logits, opp_policy_logits


def apply_legal_mask(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Apply legal move mask to policy logits and return probability distribution.

    This is a standalone function (not a method) so it can be taught independently
    and used both during play and training.

    CRITICAL: Set illegal logits to -inf, then softmax. This is the CORRECT approach.
    Masking before softmax (e.g., logits * mask) causes numerical issues.

    Args:
        logits: Raw policy logits, shape (B, 81) or (81,).
        legal_mask: Binary mask, shape (B, 81) or (81,), 1.0=legal, 0.0=illegal.

    Returns:
        Probability distribution over legal moves, same shape as logits.
        Illegal moves have probability 0.0.
        Legal moves sum to 1.0.

    Raises:
        ValueError: If no legal moves exist (all mask values are 0).
    """
    if legal_mask.sum() == 0:
        raise ValueError("No legal moves exist — mask is all zeros")

    # Set illegal moves to -inf so softmax gives them probability 0
    # Ensure mask is on the same device as logits
    legal_mask = legal_mask.to(logits.device)
    masked_logits = logits.masked_fill(~legal_mask.bool(), float('-inf'))
    probs = F.softmax(masked_logits, dim=-1)
    return probs


if __name__ == "__main__":
    print("=== Episode 5: Policy Head ===\n")

    head = PolicyHead(in_channels=128)
    x = torch.randn(4, 128, 9, 9)
    policy, opp_policy = head(x)

    print(f"Input shape:       {x.shape}")
    print(f"Policy shape:      {policy.shape}")
    print(f"Opp policy shape:  {opp_policy.shape}")

    assert policy.shape == (4, 81)
    assert opp_policy.shape == (4, 81)

    # Test legal masking
    mask = torch.zeros(81)
    mask[:10] = 1.0  # first 10 moves legal
    probs = apply_legal_mask(policy[0], mask)
    assert probs.shape == (81,)
    assert abs(probs.sum().item() - 1.0) < 1e-5, f"Probs should sum to 1, got {probs.sum()}"
    assert probs[10:].sum().item() < 1e-6, "Illegal moves should have ~0 probability"
    print("Legal mask test: PASSED")

    num_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    print("\n=== Episode 5 PASSED ===")
