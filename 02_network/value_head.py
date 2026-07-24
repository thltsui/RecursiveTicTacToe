"""Episode 6 — The Value Head — Teaching the Network to Evaluate Positions

Concept taught: Multi-task learning from a single position. How auxiliary targets
(score margin, sub-board ownership) give richer training signal than win/loss alone.
Why tanh is the right activation for value (bounded [-1, 1]).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ValueHeadOutput:
    """Output container for the value head.

    Attributes:
        win_value: (B, 1) — main output, range [-1, 1]. Probability current player wins.
        score_margin: (B, 1) — auxiliary, range [-1, 1]. Predicted sub-board differential.
        ownership: (B, 9) — auxiliary, range [0, 1]. Per-sub-board win probability.
    """
    win_value: torch.Tensor     # (B, 1)
    score_margin: torch.Tensor  # (B, 1)
    ownership: torch.Tensor     # (B, 9)


class ValueHead(nn.Module):
    """Multi-output value head with auxiliary prediction targets.

    Main output: win_value — used for MCTS and move selection.
    Auxiliary outputs: score_margin, ownership — used only during training.

    The auxiliary outputs provide richer training signal per game
    (KataGo domain-independent improvement). They force the network to
    develop an internal model of board control, not just win probability.

    Architecture:
        input (B, C, 9, 9)
            -> Conv2d(C, 32, kernel=1) -> BN -> ReLU     # (B, 32, 9, 9)
            -> Flatten                                     # (B, 2592)
            -> Linear(2592, 512) -> ReLU                   # (B, 512)
            -> Linear(512, 128) -> ReLU                    # (B, 128)
            -> shared_features
                 |-> Linear(128, 1) -> Tanh    -> win_value      (B, 1)
                 |-> Linear(128, 1) -> Tanh    -> score_margin   (B, 1)
                 |-> Linear(128, 9) -> Sigmoid -> ownership      (B, 9)

    Args:
        in_channels: Number of channels from trunk (C).

    Input:  Tensor of shape (B, C, 9, 9)
    Output: ValueHeadOutput namedtuple
    """

    def __init__(self, in_channels: int):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, 32, kernel_size=1)
        self.bn = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(32 * 81, 512)
        self.fc2 = nn.Linear(512, 128)

        self.val_out = nn.Linear(128, 1)
        self.score_out = nn.Linear(128, 1)
        self.own_out = nn.Linear(128, 9)

    def forward(self, x: torch.Tensor) -> ValueHeadOutput:
        """Forward pass through value head.

        Args:
            x: Trunk features, shape (B, C, 9, 9).

        Returns:
            ValueHeadOutput with win_value, score_margin, ownership.
        """
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        win_value = torch.tanh(self.val_out(x))
        score_margin = torch.tanh(self.score_out(x))
        ownership = torch.sigmoid(self.own_out(x))

        return ValueHeadOutput(
            win_value=win_value,
            score_margin=score_margin,
            ownership=ownership,
        )


if __name__ == "__main__":
    print("=== Episode 6: Value Head ===\n")

    head = ValueHead(in_channels=128)
    x = torch.randn(4, 128, 9, 9)
    output = head(x)

    print(f"Input shape:         {x.shape}")
    print(f"Win value shape:     {output.win_value.shape}")
    print(f"Score margin shape:  {output.score_margin.shape}")
    print(f"Ownership shape:     {output.ownership.shape}")

    assert output.win_value.shape == (4, 1)
    assert output.score_margin.shape == (4, 1)
    assert output.ownership.shape == (4, 9)

    # Check value ranges
    assert output.win_value.min() >= -1.0 and output.win_value.max() <= 1.0
    assert output.score_margin.min() >= -1.0 and output.score_margin.max() <= 1.0
    assert output.ownership.min() >= 0.0 and output.ownership.max() <= 1.0
    print("Value ranges: PASSED")

    num_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    print("\n=== Episode 6 PASSED ===")
