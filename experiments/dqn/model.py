"""Q-Network for Ultimate Tic-Tac-Toe.

Input:  (batch, 7, 9, 9) float32 tensor from encode_state()
Output: (batch, 81) Q-values, one per possible move
"""
from __future__ import annotations

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Three convolutional layers over the 7-channel board encoding,
    followed by a two-layer fully-connected head.

    Architecture:
        Conv(7→32, 3×3, pad=1) → ReLU
        Conv(32→64, 3×3, pad=1) → ReLU
        Conv(64→64, 3×3, pad=1) → ReLU
        Flatten → Linear(5184→256) → ReLU → Linear(256→81)

    The spatial structure is preserved through the conv layers; the board
    stays (9, 9) throughout (stride=1, padding=1).
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(7,  32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 9 * 9, 256), nn.ReLU(),
            nn.Linear(256, 81),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 7, 9, 9) board tensor.
        Returns:
            (batch, 81) Q-values (raw logits, not masked).
        """
        return self.fc(self.conv(x))

    def q_masked(self, x: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        """Return Q-values with illegal moves set to -inf.

        Args:
            x:           (batch, 7, 9, 9)
            legal_mask:  (batch, 81) binary float — 1 where legal
        Returns:
            (batch, 81) Q-values, illegal entries = -inf
        """
        q = self.forward(x)
        return q.masked_fill(legal_mask == 0, float("-inf"))
