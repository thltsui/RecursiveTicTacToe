"""Episode 4 — The Residual Block with Global Pooling — KataGo's Key Innovation

Concept taught: ResNets and why skip connections work. The KataGo global pooling
injection that lets local convolutions be conditioned on global board state.
This is the most technically dense episode — spend extra time on the global pooling math.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """A single residual block with global average pooling injection.

    This is the core building block of the network trunk. It preserves
    spatial dimensions (9x9) while allowing global context to influence
    local features.

    The global pooling branch is the key KataGo domain-independent improvement:
    it allows the network to condition local features on the global board state
    without any game-specific knowledge being hardcoded.

    Architecture:
        Branch 1 (local):
            input -> Conv(C,C,3,pad=1) -> BN -> ReLU -> Conv(C,C,3,pad=1) -> BN

        Branch 2 (global pooling injection):
            input -> GlobalAvgPool -> Linear(C, C//8) -> ReLU -> Linear(C//8, C) -> broadcast

        Output: ReLU(Branch1 + Branch2 + input)

    Args:
        channels: Number of feature channels (C). Must be consistent throughout trunk.
        global_channels: Size of global pooling bottleneck. Default: channels // 8.

    Input:  Tensor of shape (B, C, 9, 9)
    Output: Tensor of shape (B, C, 9, 9)  — spatial dims preserved
    """

    def __init__(self, channels: int, global_channels: int | None = None):
        super().__init__()
        if global_channels is None:
            global_channels = channels // 8

        # Branch 1: local convolutional path
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        # Branch 2: global pooling injection (KataGo)
        self.global_mlp = nn.Sequential(
            nn.Linear(channels, global_channels),
            nn.ReLU(inplace=True),
            nn.Linear(global_channels, channels),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block.

        Args:
            x: shape (B, C, 9, 9)

        Returns:
            shape (B, C, 9, 9)
        """
        identity = x  # (B, C, 9, 9)

        # Branch 1: local convolutions
        local = self.conv1(x)           # (B, C, 9, 9)
        local = self.bn1(local)         # (B, C, 9, 9)
        local = self.relu(local)        # (B, C, 9, 9)
        local = self.conv2(local)       # (B, C, 9, 9)
        local = self.bn2(local)         # (B, C, 9, 9)

        # Branch 2: global pooling injection
        pooled = x.mean(dim=[2, 3])     # (B, C)
        global_vec = self.global_mlp(pooled)  # (B, C)
        global_broadcast = global_vec.unsqueeze(-1).unsqueeze(-1).expand_as(x)  # (B, C, 9, 9)

        # Combine: skip connection + local + global
        out = self.relu(local + global_broadcast + identity)  # (B, C, 9, 9)
        return out


if __name__ == "__main__":
    print("=== Episode 4: Residual Block ===\n")

    # 1. Create a ResidualBlock with channels=128
    block = ResidualBlock(channels=128)
    print(f"Block created with channels=128")

    # 2. Pass a random tensor through
    x = torch.randn(4, 128, 9, 9)
    y = block(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")

    # 3. Assert output shape
    assert y.shape == (4, 128, 9, 9), f"Expected (4, 128, 9, 9), got {y.shape}"

    # 4. Assert output is not equal to input
    assert not torch.allclose(x, y), "Output should differ from input"

    # 5. Count and print trainable parameters
    num_params = sum(p.numel() for p in block.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    # 6. Verify gradients flow through both branches
    x2 = torch.randn(2, 128, 9, 9, requires_grad=True)
    y2 = block(x2)
    loss = y2.sum()
    loss.backward()
    assert x2.grad is not None, "Gradients should flow to input"
    assert x2.grad.abs().sum() > 0, "Gradients should be non-zero"
    print("Gradient flow: PASSED")

    print("\n=== Episode 4 PASSED ===")
