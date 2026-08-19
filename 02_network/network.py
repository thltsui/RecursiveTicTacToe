"""Episode 7 — Assembling the Full Network — Trunk, Policy Head, and Value Head

Concept taught: How to compose the residual trunk with dual heads. The concept
of shared representations — the trunk learns features useful for BOTH policy and value.
How to count parameters and think about network capacity.
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .residual_block import ResidualBlock
from .policy_head import PolicyHead
from .value_head import ValueHead, ValueHeadOutput

# Allow importing from sibling packages
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@dataclass
class NetworkOutput:
    """Complete output from the neural network.

    Attributes:
        policy_logits: (B, 81) or (81,) — raw policy logits for current player.
        opp_policy_logits: (B, 81) or (81,) — raw policy logits for opponent (auxiliary).
        wdl_logits: (B, 3) or (3,) — raw win/draw/loss logits.
        wdl_probs: (B, 3) or (3,) — win/draw/loss probabilities.
        win_value: (B, 1) or (1,) — P(win) - P(loss), range [-1, 1].
        score_margin: (B, 1) or (1,) — predicted score margin, range [-1, 1].
        ownership: (B, 9) or (9,) — per-sub-board ownership probability, range [0, 1].
    """
    policy_logits:     torch.Tensor  # (B, 81) or (81,)
    opp_policy_logits: torch.Tensor  # (B, 81) or (81,)
    wdl_logits:        torch.Tensor  # (B, 3)  or (3,)
    wdl_probs:         torch.Tensor  # (B, 3)  or (3,)
    win_value:         torch.Tensor  # (B, 1)  or (1,)
    score_margin:      torch.Tensor  # (B, 1)  or (1,)
    ownership:         torch.Tensor  # (B, 9)  or (9,)


class UltimateTTTNetwork(nn.Module):
    """Complete neural network for Ultimate TTT.

    Architecture:
        - Input conv: 7 channels -> C channels
        - Trunk: num_blocks residual blocks, each with global pooling
        - Policy head: outputs current + opponent move distributions
        - Value head: outputs win/draw/loss probabilities + auxiliary targets

    Default hyperparameters (justified by game complexity vs AlphaZero):
        - channels (C) = 128   (AlphaZero uses 256, our game is simpler)
        - num_blocks = 8        (AlphaZero uses 20, sufficient for 9x9)

    Args:
        channels: Feature channels throughout trunk. Default: 128.
        num_blocks: Number of residual blocks. Default: 8.

    Input:  Tensor of shape (B, 7, 9, 9)
    Output: NetworkOutput dataclass
    """

    INPUT_CHANNELS = 7  # 7-channel board encoding

    def __init__(self, channels: int = 128, num_blocks: int = 8):
        super().__init__()

        # Input convolution: 7 -> C channels
        self.input_conv = nn.Conv2d(self.INPUT_CHANNELS, channels, kernel_size=3,
                                     padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)

        # Residual trunk
        self.trunk = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )

        # Dual heads
        self.policy_head = PolicyHead(channels)
        self.value_head = ValueHead(channels)

    def forward(self, x: torch.Tensor) -> NetworkOutput:
        """Forward pass through the full network.

        Args:
            x: Board state tensor, shape (B, 7, 9, 9).

        Returns:
            NetworkOutput with all fields of shape (B, ...).
        """
        # Input processing
        h = self.input_conv(x)          # (B, C, 9, 9)
        h = self.input_bn(h)           # (B, C, 9, 9)
        h = F.relu(h)                  # (B, C, 9, 9)

        # Trunk
        h = self.trunk(h)              # (B, C, 9, 9)

        # Heads
        policy_logits, opp_policy_logits = self.policy_head(h)
        value_out = self.value_head(h)

        return NetworkOutput(
            policy_logits=policy_logits,
            opp_policy_logits=opp_policy_logits,
            wdl_logits=value_out.wdl_logits,
            wdl_probs=value_out.wdl_probs,
            win_value=value_out.win_value,
            score_margin=value_out.score_margin,
            ownership=value_out.ownership,
        )

    def predict(self, state: 'GameState', device: str = 'cpu') -> NetworkOutput:
        """Single-state inference with no gradient tracking.

        Convenience method for use during MCTS. Handles:
        - Encoding the game state to tensor
        - Adding batch dimension
        - Moving to device
        - Running forward pass in torch.no_grad()
        - Removing batch dimension from outputs

        Args:
            state: GameState (not a tensor).
            device: 'cpu' or 'cuda' or 'mps'.

        Returns:
            NetworkOutput with batch dimension removed (shapes without B).
        """
        from importlib import import_module
        board_mod = import_module('01_game.board')
        encode_state = board_mod.encode_state

        tensor = encode_state(state)                    # (7, 9, 9)
        tensor = tensor.unsqueeze(0).to(device)         # (1, 7, 9, 9)

        self.eval()
        with torch.no_grad():
            output = self.forward(tensor)

        # Remove batch dimension
        return NetworkOutput(
            policy_logits=output.policy_logits.squeeze(0),          # (81,)
            opp_policy_logits=output.opp_policy_logits.squeeze(0),  # (81,)
            wdl_logits=output.wdl_logits.squeeze(0),                # (3,)
            wdl_probs=output.wdl_probs.squeeze(0),                  # (3,)
            win_value=output.win_value.squeeze(0),                  # (1,)
            score_margin=output.score_margin.squeeze(0),            # (1,)
            ownership=output.ownership.squeeze(0),                  # (9,)
        )


if __name__ == "__main__":
    print("=== Episode 7: Full Network ===\n")

    # 1. Create network with defaults
    net = UltimateTTTNetwork()

    # 2. Print total parameter count
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params:,}")

    # 3. Run a forward pass with batch size 4
    x = torch.randn(4, 7, 9, 9)
    out = net(x)

    # 4. Assert all output shapes
    assert out.policy_logits.shape == (4, 81), f"Policy: {out.policy_logits.shape}"
    assert out.opp_policy_logits.shape == (4, 81), f"Opp policy: {out.opp_policy_logits.shape}"
    assert out.wdl_logits.shape == (4, 3), f"WDL logits: {out.wdl_logits.shape}"
    assert out.wdl_probs.shape == (4, 3), f"WDL probs: {out.wdl_probs.shape}"
    assert out.win_value.shape == (4, 1), f"Win value: {out.win_value.shape}"
    assert out.score_margin.shape == (4, 1), f"Score margin: {out.score_margin.shape}"
    assert out.ownership.shape == (4, 9), f"Ownership: {out.ownership.shape}"
    print("All output shapes correct")

    # 5. Verify policy logits are not all equal
    assert not torch.allclose(out.policy_logits[0], out.policy_logits[1]), \
        "Policy logits should differ across batch"
    print("Policy logits are non-degenerate")

    # 6. Test predict() on a single GameState
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from importlib import import_module
    board_mod = import_module('01_game.board')
    state = board_mod.create_initial_state()
    pred = net.predict(state)
    assert pred.policy_logits.shape == (81,), f"Predict policy: {pred.policy_logits.shape}"
    assert pred.win_value.shape == (1,), f"Predict value: {pred.win_value.shape}"
    assert pred.ownership.shape == (9,), f"Predict ownership: {pred.ownership.shape}"
    print("predict() shapes correct (no batch dim)")

    print(f"\nNetwork summary:")
    print(f"  Channels: 128, Blocks: 8")
    print(f"  Parameters: {num_params:,}")
    print(f"  Input: (B, 7, 9, 9) -> Policy: (B, 81), Value: (B, 1)")

    print("\n=== Episode 7 PASSED ===")
