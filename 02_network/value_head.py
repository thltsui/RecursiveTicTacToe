"""Episode 6 — The Value Head — Teaching the Network to Evaluate Positions

Concept taught: Multi-task learning from a single position. How auxiliary targets
(score margin, sub-board ownership) give richer training signal than win/loss alone.
Why an explicit categorical outcome is safer than folding draws into a scalar.
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
        wdl_logits: (B, 3) — unnormalized logits ordered win, draw, loss.
        wdl_probs: (B, 3) — W/D/L probabilities, optionally temperature-scaled.
        win_value: (B, 1) — zero-sum expectation P(win) - P(loss).
        score_margin: (B, 1) — auxiliary, range [-1, 1]. Predicted sub-board differential.
        ownership: (B, 9) — auxiliary, range [0, 1]. Per-sub-board win probability.
    """
    wdl_logits: torch.Tensor    # (B, 3)
    wdl_probs: torch.Tensor     # (B, 3)
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
                 |-> Linear(128, 3) -> Softmax -> W/D/L          (B, 3)
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
        self.ln = nn.LayerNorm(128, elementwise_affine=False)

        self.val_out = nn.Linear(128, 3)
        self.score_out = nn.Linear(128, 1)
        self.own_out = nn.Linear(128, 9)

        # Fitted on a held-out calibration set after network training.  Keeping
        # this as a buffer makes the calibration part of the deployed artifact.
        self.register_buffer("wdl_temperature", torch.tensor(1.0))
        self.register_buffer("wdl_is_calibrated", torch.tensor(False))
        self.register_buffer("wdl_is_native", torch.tensor(True))
        self.loaded_legacy_scalar_head = False

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Migrate historical one-output value heads without hiding the fact.

        Old checkpoints stored a single tanh pre-activation in ``val_out``.
        Symmetric win/loss logits with a neutral draw logit preserve its ordering
        and approximate its scalar expectation.  These probabilities must still
        be marked uncalibrated until the checkpoint is retrained and temperature
        scaled.
        """
        weight_key = prefix + "val_out.weight"
        bias_key = prefix + "val_out.bias"
        temperature_key = prefix + "wdl_temperature"
        calibrated_key = prefix + "wdl_is_calibrated"
        native_key = prefix + "wdl_is_native"

        self.loaded_legacy_scalar_head = False
        old_weight = state_dict.get(weight_key)
        if old_weight is not None and old_weight.shape == (1, self.val_out.in_features):
            scale = 1.5  # matches tanh's slope near zero after a 3-way softmax
            state_dict[weight_key] = torch.cat(
                [scale * old_weight, torch.zeros_like(old_weight), -scale * old_weight],
                dim=0,
            )
            old_bias = state_dict.get(bias_key)
            if old_bias is not None and old_bias.shape == (1,):
                state_dict[bias_key] = torch.cat(
                    [scale * old_bias, torch.zeros_like(old_bias), -scale * old_bias]
                )
            self.loaded_legacy_scalar_head = True
            state_dict[native_key] = torch.tensor(False)

        # The temperature buffer did not exist in historical checkpoints.
        if temperature_key not in state_dict:
            state_dict[temperature_key] = self.wdl_temperature.detach().clone()
        if calibrated_key not in state_dict:
            state_dict[calibrated_key] = self.wdl_is_calibrated.detach().clone()
        if native_key not in state_dict:
            state_dict[native_key] = self.wdl_is_native.detach().clone()

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: torch.Tensor) -> ValueHeadOutput:
        """Forward pass through value head.

        Args:
            x: Trunk features, shape (B, C, 9, 9).

        Returns:
            ValueHeadOutput with W/D/L, scalar expectation, and auxiliaries.
        """
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.ln(x)

        wdl_logits = self.val_out(x)
        temperature = self.wdl_temperature.clamp_min(1e-3)
        wdl_probs = F.softmax(wdl_logits / temperature, dim=-1)
        win_value = (wdl_probs[:, 0] - wdl_probs[:, 2]).unsqueeze(-1)
        score_margin = torch.tanh(self.score_out(x))
        ownership = torch.sigmoid(self.own_out(x))

        return ValueHeadOutput(
            wdl_logits=wdl_logits,
            wdl_probs=wdl_probs,
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
    assert output.wdl_logits.shape == (4, 3)
    assert output.wdl_probs.shape == (4, 3)
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
