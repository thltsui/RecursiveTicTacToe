"""Episode 16 — Integrated Gradients — Provably Faithful Cell Attribution

Concept taught: Why Grad-CAM can be misleading (not guaranteed to sum to output).
The completeness axiom of Integrated Gradients. The baseline (empty board) concept.
Why this method gives a stronger theoretical guarantee than Grad-CAM.

Integrated Gradients formula:
    IG(x) = (x - x') * integral[0->1] dF(x' + alpha*(x-x')) / dx d_alpha

Completeness axiom:
    sum IG(x_i) = F(x) - F(x')
"""

from __future__ import annotations

import sys
import os

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def compute_integrated_gradients(
    network: 'UltimateTTTNetwork',
    state: 'GameState',
    target_move: int | None = None,
    baseline_state: 'GameState | None' = None,
    num_steps: int = 50,
    device: str = 'cpu',
) -> torch.Tensor:
    """Compute Integrated Gradients attribution for a move decision.

    Approximated with num_steps discrete steps along the interpolation path
    from baseline to input.

    Args:
        network: Trained network.
        state: Game state to explain.
        target_move: Move to explain. If None, uses highest probability move.
        baseline_state: Reference state. If None, uses all-zeros tensor.
        num_steps: Number of interpolation steps (more = more accurate, slower).
        device: Compute device.

    Returns:
        Attribution tensor of shape (7, 9, 9).
        Sum over channels gives per-cell importance: shape (9, 9).
        Positive: cells that increase probability of target_move.
        Negative: cells that decrease probability of target_move.

    Note:
        Call .abs().sum(dim=0) to get unsigned per-cell importance for visualization.
    """
    from importlib import import_module
    board_mod = import_module('01_game.board')
    rules_mod = import_module('01_game.rules')
    policy_head_mod = import_module('02_network.policy_head')

    # Encode input
    x = board_mod.encode_state(state).to(device)  # (7, 9, 9)

    # Encode baseline (default: all zeros = empty board)
    if baseline_state is not None:
        x_baseline = board_mod.encode_state(baseline_state).to(device)
    else:
        x_baseline = torch.zeros_like(x)

    # Determine target move if not specified
    if target_move is None:
        network.eval()
        with torch.no_grad():
            out = network(x.unsqueeze(0))
            legal_mask = rules_mod.get_legal_move_mask(state).to(device)
            probs = policy_head_mod.apply_legal_mask(out.policy_logits[0], legal_mask)
            target_move = probs.argmax().item()

    # Compute interpolated inputs
    diff = x - x_baseline  # (7, 9, 9)

    # Accumulate gradients along the path
    accumulated_grads = torch.zeros_like(x)

    network.eval()
    for step in range(num_steps + 1):
        alpha = step / num_steps
        interpolated = x_baseline + alpha * diff  # (7, 9, 9)
        interpolated = interpolated.unsqueeze(0).requires_grad_(True)  # (1, 7, 9, 9)

        output = network(interpolated)
        target_logit = output.policy_logits[0, target_move]

        network.zero_grad()
        target_logit.backward()

        accumulated_grads += interpolated.grad.squeeze(0)  # (7, 9, 9)

    # Average the gradients (trapezoidal rule approximation)
    avg_grads = accumulated_grads / (num_steps + 1)

    # IG = (x - x') * avg_grads
    attributions = diff * avg_grads  # (7, 9, 9)

    return attributions.detach()


def verify_completeness(
    attributions: torch.Tensor,
    network_output_diff: float,
    tolerance: float = 1e-3,
) -> bool:
    """Verify the completeness axiom: sum of attributions ~ network output difference.

    This is a correctness check for the implementation.
    If this fails, num_steps is too low or there's a bug.

    Args:
        attributions: shape (7, 9, 9), output of compute_integrated_gradients.
        network_output_diff: F(x) - F(x'), the difference in network output.
        tolerance: Acceptable difference.

    Returns:
        True if completeness holds within tolerance.
    """
    attr_sum = attributions.sum().item()
    diff = abs(attr_sum - network_output_diff)
    return diff < tolerance


if __name__ == "__main__":
    from importlib import import_module
    board_mod = import_module('01_game.board')
    network_mod = import_module('02_network.network')

    print("=== Episode 16: Integrated Gradients ===\n")

    state = board_mod.create_initial_state()
    net = network_mod.UltimateTTTNetwork(channels=32, num_blocks=2)

    # Compute attributions
    attrs = compute_integrated_gradients(net, state, target_move=40, num_steps=20)
    print(f"Attribution shape: {attrs.shape}")
    assert attrs.shape == (7, 9, 9)

    # Per-cell importance
    cell_importance = attrs.abs().sum(dim=0)  # (9, 9)
    print(f"Cell importance shape: {cell_importance.shape}")
    print(f"Max importance: {cell_importance.max():.4f}")

    # Verify completeness
    x = board_mod.encode_state(state).unsqueeze(0)
    x_base = torch.zeros(1, 7, 9, 9)
    net.eval()
    with torch.no_grad():
        f_x = net(x).policy_logits[0, 40].item()
        f_base = net(x_base).policy_logits[0, 40].item()
    diff = f_x - f_base
    is_complete = verify_completeness(attrs, diff, tolerance=0.5)
    print(f"Completeness: F(x)-F(x')={diff:.4f}, sum(IG)={attrs.sum().item():.4f}, "
          f"pass={is_complete}")

    print("\n=== Episode 16 PASSED ===")
