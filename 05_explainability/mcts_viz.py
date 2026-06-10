"""Episode 17 — Visualizing the AI's Thinking — MCTS Visit Counts as a Heatmap

Concept taught: How MCTS visit counts naturally give interpretable explanations.
This is the most intuitive explainability method — no gradient math required.
Value delta per move as causal attribution.
"""

from __future__ import annotations

import sys
import os

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def compute_visit_heatmap(
    root: 'MCTSNode',
    normalize: bool = True,
) -> torch.Tensor:
    """Convert MCTS visit counts to a 9x9 heatmap.

    Args:
        root: Root MCTS node after search has been run.
        normalize: If True, normalize to [0, 1]. If False, return raw counts.

    Returns:
        Tensor of shape (9, 9). Zero for unvisited/illegal moves.
    """
    from importlib import import_module
    board_mod = import_module('01_game.board')

    heatmap = torch.zeros(9, 9, dtype=torch.float32)
    visits = root.get_visit_counts()

    for move_idx, count in visits.items():
        sb, cell = board_mod.decode_move(move_idx)
        row = (sb // 3) * 3 + cell // 3
        col = (sb % 3) * 3 + cell % 3
        heatmap[row, col] = float(count)

    if normalize and heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap


def compute_value_delta_heatmap(
    state: 'GameState',
    network: 'UltimateTTTNetwork',
    device: str = 'cpu',
) -> torch.Tensor:
    """Compute value delta for each legal move.

    For each legal move m:
        delta_value(m) = V(state_after_m) - V(state)

    Where V is the network's win_value estimate (negated for the opponent
    since value is from current player's perspective).

    This shows CAUSAL attribution — which moves the AI expects to
    most improve (or worsen) the position.

    Args:
        state: Current game state.
        network: Trained network.
        device: Compute device.

    Returns:
        Tensor of shape (9, 9).
        Positive: move improves position.
        Negative: move worsens position.
        Zero: illegal move.
    """
    from importlib import import_module
    board_mod = import_module('01_game.board')
    rules_mod = import_module('01_game.rules')

    heatmap = torch.zeros(9, 9, dtype=torch.float32)

    # Get current value
    current_output = network.predict(state, device=device)
    current_value = current_output.win_value.item()

    legal_moves = rules_mod.get_legal_moves(state)

    for move_idx in legal_moves:
        next_state = rules_mod.apply_move(state, move_idx)

        if next_state.is_terminal:
            # Use actual outcome
            winner = next_state.winner if next_state.winner is not None else 0
            # From current player's perspective at original state
            next_value = float(winner * state.current_player)
        else:
            next_output = network.predict(next_state, device=device)
            # Negate because next state is from opponent's perspective
            next_value = -next_output.win_value.item()

        delta = next_value - current_value

        sb, cell = board_mod.decode_move(move_idx)
        row = (sb // 3) * 3 + cell // 3
        col = (sb % 3) * 3 + cell % 3
        heatmap[row, col] = delta

    return heatmap


def render_three_panel_explanation(
    state: 'GameState',
    root: 'MCTSNode',
    network: 'UltimateTTTNetwork',
    chosen_move: int,
    device: str = 'cpu',
    save_path: str | None = None,
) -> None:
    """Render a three-panel explanation figure using matplotlib.

    Panel 1: MCTS visit count heatmap — "What the AI considered".
    Panel 2: Grad-CAM heatmap — "What features drove the choice".
    Panel 3: Value delta heatmap — "What the AI expects to gain".

    All three panels show the board state with the chosen move highlighted.

    Args:
        state: Current game state.
        root: MCTS root node after search.
        network: Trained network.
        chosen_move: The move that was selected.
        device: Compute device.
        save_path: If provided, save figure to this path.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from .gradcam import compute_gradcam
    from importlib import import_module
    board_mod = import_module('01_game.board')

    # Compute all three heatmaps
    visit_hm = compute_visit_heatmap(root, normalize=True)
    gradcam_hm = compute_gradcam(network, state, target_move=chosen_move, device=device)
    value_hm = compute_value_delta_heatmap(state, network, device=device)

    # Target position
    sb, cell = board_mod.decode_move(chosen_move)
    target_row = (sb // 3) * 3 + cell // 3
    target_col = (sb % 3) * 3 + cell % 3

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    titles = [
        "MCTS Visits\n(What the AI considered)",
        "Grad-CAM\n(What features drove the choice)",
        "Value Delta\n(What the AI expects to gain)",
    ]
    heatmaps = [visit_hm, gradcam_hm, value_hm]
    cmaps = ['YlOrRd', 'jet', 'RdBu_r']

    for ax, hm, title, cmap in zip(axes, heatmaps, titles, cmaps):
        im = ax.imshow(hm.numpy(), cmap=cmap, origin='upper')
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Sub-board grid
        for i in range(1, 3):
            ax.axhline(i * 3 - 0.5, color='white', linewidth=2)
            ax.axvline(i * 3 - 0.5, color='white', linewidth=2)

        # Mark chosen move
        ax.plot(target_col, target_row, 'w*', markersize=15)

        ax.set_title(title)
        ax.set_xticks(range(9))
        ax.set_yticks(range(9))

    plt.tight_layout()
    path = save_path or 'three_panel_explanation.png'
    plt.savefig(path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    from importlib import import_module
    board_mod = import_module('01_game.board')
    network_mod = import_module('02_network.network')
    search_mod = import_module('03_mcts.search')

    print("=== Episode 17: MCTS Visualization ===\n")

    state = board_mod.create_initial_state()
    net = network_mod.UltimateTTTNetwork(channels=32, num_blocks=2)

    # Run MCTS
    root = search_mod.run_mcts(state, net, num_simulations=30, device='cpu')

    # Visit heatmap
    hm = compute_visit_heatmap(root)
    print(f"Visit heatmap shape: {hm.shape}")
    assert hm.shape == (9, 9)
    assert hm.max() <= 1.0

    # Value delta
    vd = compute_value_delta_heatmap(state, net, device='cpu')
    print(f"Value delta shape: {vd.shape}")
    assert vd.shape == (9, 9)

    print("\n=== Episode 17 PASSED ===")
