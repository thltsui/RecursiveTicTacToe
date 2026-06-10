"""Episode 15 — Grad-CAM — Which Cells Is the AI Looking At?

Concept taught: How gradients flowing into the last convolutional layer reveal
what spatial regions the network focuses on for a given decision.
Why preserving 9x9 spatial dimensions throughout the network was architecturally critical.
"""

from __future__ import annotations

import sys
import os

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def compute_gradcam(
    network: 'UltimateTTTNetwork',
    state: 'GameState',
    target_move: int | None = None,
    device: str = 'cpu',
) -> torch.Tensor:
    """Compute Grad-CAM heatmap for a specific move decision.

    Algorithm:
        1. Encode state to tensor, enable gradients.
        2. Forward pass through network.
        3. Select target logit (policy_logits[target_move]).
        4. Backward pass to get gradients at last conv layer.
        5. alpha_k = mean of gradients over spatial dims (9x9) -> shape (C,).
        6. heatmap = ReLU(sum_k alpha_k * feature_map_k) -> shape (9, 9).
        7. Normalize heatmap to [0, 1].

    Args:
        network: Trained network.
        state: Game state to analyze.
        target_move: Move index (0-80) to explain. If None, uses the
                     move with highest policy probability.
        device: Compute device.

    Returns:
        Heatmap tensor of shape (9, 9), values in [0, 1].
        Higher values = cells that most influenced the decision.
    """
    from importlib import import_module
    board_mod = import_module('01_game.board')
    rules_mod = import_module('01_game.rules')
    policy_head_mod = import_module('02_network.policy_head')

    # 1. Encode and setup
    tensor = board_mod.encode_state(state).unsqueeze(0).to(device)  # (1, 7, 9, 9)
    tensor.requires_grad_(True)

    network.eval()

    # Hook to capture activations from the last residual block
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations['last'] = output

    def backward_hook(module, grad_input, grad_output):
        gradients['last'] = grad_output[0]

    # Register hooks on the last trunk block
    last_block = network.trunk[-1]
    fh = last_block.register_forward_hook(forward_hook)
    bh = last_block.register_full_backward_hook(backward_hook)

    try:
        # 2. Forward pass
        output = network(tensor)

        # 3. Select target
        if target_move is None:
            legal_mask = rules_mod.get_legal_move_mask(state).to(device)
            probs = policy_head_mod.apply_legal_mask(output.policy_logits[0], legal_mask)
            target_move = probs.argmax().item()

        target_logit = output.policy_logits[0, target_move]

        # 4. Backward
        network.zero_grad()
        target_logit.backward()

        # 5. Compute weights: mean gradient over spatial dims
        grads = gradients['last']               # (1, C, 9, 9)
        alpha = grads.mean(dim=[2, 3])          # (1, C)

        # 6. Weighted combination
        feats = activations['last']             # (1, C, 9, 9)
        weighted = (alpha.unsqueeze(-1).unsqueeze(-1) * feats).sum(dim=1)  # (1, 9, 9)
        heatmap = F.relu(weighted).squeeze(0)   # (9, 9)

        # 7. Normalize to [0, 1]
        hmin, hmax = heatmap.min(), heatmap.max()
        if hmax > hmin:
            heatmap = (heatmap - hmin) / (hmax - hmin)
        else:
            heatmap = torch.zeros(9, 9)

    finally:
        fh.remove()
        bh.remove()

    return heatmap.detach()


def visualize_gradcam(
    state: 'GameState',
    heatmap: torch.Tensor,
    target_move: int,
    save_path: str | None = None,
) -> None:
    """Overlay Grad-CAM heatmap on board visualization.

    Shows:
        - Board state with pieces.
        - Heatmap overlay (red = high importance, blue = low).
        - Highlighted target move.
        - Title with move description.

    Args:
        state: Current game state.
        heatmap: shape (9, 9), values in [0, 1].
        target_move: The move being explained.
        save_path: If provided, save figure to this path.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from importlib import import_module
    board_mod = import_module('01_game.board')
    sb, cell = board_mod.decode_move(target_move)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(heatmap.numpy(), cmap='jet', vmin=0, vmax=1, origin='upper')

    # Draw sub-board grid
    for i in range(1, 3):
        ax.axhline(i * 3 - 0.5, color='white', linewidth=2)
        ax.axvline(i * 3 - 0.5, color='white', linewidth=2)

    # Mark target move
    target_row = (sb // 3) * 3 + cell // 3
    target_col = (sb % 3) * 3 + cell % 3
    ax.plot(target_col, target_row, 'w*', markersize=20)

    ax.set_title(f"Grad-CAM: Move {target_move} (sb={sb}, cell={cell})")
    ax.set_xticks(range(9))
    ax.set_yticks(range(9))
    plt.tight_layout()

    path = save_path or 'gradcam.png'
    plt.savefig(path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    from importlib import import_module
    board_mod = import_module('01_game.board')
    network_mod = import_module('02_network.network')

    print("=== Episode 15: Grad-CAM ===\n")

    state = board_mod.create_initial_state()
    net = network_mod.UltimateTTTNetwork(channels=32, num_blocks=2)

    heatmap = compute_gradcam(net, state, target_move=40, device='cpu')
    print(f"Heatmap shape: {heatmap.shape}")
    assert heatmap.shape == (9, 9)
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0
    print(f"Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")

    # Auto-select move
    heatmap2 = compute_gradcam(net, state, device='cpu')
    assert heatmap2.shape == (9, 9)
    print("Auto-select move: PASSED")

    print("\n=== Episode 15 PASSED ===")
