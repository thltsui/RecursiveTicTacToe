"""Episode 3 — Visualizing the Board — Making the Game State Human-Readable

Concept taught: How to render complex nested game states for debugging and teaching.
"""

from __future__ import annotations

import torch
import numpy as np

from .board import GameState, decode_move


def render_board_ascii(state: GameState) -> str:
    """Return ASCII art of the full 9x9 board.

    Format: 9x9 grid with sub-board boundaries drawn as thicker lines.
    Uses 'X' for player 1, 'O' for player -1, '.' for empty.
    Highlights the active sub-board with asterisks.
    Shows sub-board win status.

    Args:
        state: Current game state.

    Returns:
        Multi-line string with the board visualization.
    """
    SYMBOLS = {0: '.', 1: 'X', -1: 'O'}
    lines: list[str] = []
    lines.append(f"  Move {state.move_count} | Player {'X' if state.current_player == 1 else 'O'} to move")
    lines.append(f"  Active sub-board: {'ANY' if state.active_sub_board == -1 else state.active_sub_board}")
    lines.append("")

    for block_row in range(3):
        for cell_row in range(3):
            row_parts: list[str] = []
            for block_col in range(3):
                sb = block_row * 3 + block_col
                is_active = (state.active_sub_board == -1 or state.active_sub_board == sb)
                cells_str: list[str] = []

                for cell_col in range(3):
                    cell_idx = cell_row * 3 + cell_col
                    val = state.cells[sb, cell_idx]

                    # If sub-board is won, show the winner symbol
                    if state.sub_board_results[sb] in (1, -1):
                        cells_str.append(SYMBOLS[int(state.sub_board_results[sb])])
                    elif state.sub_board_results[sb] == 2:
                        cells_str.append('#')
                    else:
                        cells_str.append(SYMBOLS[int(val)])

                part = ' '.join(cells_str)
                if is_active and state.sub_board_results[sb] == 0:
                    part = f'*{part}*'
                else:
                    part = f' {part} '
                row_parts.append(part)

            lines.append(' |'.join(row_parts))

        if block_row < 2:
            lines.append('-------+--------+-------')

    # Sub-board results summary
    lines.append("")
    results_map = {0: '.', 1: 'X', -1: 'O', 2: '#'}
    lines.append("  Sub-board results:")
    for r in range(3):
        row_str = ' '.join(results_map[int(state.sub_board_results[r * 3 + c])] for c in range(3))
        lines.append(f"    {row_str}")

    if state.is_terminal:
        if state.winner == 0:
            lines.append("\n  GAME OVER: Draw!")
        else:
            lines.append(f"\n  GAME OVER: {'X' if state.winner == 1 else 'O'} wins!")

    return '\n'.join(lines)


def render_tensor_channels(tensor: torch.Tensor) -> str:
    """Render all 7 channels of an encoded state tensor as ASCII.

    Shows each channel as a 9x9 grid of 0s and 1s.
    Labels each channel with its meaning.

    Args:
        tensor: shape (7, 9, 9).

    Returns:
        Multi-line string visualization.
    """
    assert tensor.shape == (7, 9, 9), f"Expected (7, 9, 9), got {tensor.shape}"

    channel_names = [
        "Ch 0: Current player pieces",
        "Ch 1: Opponent pieces",
        "Ch 2: Active sub-board mask",
        "Ch 3: Sub-boards won by current player",
        "Ch 4: Sub-boards won by opponent",
        "Ch 5: Drawn sub-boards",
        "Ch 6: Turn indicator",
    ]

    lines: list[str] = []
    for ch in range(7):
        lines.append(f"\n{channel_names[ch]}:")
        for row in range(9):
            row_str = ' '.join(str(int(tensor[ch, row, col].item())) for col in range(9))
            lines.append(f"  {row_str}")

    return '\n'.join(lines)


def render_heatmap(state: GameState, values: torch.Tensor, title: str) -> None:
    """Render a heatmap overlaid on the board using matplotlib.

    Used by explainability modules to show Grad-CAM, visit counts, etc.

    Args:
        state: Current game state (for board context).
        values: shape (9, 9) or (81,) — heatmap values, will be normalized to [0,1].
        title: Plot title.

    Saves to file and displays if possible. Never blocks execution.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if values.dim() == 1:
        values = values.reshape(9, 9)

    vals = values.detach().cpu().numpy().astype(float)
    vmin, vmax = vals.min(), vals.max()
    if vmax > vmin:
        vals = (vals - vmin) / (vmax - vmin)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(vals, cmap='YlOrRd', vmin=0, vmax=1, origin='upper')

    # Draw sub-board boundaries
    for i in range(1, 3):
        ax.axhline(i * 3 - 0.5, color='black', linewidth=2)
        ax.axvline(i * 3 - 0.5, color='black', linewidth=2)

    # Add piece symbols
    SYMBOLS = {1: 'X', -1: 'O'}
    for sb in range(9):
        row_block = sb // 3
        col_block = sb % 3
        for cell in range(9):
            row = row_block * 3 + cell // 3
            col = col_block * 3 + cell % 3
            val = state.cells[sb, cell]
            if val != 0:
                ax.text(col, row, SYMBOLS[val], ha='center', va='center',
                        fontsize=12, fontweight='bold', color='black')

    ax.set_title(title)
    ax.set_xticks(range(9))
    ax.set_yticks(range(9))
    plt.tight_layout()
    filename = title.lower().replace(' ', '_') + '.png'
    plt.savefig(filename, dpi=100)
    plt.close(fig)


def render_mcts_visits(state: GameState, visit_counts: dict[int, int]) -> None:
    """Render MCTS visit counts as a heatmap over the board.

    Args:
        state: Current game state.
        visit_counts: Dict mapping move_idx to visit count.
    """
    values = torch.zeros(81, dtype=torch.float32)
    for move_idx, count in visit_counts.items():
        values[move_idx] = float(count)
    render_heatmap(state, values, "MCTS Visit Counts")


if __name__ == "__main__":
    from .board import create_initial_state, encode_state

    print("=== Episode 3: Visualizer ===\n")

    state = create_initial_state()
    print(render_board_ascii(state))

    tensor = encode_state(state)
    print(render_tensor_channels(tensor))

    print("\n=== Episode 3 PASSED ===")
