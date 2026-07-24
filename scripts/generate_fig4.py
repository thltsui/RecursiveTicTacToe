import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import importlib

board = importlib.import_module('01_game.board')
GameState = board.GameState
create_initial_state = board.create_initial_state
encode_state = board.encode_state

state = create_initial_state()
state.current_player = -1 # O to move
state.move_count = 5

# O's pieces (Ch 0 from O's perspective)
state.cells[4, 4] = -1
state.cells[4, 0] = -1

# X's pieces (Ch 1 from O's perspective)
state.cells[0, 0] = 1
state.cells[0, 1] = 1
state.cells[0, 2] = 1
state.sub_board_results[0] = 1 # X won sb 0
state.cells[4, 2] = 1

# Active sub-board (O must play in sb 2)
state.active_sub_board = 2

# Encode state (from O's perspective because O is to move)
tensor = encode_state(state)

# Plotting: portrait mode (7 rows, 1 column)
# Using constrained_layout to prevent titles from stacking on top of each other
fig, axes = plt.subplots(7, 1, figsize=(4, 24), constrained_layout=True)
channel_names = [
    "Ch 0: O's pieces (Current Player)",
    "Ch 1: X's pieces (Opponent)",
    "Ch 2: Active sub-board mask",
    "Ch 3: Sub-boards won by O",
    "Ch 4: Sub-boards won by X",
    "Ch 5: Drawn sub-boards",
    "Ch 6: Turn indicator (All 1s for O)",
]

for ch in range(7):
    ax = axes[ch]
    vals = tensor[ch].numpy()
    # Adding an inner grid to visually separate the 9x9 cells
    ax.imshow(vals, cmap='Blues' if ch % 2 == 0 else 'Reds', vmin=0, vmax=1, origin='upper')
    
    # Draw sub-board boundaries (thick lines)
    for i in range(1, 3):
        ax.axhline(i * 3 - 0.5, color='black', linewidth=2.5)
        ax.axvline(i * 3 - 0.5, color='black', linewidth=2.5)
        
    # Draw minor grid lines (thin lines for individual cells)
    for i in range(1, 9):
        if i % 3 != 0:
            ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.5)
            ax.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.5)
            
    ax.set_title(channel_names[ch], fontsize=14, pad=12)
    ax.set_xticks([])
    ax.set_yticks([])

plt.savefig('substack/figures/fig4_tensor_channels.png', dpi=150, bbox_inches='tight')
print("fig4_tensor_channels.png generated successfully.")
