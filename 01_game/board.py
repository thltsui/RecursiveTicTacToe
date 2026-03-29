"""Episode 1 — Encoding a Game State as a Tensor — The Foundation of Board Game AI

Concept taught: How to represent game state as a multi-channel tensor for neural network
input. Introduces the current-player convention from AlphaZero.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class GameState:
    """Complete game state for Ultimate Tic-Tac-Toe.

    Attributes:
        cells: shape (9, 9) — 0=empty, 1=player1, -1=player2.
            Indexed as (sub_board_idx, cell_idx), both in range [0, 8].
        sub_board_results: shape (9,) — 0=ongoing, 1=p1 won, -1=p2 won, 2=draw.
        active_sub_board: 0-8, or -1 meaning "free choice" (sent to won/drawn board).
        current_player: 1 or -1.
        move_count: Total moves played.
        is_terminal: Whether the game has ended.
        winner: 1, -1, 0 (draw), or None (ongoing).
    """

    cells: np.ndarray = field(default_factory=lambda: np.zeros((9, 9), dtype=np.int8))
    sub_board_results: np.ndarray = field(default_factory=lambda: np.zeros(9, dtype=np.int8))
    active_sub_board: int = -1
    current_player: int = 1
    move_count: int = 0
    is_terminal: bool = False
    winner: int | None = None

    def copy(self) -> GameState:
        """Return a deep copy of this game state."""
        return GameState(
            cells=self.cells.copy(),
            sub_board_results=self.sub_board_results.copy(),
            active_sub_board=self.active_sub_board,
            current_player=self.current_player,
            move_count=self.move_count,
            is_terminal=self.is_terminal,
            winner=self.winner,
        )


def create_initial_state() -> GameState:
    """Create a fresh game. Player 1 goes first. Active sub-board is -1 (free choice).

    Returns:
        A new GameState with an empty board and player 1 to move.
    """
    return GameState()


def encode_state(state: GameState) -> torch.Tensor:
    """Encode game state as tensor for neural network input.

    Always encodes from current player's perspective (current player = 'me').
    This means the network sees the same representation regardless of which
    player it is — it always sees itself as the positive player.

    Channel layout (7 channels, 9x9 spatial):
        Channel 0: Current player's pieces — 1.0 where current player has a piece
        Channel 1: Opponent's pieces — 1.0 where opponent has a piece
        Channel 2: Active sub-board mask — 1.0 for cells of active sub-board(s)
        Channel 3: Sub-boards won by current player — 1.0 for all 9 cells
        Channel 4: Sub-boards won by opponent — 1.0 for all 9 cells
        Channel 5: Drawn/dead sub-boards — 1.0 for all 9 cells of drawn sub-boards
        Channel 6: Turn indicator — all 0.0 if current=player1, all 1.0 if current=player-1

    Spatial layout: (sub_board_idx, cell_idx) -> (row, col) in 9x9:
        row = (sub_board_idx // 3) * 3 + (cell_idx // 3)
        col = (sub_board_idx % 3) * 3 + (cell_idx % 3)

    Args:
        state: GameState dataclass.

    Returns:
        Tensor of shape (7, 9, 9), dtype=torch.float32.
    """
    tensor = torch.zeros(7, 9, 9, dtype=torch.float32)
    cp = state.current_player  # 1 or -1

    for sb in range(9):
        row_block = sb // 3
        col_block = sb % 3
        for cell in range(9):
            row = row_block * 3 + cell // 3
            col = col_block * 3 + cell % 3
            val = state.cells[sb, cell]

            # Channel 0: current player's pieces
            if val == cp:
                tensor[0, row, col] = 1.0
            # Channel 1: opponent's pieces
            elif val == -cp:
                tensor[1, row, col] = 1.0

        # Channel 3: sub-boards won by current player
        if state.sub_board_results[sb] == cp:
            for cell in range(9):
                row = row_block * 3 + cell // 3
                col = col_block * 3 + cell % 3
                tensor[3, row, col] = 1.0

        # Channel 4: sub-boards won by opponent
        if state.sub_board_results[sb] == -cp:
            for cell in range(9):
                row = row_block * 3 + cell // 3
                col = col_block * 3 + cell % 3
                tensor[4, row, col] = 1.0

        # Channel 5: drawn sub-boards
        if state.sub_board_results[sb] == 2:
            for cell in range(9):
                row = row_block * 3 + cell // 3
                col = col_block * 3 + cell % 3
                tensor[5, row, col] = 1.0

    # Channel 2: active sub-board mask
    if state.active_sub_board == -1:
        tensor[2, :, :] = 1.0
    else:
        sb = state.active_sub_board
        row_block = sb // 3
        col_block = sb % 3
        for cell in range(9):
            row = row_block * 3 + cell // 3
            col = col_block * 3 + cell % 3
            tensor[2, row, col] = 1.0

    # Channel 6: turn indicator
    if cp == -1:
        tensor[6, :, :] = 1.0

    return tensor


def decode_move(move_idx: int) -> tuple[int, int]:
    """Convert flat move index (0-80) to (sub_board_idx, cell_idx).

    Args:
        move_idx: Flat index in range [0, 80].

    Returns:
        (sub_board_idx, cell_idx) both in range [0, 8].
    """
    return move_idx // 9, move_idx % 9


def encode_move(sub_board_idx: int, cell_idx: int) -> int:
    """Convert (sub_board_idx, cell_idx) to flat move index (0-80).

    Args:
        sub_board_idx: Sub-board index in range [0, 8].
        cell_idx: Cell index within sub-board in range [0, 8].

    Returns:
        Flat move index in range [0, 80].
    """
    return sub_board_idx * 9 + cell_idx


if __name__ == "__main__":
    # Standalone test for Episode 1
    print("=== Episode 1: Board State Encoding ===\n")

    # 1. Create initial state
    state = create_initial_state()
    print(f"Initial state created: player={state.current_player}, "
          f"active_sub_board={state.active_sub_board}")

    # 2. Encode it as tensor
    tensor = encode_state(state)
    print(f"Tensor shape: {tensor.shape}")

    # 3. Assert shape is (7, 9, 9)
    assert tensor.shape == (7, 9, 9), f"Expected (7, 9, 9), got {tensor.shape}"

    # 4. Assert channel 6 is all zeros (player 1 to move)
    assert tensor[6].sum() == 0.0, "Channel 6 should be all zeros for player 1"

    # 5. Assert channel 2 is all ones (free choice on first move)
    assert tensor[2].sum() == 81.0, "Channel 2 should be all ones for free choice"

    # 6. Print channel sums
    channel_names = [
        "Current player pieces", "Opponent pieces", "Active sub-board mask",
        "Current player sub-boards won", "Opponent sub-boards won",
        "Drawn sub-boards", "Turn indicator"
    ]
    for i, name in enumerate(channel_names):
        print(f"  Channel {i} ({name}): sum = {tensor[i].sum().item():.1f}")

    # Test encode/decode roundtrip
    for idx in [0, 40, 80]:
        sb, cell = decode_move(idx)
        assert encode_move(sb, cell) == idx, f"Roundtrip failed for {idx}"
    print("\nEncode/decode roundtrip: PASSED")

    print("\n=== Episode 1 PASSED ===")
