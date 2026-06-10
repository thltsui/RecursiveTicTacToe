"""Episode 2 — Game Logic — Legal Moves, Win Detection, and State Transitions

Concept taught: How to implement complete game rules as pure functions.
The directed-move mechanic of Ultimate TTT is the key teaching moment.
"""

from __future__ import annotations

import numpy as np
import torch

from .board import GameState, decode_move, encode_move

# The 8 winning lines on a 3x3 grid: rows, columns, diagonals
WIN_LINES: list[tuple[int, int, int]] = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
    (0, 4, 8), (2, 4, 6),             # diagonals
]


def check_sub_board_winner(cells_flat: np.ndarray) -> int:
    """Check if a 3x3 sub-board (given as flat array of 9) has a winner.

    Args:
        cells_flat: shape (9,) with values 1, -1, or 0.

    Returns:
        1 if player 1 wins, -1 if player 2 wins, 2 if draw, 0 if ongoing.
    """
    for a, b, c in WIN_LINES:
        if cells_flat[a] != 0 and cells_flat[a] == cells_flat[b] == cells_flat[c]:
            return int(cells_flat[a])
    # No winner — check if full (draw) or ongoing
    if np.all(cells_flat != 0):
        return 2
    return 0


def check_meta_winner(sub_board_results: np.ndarray) -> int:
    """Check if the meta-board (9 sub-board results) has a winner.

    Treats sub_board_results as a 3x3 grid where:
        1 = player 1 won this sub-board
        -1 = player 2 won this sub-board
        2 = draw (does not count for either player in meta-board lines)
        0 = ongoing

    Args:
        sub_board_results: shape (9,).

    Returns:
        1 if player 1 wins meta, -1 if player 2 wins meta,
        2 if full draw (all decided, no winner), 0 if ongoing.
    """
    for a, b, c in WIN_LINES:
        if (sub_board_results[a] in (1, -1)
                and sub_board_results[a] == sub_board_results[b] == sub_board_results[c]):
            return int(sub_board_results[a])
    # No meta-winner — check if all sub-boards are decided
    if np.all(sub_board_results != 0):
        return 2
    return 0


def get_legal_moves(state: GameState) -> list[int]:
    """Return list of legal move indices (0-80).

    Key rules:
        - You must play in the sub-board indicated by active_sub_board.
        - If active_sub_board == -1, you may play in ANY non-terminal sub-board.
        - You may never play in an already-occupied cell.
        - You may never play in a sub-board that is already won or drawn.

    Args:
        state: Current game state.

    Returns:
        List of flat move indices. Never empty unless is_terminal is True.
    """
    if state.is_terminal:
        return []

    moves: list[int] = []

    if state.active_sub_board == -1:
        # Free choice — any non-terminal sub-board
        for sb in range(9):
            if state.sub_board_results[sb] != 0:
                continue
            for cell in range(9):
                if state.cells[sb, cell] == 0:
                    moves.append(encode_move(sb, cell))
    else:
        sb = state.active_sub_board
        if state.sub_board_results[sb] == 0:
            for cell in range(9):
                if state.cells[sb, cell] == 0:
                    moves.append(encode_move(sb, cell))
        # NOTE: If the active sub-board is terminal, this shouldn't happen
        # because apply_move sets active_sub_board = -1 in that case.
        # But as a safety fallback, return all legal moves from non-terminal boards.
        if not moves:
            for sb2 in range(9):
                if state.sub_board_results[sb2] != 0:
                    continue
                for cell in range(9):
                    if state.cells[sb2, cell] == 0:
                        moves.append(encode_move(sb2, cell))

    return moves


def apply_move(state: GameState, move_idx: int) -> GameState:
    """Apply a move and return a NEW game state. Never mutates input state.

    Steps (in order):
        1. Validate move is legal
        2. Place piece in cell
        3. Check if that sub-board is now won or drawn
        4. Update sub_board_results if needed
        5. Check if the meta-board is now won or drawn (terminal check)
        6. Determine next active_sub_board
        7. Flip current_player
        8. Increment move_count

    Args:
        state: Current game state (not mutated).
        move_idx: Flat move index 0-80.

    Returns:
        New GameState after move applied.

    Raises:
        ValueError: If move_idx is not in get_legal_moves(state).
    """
    legal = get_legal_moves(state)
    if move_idx not in legal:
        raise ValueError(
            f"Illegal move {move_idx}. Legal moves: {legal}"
        )

    sb, cell = decode_move(move_idx)
    new_state = state.copy()

    # 2. Place piece
    new_state.cells[sb, cell] = state.current_player

    # 3-4. Check sub-board result
    if new_state.sub_board_results[sb] == 0:
        result = check_sub_board_winner(new_state.cells[sb])
        if result != 0:
            new_state.sub_board_results[sb] = result

    # 5. Check meta-board
    meta = check_meta_winner(new_state.sub_board_results)
    if meta != 0:
        new_state.is_terminal = True
        new_state.winner = 0 if meta == 2 else meta

    # 6. Determine next active_sub_board
    # Next sub-board = cell_idx of the move just played
    next_sb = cell
    if new_state.sub_board_results[next_sb] != 0:
        # That sub-board is already decided — free choice
        new_state.active_sub_board = -1
    else:
        new_state.active_sub_board = next_sb

    # 7. Flip current player
    new_state.current_player = -state.current_player

    # 8. Increment move count
    new_state.move_count = state.move_count + 1

    return new_state


def get_legal_move_mask(state: GameState) -> torch.Tensor:
    """Return binary mask of legal moves for network output masking.

    Args:
        state: Current game state.

    Returns:
        Tensor of shape (81,), dtype=torch.float32.
        1.0 for legal moves, 0.0 for illegal moves.
    """
    mask = torch.zeros(81, dtype=torch.float32)
    for m in get_legal_moves(state):
        mask[m] = 1.0
    return mask


if __name__ == "__main__":
    import random
    from .board import create_initial_state

    print("=== Episode 2: Game Rules ===\n")

    # 1. Create initial state, verify 81 legal moves (free choice on first move)
    state = create_initial_state()
    legal = get_legal_moves(state)
    print(f"Initial legal moves: {len(legal)}")
    assert len(legal) == 81, f"Expected 81, got {len(legal)}"

    # 2. Apply a move to sub-board 4, cell 0 → next active is 0
    move = encode_move(4, 0)
    state2 = apply_move(state, move)
    print(f"After move (sb=4, cell=0): active_sub_board={state2.active_sub_board}")
    assert state2.active_sub_board == 0, f"Expected 0, got {state2.active_sub_board}"

    # Verify original state is not mutated
    assert state.cells[4, 0] == 0, "Original state was mutated!"
    assert state2.cells[4, 0] == 1, "New state should have piece placed"

    # 3. Simulate a full random game to terminal state
    s = create_initial_state()
    moves_played = 0
    while not s.is_terminal:
        legal = get_legal_moves(s)
        assert len(legal) > 0, "Non-terminal state has no legal moves"
        m = random.choice(legal)
        s = apply_move(s, m)
        moves_played += 1
    print(f"Random game finished in {moves_played} moves")

    # 4. Verify terminal state
    assert s.is_terminal, "Game should be terminal"
    assert s.winner is not None, "Terminal game should have a winner/draw result"
    print(f"Winner: {s.winner}")

    # 5. Verify legal moves is empty on terminal state
    assert len(get_legal_moves(s)) == 0, "Terminal state should have no legal moves"

    # Test legal move mask
    state3 = create_initial_state()
    mask = get_legal_move_mask(state3)
    assert mask.shape == (81,), f"Expected (81,), got {mask.shape}"
    assert mask.sum() == 81.0, "Initial state mask should have 81 ones"

    print("\n=== Episode 2 PASSED ===")
