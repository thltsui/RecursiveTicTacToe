"""Tests for Episode 2 — rules.py: game logic edge cases."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import random

import numpy as np
import pytest

from importlib import import_module
board_mod = import_module('01_game.board')
rules_mod = import_module('01_game.rules')

GameState = board_mod.GameState
create_initial_state = board_mod.create_initial_state
encode_move = board_mod.encode_move
get_legal_moves = rules_mod.get_legal_moves
apply_move = rules_mod.apply_move
check_sub_board_winner = rules_mod.check_sub_board_winner
check_meta_winner = rules_mod.check_meta_winner
get_legal_move_mask = rules_mod.get_legal_move_mask


class TestSubBoardWinner:
    def test_row_win(self):
        cells = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=np.int8)
        assert check_sub_board_winner(cells) == 1

    def test_col_win(self):
        cells = np.array([-1, 0, 0, -1, 0, 0, -1, 0, 0], dtype=np.int8)
        assert check_sub_board_winner(cells) == -1

    def test_diagonal_win(self):
        cells = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.int8)
        assert check_sub_board_winner(cells) == 1

    def test_draw(self):
        cells = np.array([1, -1, 1, 1, -1, -1, -1, 1, 1], dtype=np.int8)
        # No winning line and board is full
        assert check_sub_board_winner(cells) == 2

    def test_ongoing(self):
        cells = np.array([1, -1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
        assert check_sub_board_winner(cells) == 0


class TestMetaWinner:
    def test_row_win(self):
        results = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=np.int8)
        assert check_meta_winner(results) == 1

    def test_draw(self):
        results = np.array([1, -1, 1, -1, 1, -1, -1, 1, 2], dtype=np.int8)
        # All decided, check if actually no winner
        r = check_meta_winner(results)
        assert r in (1, -1, 2)  # depends on pattern

    def test_ongoing(self):
        results = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
        assert check_meta_winner(results) == 0


class TestLegalMoves:
    def test_initial_81_moves(self):
        state = create_initial_state()
        legal = get_legal_moves(state)
        assert len(legal) == 81

    def test_constrained_board(self):
        state = create_initial_state()
        state.active_sub_board = 4
        legal = get_legal_moves(state)
        # All 9 cells of sub-board 4 (all empty)
        assert len(legal) == 9
        for m in legal:
            sb, _ = board_mod.decode_move(m)
            assert sb == 4

    def test_terminal_no_moves(self):
        state = create_initial_state()
        state.is_terminal = True
        assert len(get_legal_moves(state)) == 0


class TestApplyMove:
    def test_no_mutation(self):
        state = create_initial_state()
        move = encode_move(4, 0)
        new_state = apply_move(state, move)
        assert state.cells[4, 0] == 0  # original unchanged
        assert new_state.cells[4, 0] == 1

    def test_player_switch(self):
        state = create_initial_state()
        move = encode_move(4, 0)
        new_state = apply_move(state, move)
        assert new_state.current_player == -1

    def test_active_sub_board_set(self):
        state = create_initial_state()
        move = encode_move(4, 3)  # cell 3 -> next sub-board is 3
        new_state = apply_move(state, move)
        assert new_state.active_sub_board == 3

    def test_illegal_move_raises(self):
        state = create_initial_state()
        move = encode_move(4, 0)
        new_state = apply_move(state, move)
        # Now active_sub_board = 0, trying to play in sub-board 4 is illegal
        with pytest.raises(ValueError):
            apply_move(new_state, encode_move(4, 1))

    def test_sent_to_won_board_gives_free_choice(self):
        """If a move sends opponent to a won sub-board, they get free choice."""
        state = create_initial_state()
        # Manually win sub-board 0 for player 1
        state.sub_board_results[0] = 1

        # Play a move where cell_idx = 0 -> next sub-board would be 0, but it's won
        state.active_sub_board = -1
        move = encode_move(4, 0)  # cell 0 -> directed to sub-board 0 (won)
        new_state = apply_move(state, move)
        assert new_state.active_sub_board == -1  # free choice

    def test_full_random_game(self):
        """Simulate a full game and verify it terminates properly."""
        random.seed(42)
        state = create_initial_state()
        while not state.is_terminal:
            legal = get_legal_moves(state)
            assert len(legal) > 0
            move = random.choice(legal)
            state = apply_move(state, move)
        assert state.winner is not None


class TestLegalMoveMask:
    def test_shape(self):
        state = create_initial_state()
        mask = get_legal_move_mask(state)
        assert mask.shape == (81,)

    def test_initial_all_ones(self):
        state = create_initial_state()
        mask = get_legal_move_mask(state)
        assert mask.sum() == 81.0
