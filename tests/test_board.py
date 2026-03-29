"""Tests for Episode 1 — board.py: encode_state, decode_move, encode_move."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
import torch

from importlib import import_module
board_mod = import_module('01_game.board')

GameState = board_mod.GameState
create_initial_state = board_mod.create_initial_state
encode_state = board_mod.encode_state
decode_move = board_mod.decode_move
encode_move = board_mod.encode_move


class TestEncodeState:
    def test_initial_state_shape(self):
        state = create_initial_state()
        tensor = encode_state(state)
        assert tensor.shape == (7, 9, 9)
        assert tensor.dtype == torch.float32

    def test_initial_state_channels(self):
        state = create_initial_state()
        tensor = encode_state(state)
        # No pieces placed
        assert tensor[0].sum() == 0.0  # current player pieces
        assert tensor[1].sum() == 0.0  # opponent pieces
        # Free choice -> all ones
        assert tensor[2].sum() == 81.0  # active sub-board mask
        # No sub-boards won
        assert tensor[3].sum() == 0.0
        assert tensor[4].sum() == 0.0
        assert tensor[5].sum() == 0.0
        # Player 1 to move -> channel 6 all zeros
        assert tensor[6].sum() == 0.0

    def test_player_minus1_turn_indicator(self):
        state = create_initial_state()
        state.current_player = -1
        tensor = encode_state(state)
        assert tensor[6].sum() == 81.0  # all ones for player -1

    def test_piece_encoding_perspective(self):
        state = create_initial_state()
        state.cells[4, 4] = 1   # player 1 has a piece
        state.cells[0, 0] = -1  # player -1 has a piece
        state.current_player = 1

        tensor = encode_state(state)
        # From player 1's perspective: player 1's piece on channel 0
        sb, cell = 4, 4
        row = (sb // 3) * 3 + cell // 3
        col = (sb % 3) * 3 + cell % 3
        assert tensor[0, row, col] == 1.0

        # Player -1's piece on channel 1 (opponent)
        sb, cell = 0, 0
        row = (sb // 3) * 3 + cell // 3
        col = (sb % 3) * 3 + cell % 3
        assert tensor[1, row, col] == 1.0

    def test_active_sub_board_specific(self):
        state = create_initial_state()
        state.active_sub_board = 4
        tensor = encode_state(state)
        # Only sub-board 4 should be active (9 cells)
        assert tensor[2].sum() == 9.0


class TestMoveCoding:
    def test_roundtrip(self):
        for idx in range(81):
            sb, cell = decode_move(idx)
            assert encode_move(sb, cell) == idx

    def test_decode_boundaries(self):
        assert decode_move(0) == (0, 0)
        assert decode_move(8) == (0, 8)
        assert decode_move(9) == (1, 0)
        assert decode_move(80) == (8, 8)

    def test_encode_boundaries(self):
        assert encode_move(0, 0) == 0
        assert encode_move(0, 8) == 8
        assert encode_move(8, 8) == 80


class TestGameState:
    def test_copy_independence(self):
        state = create_initial_state()
        state.cells[0, 0] = 1
        copy = state.copy()
        copy.cells[0, 0] = -1
        assert state.cells[0, 0] == 1  # original unchanged
