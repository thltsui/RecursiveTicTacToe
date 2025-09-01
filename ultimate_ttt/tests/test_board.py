"""
Unit tests for Ultimate Tic Tac Toe board logic.
"""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from game.board import UltimateTTTBoard
from game.config import EMPTY, PLAYER_1, PLAYER_2
from game.utils import check_win, get_available_moves, is_board_full


class TestUltimateTTTBoard(unittest.TestCase):
    """Test cases for UltimateTTTBoard class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.board = UltimateTTTBoard()
    
    def test_initial_state(self):
        """Test initial board state."""
        self.assertEqual(self.board.current_player, PLAYER_1)
        self.assertFalse(self.board.game_over)
        self.assertEqual(self.board.winner, 0)
        self.assertEqual(self.board.move_count, 0)
        self.assertEqual(self.board.next_board, -1)
        
        # Check that all cells are empty
        self.assertTrue(np.all(self.board.board == EMPTY))
        self.assertTrue(np.all(self.board.small_board_wins == 0))
        self.assertTrue(np.all(self.board.small_board_full == False))
    
    def test_first_move(self):
        """Test making the first move."""
        # First move can be anywhere
        self.assertTrue(self.board.is_valid_move(4, 6))
        self.assertTrue(self.board.make_move(4, 6))
        
        self.assertEqual(self.board.board[4][6], PLAYER_1)
        self.assertEqual(self.board.current_player, PLAYER_2)
        self.assertEqual(self.board.move_count, 1)
        self.assertEqual(self.board.next_board, 6)  # Next move must be in board 6
    
    def test_move_constraint(self):
        """Test that moves follow the navigation rule."""
        # Make first move in board 4, cell 6
        self.board.make_move(4, 6)
        
        # Next move must be in board 6
        self.assertFalse(self.board.is_valid_move(3, 5))  # Wrong board
        self.assertTrue(self.board.is_valid_move(6, 2))   # Correct board
        
        # Make move in correct board
        self.board.make_move(6, 2)
        self.assertEqual(self.board.next_board, 2)  # Next move must be in board 2
    
    def test_small_board_win(self):
        """Test winning a small board."""
        # Play moves to win board 4
        moves = [(4, 0), (4, 1), (4, 2)]  # Row win
        
        for i, (big_index, small_index) in enumerate(moves):
            player = PLAYER_1 if i % 2 == 0 else PLAYER_2
            if i == 0:
                # First move, any board allowed
                self.board.make_move(big_index, small_index)
            else:
                # Subsequent moves must follow constraint
                self.board.make_move(big_index, small_index)
        
        # Check that board 4 is won by player 1
        self.assertEqual(self.board.small_board_wins[4], PLAYER_1)
        
        # Board 4 should no longer be playable
        self.assertFalse(self.board.is_valid_move(4, 3))
    
    def test_big_board_win(self):
        """Test winning the big board."""
        # Play moves to win three small boards in a row
        # Win boards 0, 1, 2 (top row)
        
        # Board 0: Play in cells 0, 1, 2
        self.board.make_move(0, 0)  # Player 1
        self.board.make_move(0, 1)  # Player 2
        self.board.make_move(0, 2)  # Player 1 wins board 0
        
        # Board 1: Play in cells 0, 1, 2
        self.board.make_move(1, 0)  # Player 2
        self.board.make_move(1, 1)  # Player 1
        self.board.make_move(1, 2)  # Player 2 wins board 1
        
        # Board 2: Play in cells 0, 1, 2
        self.board.make_move(2, 0)  # Player 1
        self.board.make_move(2, 1)  # Player 2
        self.board.make_move(2, 2)  # Player 1 wins board 2
        
        # Check that player 1 won the big board
        self.assertTrue(self.board.game_over)
        self.assertEqual(self.board.winner, PLAYER_1)
    
    def test_draw(self):
        """Test game ending in a draw."""
        # Fill all boards without winning
        for big_index in range(9):
            for small_index in range(9):
                if not self.board.game_over:
                    self.board.make_move(big_index, small_index)
        
        # Game should end in draw
        self.assertTrue(self.board.game_over)
        self.assertEqual(self.board.winner, 0)
        self.assertEqual(self.board.move_count, 81)
    
    def test_invalid_moves(self):
        """Test that invalid moves are rejected."""
        # Try to move in non-empty cell
        self.board.make_move(4, 6)
        self.assertFalse(self.board.is_valid_move(4, 6))
        
        # Try to move in wrong board
        self.board.make_move(6, 2)
        self.assertFalse(self.board.is_valid_move(3, 5))  # Should be in board 2
        
        # Try to move in won board
        # First win board 4
        self.board.make_move(4, 0)
        self.board.make_move(4, 1)
        self.board.make_move(4, 2)  # Player 1 wins board 4
        
        # Try to move in won board
        self.assertFalse(self.board.is_valid_move(4, 3))
    
    def test_get_valid_moves(self):
        """Test getting valid moves."""
        # Initially, all moves are valid
        valid_moves = self.board.get_valid_moves()
        self.assertEqual(len(valid_moves), 81)
        
        # After first move, only moves in specific board are valid
        self.board.make_move(4, 6)
        valid_moves = self.board.get_valid_moves()
        self.assertEqual(len(valid_moves), 8)  # 9 cells in board 6, minus the one played
        
        # All valid moves should be in board 6
        for big_index, small_index in valid_moves:
            self.assertEqual(big_index, 6)
    
    def test_reset(self):
        """Test resetting the board."""
        # Make some moves
        self.board.make_move(4, 6)
        self.board.make_move(6, 2)
        
        # Reset
        self.board.reset()
        
        # Check that board is back to initial state
        self.assertEqual(self.board.current_player, PLAYER_1)
        self.assertFalse(self.board.game_over)
        self.assertEqual(self.board.winner, 0)
        self.assertEqual(self.board.move_count, 0)
        self.assertEqual(self.board.next_board, -1)
        self.assertTrue(np.all(self.board.board == EMPTY))
    
    def test_state_tensor(self):
        """Test state tensor generation."""
        # Make a move
        self.board.make_move(4, 6)
        
        # Get state tensor
        tensor = self.board.get_state_tensor()
        
        # Check tensor shape
        self.assertEqual(tensor.shape, (9, 9, 3))
        
        # Check that cell (4, 6) has player 1 encoding
        self.assertTrue(np.array_equal(tensor[4][6], [0, 1, 0]))
        
        # Check that empty cells have empty encoding
        self.assertTrue(np.array_equal(tensor[0][0], [1, 0, 0]))


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_check_win(self):
        """Test win checking function."""
        from game.config import WINNING_COMBINATIONS
        
        # Test row win
        board = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0])
        winner = check_win(board, WINNING_COMBINATIONS)
        self.assertEqual(winner, 1)
        
        # Test column win
        board = np.array([2, 0, 0, 2, 0, 0, 2, 0, 0])
        winner = check_win(board, WINNING_COMBINATIONS)
        self.assertEqual(winner, 2)
        
        # Test diagonal win
        board = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
        winner = check_win(board, WINNING_COMBINATIONS)
        self.assertEqual(winner, 1)
        
        # Test no win
        board = np.array([1, 2, 1, 2, 1, 2, 2, 1, 2])
        winner = check_win(board, WINNING_COMBINATIONS)
        self.assertEqual(winner, 0)
    
    def test_get_available_moves(self):
        """Test getting available moves."""
        board = np.array([1, 0, 2, 0, 1, 0, 2, 0, 1])
        available = get_available_moves(board)
        self.assertEqual(available, [1, 3, 5, 7])
    
    def test_is_board_full(self):
        """Test board full checking."""
        # Not full
        board = np.array([1, 0, 2, 0, 1, 0, 2, 0, 1])
        self.assertFalse(is_board_full(board))
        
        # Full
        board = np.array([1, 2, 1, 2, 1, 2, 2, 1, 2])
        self.assertTrue(is_board_full(board))


if __name__ == '__main__':
    unittest.main()
