"""
Core game logic for Ultimate Tic Tac Toe.
"""

import numpy as np
from .config import (
    EMPTY, PLAYER_1, PLAYER_2, WINNING_COMBINATIONS, 
    BIG_BOARD_WINNING_COMBINATIONS, BOARD_SIZE, TOTAL_CELLS
)
from .utils import (
    check_win, flatten_board, get_available_moves, 
    is_board_full, get_next_board_index
)


class UltimateTTTBoard:
    """
    Ultimate Tic Tac Toe board implementation.
    """
    
    def __init__(self):
        """Initialize an empty Ultimate Tic Tac Toe board."""
        # 9x9 board where each cell represents a position in the big board
        self.board = np.zeros((9, 9), dtype=int)
        
        # Track which small boards are won (0 = not won, 1 = player 1, 2 = player 2)
        self.small_board_wins = np.zeros(9, dtype=int)
        
        # Track which small boards are full
        self.small_board_full = np.zeros(9, dtype=bool)
        
        # Current player (1 or 2)
        self.current_player = PLAYER_1
        
        # Game state
        self.game_over = False
        self.winner = 0
        self.move_count = 0
        
        # Move history
        self.move_history = []
        
        # Next board constraint (-1 means any board)
        self.next_board = -1
    
    def reset(self):
        """Reset the board to initial state."""
        self.board = np.zeros((9, 9), dtype=int)
        self.small_board_wins = np.zeros(9, dtype=int)
        self.small_board_full = np.zeros(9, dtype=bool)
        self.current_player = PLAYER_1
        self.game_over = False
        self.winner = 0
        self.move_count = 0
        self.move_history = []
        self.next_board = -1
    
    def is_valid_move(self, big_index, small_index):
        """
        Check if a move is valid.
        
        Args:
            big_index: Index of the big board (0-8)
            small_index: Index within the small board (0-8)
            
        Returns:
            True if move is valid, False otherwise
        """
        # Check if game is over
        if self.game_over:
            return False
        
        # Check if big_index is constrained (but allow override if constrained board is full/won)
        if self.next_board != -1 and big_index != self.next_board:
            # Only enforce constraint if the constrained board has available moves
            if (self.small_board_wins[self.next_board] == 0 and 
                not self.small_board_full[self.next_board]):
                return False
        
        # Check if small board is already won or full
        if self.small_board_wins[big_index] != 0 or self.small_board_full[big_index]:
            return False
        
        # Check if small cell is empty
        if self.board[big_index][small_index] != EMPTY:
            return False
        
        return True
    
    def get_valid_moves(self):
        """
        Get all valid moves for the current player.
        
        Returns:
            List of (big_index, small_index) tuples
        """
        valid_moves = []
        
        if self.game_over:
            return valid_moves
        
        # If next_board is constrained, check that board first
        if self.next_board != -1:
            # Check if the constrained board has available moves
            if (self.small_board_wins[self.next_board] == 0 and 
                not self.small_board_full[self.next_board]):
                for small_index in range(9):
                    if self.board[self.next_board][small_index] == EMPTY:
                        valid_moves.append((self.next_board, small_index))
        
        # If no moves in constrained board (or no constraint), check all boards
        if not valid_moves:
            for big_index in range(9):
                if (self.small_board_wins[big_index] == 0 and 
                    not self.small_board_full[big_index]):
                    for small_index in range(9):
                        if self.board[big_index][small_index] == EMPTY:
                            valid_moves.append((big_index, small_index))
        
        return valid_moves
    
    def make_move(self, big_index, small_index):
        """
        Make a move on the board.
        
        Args:
            big_index: Index of the big board (0-8)
            small_index: Index within the small board (0-8)
            
        Returns:
            True if move was successful, False otherwise
        """
        if not self.is_valid_move(big_index, small_index):
            return False
        
        # Make the move
        self.board[big_index][small_index] = self.current_player
        self.move_count += 1
        self.move_history.append((big_index, small_index, self.current_player))
        
        # Check if small board is won
        small_board = self.board[big_index]
        winner = check_win(small_board, WINNING_COMBINATIONS)
        if winner != 0:
            self.small_board_wins[big_index] = winner
        
        # Check if small board is full
        if is_board_full(small_board):
            self.small_board_full[big_index] = True
        
        # Check if big board is won
        big_winner = check_win(self.small_board_wins, BIG_BOARD_WINNING_COMBINATIONS)
        if big_winner != 0:
            self.game_over = True
            self.winner = big_winner
        elif self.move_count == TOTAL_CELLS:
            # Draw
            self.game_over = True
        
        # Set next board constraint
        self.next_board = get_next_board_index(small_index)
        
        # Switch players
        self.current_player = PLAYER_2 if self.current_player == PLAYER_1 else PLAYER_1
        
        return True
    
    def get_state_tensor(self):
        """
        Get the current board state as a tensor.
        
        Returns:
            9x9x3 tensor with one-hot encoding
        """
        from .utils import board_to_state_tensor
        return board_to_state_tensor(self.board)
    
    def get_game_state(self):
        """
        Get the current game state.
        
        Returns:
            Dictionary with game state information
        """
        return {
            'board': self.board.copy(),
            'small_board_wins': self.small_board_wins.copy(),
            'small_board_full': self.small_board_full.copy(),
            'current_player': self.current_player,
            'game_over': self.game_over,
            'winner': self.winner,
            'move_count': self.move_count,
            'next_board': self.next_board,
            'move_history': self.move_history.copy()
        }
    
    def display_board(self):
        """Display the current board state."""
        print("Ultimate Tic Tac Toe Board:")
        print("=" * 50)
        
        for big_row in range(3):
            for small_row in range(3):
                line = ""
                for big_col in range(3):
                    big_index = big_row * 3 + big_col
                    for small_col in range(3):
                        small_index = small_row * 3 + small_col
                        cell = self.board[big_index][small_index]
                        
                        if cell == EMPTY:
                            line += "."
                        elif cell == PLAYER_1:
                            line += "X"
                        else:
                            line += "O"
                        
                        if small_col < 2:
                            line += " "
                    
                    if big_col < 2:
                        line += " | "
                
                print(line)
            
            if big_row < 2:
                print("-" * 50)
        
        print("\nSmall Board Wins:")
        for i in range(3):
            row = ""
            for j in range(3):
                index = i * 3 + j
                if self.small_board_wins[index] == 0:
                    row += "."
                elif self.small_board_wins[index] == PLAYER_1:
                    row += "X"
                else:
                    row += "O"
                if j < 2:
                    row += " "
            print(row)
        
        print(f"\nCurrent Player: {'X' if self.current_player == PLAYER_1 else 'O'}")
        if self.next_board != -1:
            print(f"Next move must be in board: {self.next_board}")
        else:
            print("Next move can be in any board")
        
        if self.game_over:
            if self.winner != 0:
                winner_symbol = 'X' if self.winner == PLAYER_1 else 'O'
                print(f"\nGame Over! {winner_symbol} wins!")
            else:
                print("\nGame Over! It's a draw!")
