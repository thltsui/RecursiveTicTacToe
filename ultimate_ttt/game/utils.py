"""
Utility functions for Ultimate Tic Tac Toe.
"""

import numpy as np
from .config import WINNING_COMBINATIONS, BIG_BOARD_WINNING_COMBINATIONS


def check_win(board, combinations):
    """
    Check if a player has won on a board.
    
    Args:
        board: 1D array representing the board state
        combinations: List of winning combinations to check
        
    Returns:
        Player number if won, 0 if no win
    """
    for combo in combinations:
        if (board[combo[0]] != 0 and 
            board[combo[0]] == board[combo[1]] == board[combo[2]]):
            return board[combo[0]]
    return 0


def flatten_board(board):
    """
    Flatten a 2D board to 1D for easier processing.
    
    Args:
        board: 2D numpy array
        
    Returns:
        1D numpy array
    """
    return board.flatten()


def unflatten_board(board_1d):
    """
    Convert a 1D board back to 2D.
    
    Args:
        board_1d: 1D numpy array
        
    Returns:
        2D numpy array
    """
    return board_1d.reshape(3, 3)


def get_available_moves(board):
    """
    Get list of available moves on a board.
    
    Args:
        board: 1D array representing the board state
        
    Returns:
        List of indices where moves can be made
    """
    return [i for i, cell in enumerate(board) if cell == 0]


def is_board_full(board):
    """
    Check if a board is full.
    
    Args:
        board: 1D array representing the board state
        
    Returns:
        True if board is full, False otherwise
    """
    return 0 not in board


def get_next_board_index(small_cell_index):
    """
    Get the next board index based on the small cell index.
    
    Args:
        small_cell_index: Index of the cell played (0-8)
        
    Returns:
        Index of the next board to play in
    """
    return small_cell_index


def board_to_state_tensor(board_state):
    """
    Convert board state to one-hot encoded tensor.
    
    Args:
        board_state: 9x9 array representing the full game state
        
    Returns:
        9x9x3 tensor with one-hot encoding
    """
    from .config import STATE_ENCODING
    
    tensor = np.zeros((9, 9, 3))
    for i in range(9):
        for j in range(9):
            player = board_state[i][j]
            tensor[i][j] = STATE_ENCODING[player]
    
    return tensor


def state_tensor_to_board(tensor):
    """
    Convert one-hot encoded tensor back to board state.
    
    Args:
        tensor: 9x9x3 tensor
        
    Returns:
        9x9 array representing the board state
    """
    board = np.zeros((9, 9), dtype=int)
    for i in range(9):
        for j in range(9):
            if tensor[i][j][1] == 1:  # Player 1
                board[i][j] = 1
            elif tensor[i][j][2] == 1:  # Player 2
                board[i][j] = 2
            # Default is 0 (empty)
    
    return board
