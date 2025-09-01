"""
Configuration constants and index maps for Ultimate Tic Tac Toe.
"""

# Board indices for small boards (0-8)
SMALL_BOARD_INDICES = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]
]

# Big board indices (0-8)
BIG_BOARD_INDICES = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]
]

# Win combinations for small boards
WINNING_COMBINATIONS = [
    # Rows
    [0, 1, 2], [3, 4, 5], [6, 7, 8],
    # Columns
    [0, 3, 6], [1, 4, 7], [2, 5, 8],
    # Diagonals
    [0, 4, 8], [2, 4, 6]
]

# Win combinations for big board
BIG_BOARD_WINNING_COMBINATIONS = [
    # Rows
    [0, 1, 2], [3, 4, 5], [6, 7, 8],
    # Columns
    [0, 3, 6], [1, 4, 7], [2, 5, 8],
    # Diagonals
    [0, 4, 8], [2, 4, 6]
]

# Player constants
EMPTY = 0
PLAYER_1 = 1
PLAYER_2 = 2

# State encoding
STATE_ENCODING = {
    EMPTY: [1, 0, 0],
    PLAYER_1: [0, 1, 0],
    PLAYER_2: [0, 0, 1]
}

# Game constants
BOARD_SIZE = 3
TOTAL_CELLS = 81
MAX_MOVES = 81
