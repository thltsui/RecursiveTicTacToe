"""
Game logic package for Ultimate Tic Tac Toe.
"""

from .board import UltimateTTTBoard
from .config import *
from .utils import *

__all__ = [
    'UltimateTTTBoard',
    'EMPTY',
    'PLAYER_1',
    'PLAYER_2',
    'WINNING_COMBINATIONS',
    'BIG_BOARD_WINNING_COMBINATIONS',
    'STATE_ENCODING',
    'BOARD_SIZE',
    'TOTAL_CELLS',
    'MAX_MOVES'
]
