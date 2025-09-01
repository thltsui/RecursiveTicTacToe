"""
Ultimate Tic Tac Toe - An advanced variant of classic Tic Tac Toe.
"""

__version__ = "1.0.0"
__author__ = "Ultimate Tic Tac Toe Team"

from .game.board import UltimateTTTBoard
from .game.config import EMPTY, PLAYER_1, PLAYER_2

__all__ = [
    'UltimateTTTBoard',
    'EMPTY',
    'PLAYER_1', 
    'PLAYER_2'
]
