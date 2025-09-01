"""
Game interfaces package for Ultimate Tic Tac Toe.
"""

from .play_console import play_game, HumanPlayer, RandomPlayer
from .play_random import simulate_game, simulate_multiple_games, save_simulation_results

__all__ = [
    'play_game',
    'HumanPlayer',
    'RandomPlayer',
    'simulate_game',
    'simulate_multiple_games',
    'save_simulation_results'
]
