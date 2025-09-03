#!/usr/bin/env python3
"""
Debug script to test win detection step by step.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game.board import UltimateTTTBoard
from game.config import PLAYER_1, PLAYER_2

def debug_win_detection():
    """Debug the win detection step by step."""
    print("Debugging win detection...")
    
    board = UltimateTTTBoard()
    
    print(f"Initial player: {board.current_player}")
    
    # Make first move
    print("\n1. Making first move: board 4, cell 6")
    board.make_move(4, 6)
    print(f"   Current player: {board.current_player}")
    print(f"   Next board constraint: {board.next_board}")
    
    # Make second move
    print("\n2. Making second move: board 6, cell 2")
    board.make_move(6, 2)
    print(f"   Current player: {board.current_player}")
    print(f"   Next board constraint: {board.next_board}")
    
    # Make third move
    print("\n3. Making third move: board 2, cell 0")
    board.make_move(2, 0)
    print(f"   Current player: {board.current_player}")
    print(f"   Next board constraint: {board.next_board}")
    
    # Make fourth move
    print("\n4. Making fourth move: board 0, cell 1")
    board.make_move(0, 1)
    print(f"   Current player: {board.current_player}")
    print(f"   Next board constraint: {board.next_board}")
    
    # Make fifth move
    print("\n5. Making fifth move: board 1, cell 2")
    board.make_move(1, 2)
    print(f"   Current player: {board.current_player}")
    print(f"   Next board constraint: {board.next_board}")
    
    # Make sixth move
    print("\n6. Making sixth move: board 2, cell 1")
    board.make_move(2, 1)
    print(f"   Current player: {board.current_player}")
    print(f"   Next board constraint: {board.next_board}")
    
    # Make seventh move
    print("\n7. Making seventh move: board 1, cell 2")
    board.make_move(1, 2)
    print(f"   Current player: {board.current_player}")
    print(f"   Next board constraint: {board.next_board}")
    
    # Check board 2 state
    print(f"\nBoard 2 state: {board.board[2]}")
    print(f"Board 2 wins: {board.small_board_wins[2]}")
    
    # Display the board
    board.display_board()

if __name__ == "__main__":
    debug_win_detection()
