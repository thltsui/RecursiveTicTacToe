#!/usr/bin/env python3
"""
Demo script for Ultimate Tic Tac Toe.
Shows a quick game between two random players.
"""

import sys
import os
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_demo():
    """Run a demo game."""
    print("Ultimate Tic Tac Toe Demo")
    print("=" * 30)
    print("This will show a quick game between two random players.")
    print("Press Enter to continue...")
    input()
    
    try:
        from game.board import UltimateTTTBoard
        from game.config import PLAYER_1, PLAYER_2
        
        # Create board
        board = UltimateTTTBoard()
        
        print("\nStarting new game...")
        print("Player 1: X, Player 2: O")
        print("=" * 50)
        
        move_count = 0
        while not board.game_over and move_count < 20:  # Limit to 20 moves for demo
            # Display current board
            board.display_board()
            
            # Get valid moves
            valid_moves = board.get_valid_moves()
            if not valid_moves:
                break
            
            # Make a random move
            import random
            big_index, small_index = random.choice(valid_moves)
            
            print(f"\nPlayer {board.current_player} moves to board {big_index}, cell {small_index}")
            
            # Make the move
            board.make_move(big_index, small_index)
            move_count += 1
            
            # Small delay to make it readable
            time.sleep(1)
            
            print("\n" + "="*50)
        
        # Show final board
        board.display_board()
        
        print(f"\nDemo completed after {move_count} moves!")
        if board.winner != 0:
            winner_symbol = 'X' if board.winner == PLAYER_1 else 'O'
            print(f"Player {winner_symbol} won!")
        else:
            print("Game ended without a winner.")
            
    except ImportError as e:
        print(f"Failed to import required modules: {e}")
        print("Make sure all dependencies are installed.")
        return False
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """Main function."""
    try:
        success = run_demo()
        if success:
            print("\nDemo completed successfully!")
        else:
            print("\nDemo failed.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
