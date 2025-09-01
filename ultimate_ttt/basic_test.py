#!/usr/bin/env python3
"""
Basic functionality test for Ultimate Tic Tac Toe.
This script tests the core game logic without external dependencies.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from game.config import EMPTY, PLAYER_1, PLAYER_2
        print("✓ Config module imported successfully")
        print(f"  - EMPTY: {EMPTY}")
        print(f"  - PLAYER_1: {PLAYER_1}")
        print(f"  - PLAYER_2: {PLAYER_2}")
    except ImportError as e:
        print(f"✗ Failed to import config: {e}")
        return False
    
    try:
        from game.utils import check_win, get_available_moves
        print("✓ Utils module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import utils: {e}")
        return False
    
    try:
        from game.board import UltimateTTTBoard
        print("✓ Board module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import board: {e}")
        return False
    
    return True


def test_basic_game_logic():
    """Test basic game logic."""
    print("\nTesting basic game logic...")
    
    try:
        from game.board import UltimateTTTBoard
        from game.config import PLAYER_1, PLAYER_2
        
        # Create a new board
        board = UltimateTTTBoard()
        print("✓ Board created successfully")
        
        # Test initial state
        assert board.current_player == PLAYER_1, "Initial player should be PLAYER_1"
        assert not board.game_over, "Game should not be over initially"
        assert board.move_count == 0, "Move count should be 0 initially"
        print("✓ Initial state is correct")
        
        # Test first move
        assert board.is_valid_move(4, 6), "First move should be valid anywhere"
        assert board.make_move(4, 6), "First move should succeed"
        assert board.current_player == PLAYER_2, "Player should switch after move"
        assert board.next_board == 6, "Next board should be constrained to 6"
        print("✓ First move logic works correctly")
        
        # Test move constraint
        assert not board.is_valid_move(3, 5), "Move in wrong board should be invalid"
        assert board.is_valid_move(6, 2), "Move in correct board should be valid"
        print("✓ Move constraint logic works correctly")
        
        # Test winning a small board
        board.make_move(6, 2)  # Player 2 moves in board 6
        board.make_move(2, 0)  # Player 1 moves in board 2
        board.make_move(2, 1)  # Player 2 moves in board 2
        board.make_move(2, 2)  # Player 1 wins board 2
        
        assert board.small_board_wins[2] == PLAYER_1, "Board 2 should be won by Player 1"
        print("✓ Small board win detection works correctly")
        
        print("✓ Basic game logic tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Basic game logic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_utility_functions():
    """Test utility functions."""
    print("\nTesting utility functions...")
    
    try:
        from game.utils import check_win, get_available_moves, is_board_full
        from game.config import WINNING_COMBINATIONS
        
        # Test win detection
        board = [1, 1, 1, 0, 0, 0, 0, 0, 0]  # Row win
        winner = check_win(board, WINNING_COMBINATIONS)
        assert winner == 1, "Row win should be detected"
        
        board = [2, 0, 0, 2, 0, 0, 2, 0, 0]  # Column win
        winner = check_win(board, WINNING_COMBINATIONS)
        assert winner == 2, "Column win should be detected"
        
        board = [1, 0, 0, 0, 1, 0, 0, 0, 1]  # Diagonal win
        winner = check_win(board, WINNING_COMBINATIONS)
        assert winner == 1, "Diagonal win should be detected"
        
        print("✓ Win detection works correctly")
        
        # Test available moves
        board = [1, 0, 2, 0, 1, 0, 2, 0, 1]
        available = get_available_moves(board)
        assert available == [1, 3, 5, 7], "Available moves should be [1, 3, 5, 7]"
        print("✓ Available moves detection works correctly")
        
        # Test board full detection
        board = [1, 2, 1, 2, 1, 2, 2, 1, 2]
        assert is_board_full(board), "Full board should be detected"
        
        board = [1, 0, 2, 0, 1, 0, 2, 0, 1]
        assert not is_board_full(board), "Non-full board should be detected"
        print("✓ Board full detection works correctly")
        
        print("✓ Utility function tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Utility function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("Ultimate Tic Tac Toe - Basic Functionality Test")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Cannot continue.")
        return False
    
    # Test utility functions
    if not test_utility_functions():
        print("\n❌ Utility function tests failed.")
        return False
    
    # Test basic game logic
    if not test_basic_game_logic():
        print("\n❌ Basic game logic tests failed.")
        return False
    
    print("\n🎉 All basic tests passed! The game should work correctly.")
    print("\nTo play the game, run:")
    print("  python3 main.py")
    print("\nTo run full tests, run:")
    print("  python3 main.py test")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
