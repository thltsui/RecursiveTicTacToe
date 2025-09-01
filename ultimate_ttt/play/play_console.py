"""
Interactive console interface for Ultimate Tic Tac Toe.
"""

import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.board import UltimateTTTBoard
from game.config import PLAYER_1, PLAYER_2


class RandomPlayer:
    """Random player that makes random valid moves."""
    
    def __init__(self, player_id):
        self.player_id = player_id
    
    def get_move(self, board):
        """Get a random valid move."""
        valid_moves = board.get_valid_moves()
        if valid_moves:
            return random.choice(valid_moves)
        return None


class HumanPlayer:
    """Human player that gets moves from console input."""
    
    def __init__(self, player_id):
        self.player_id = player_id
        self.symbol = 'X' if player_id == PLAYER_1 else 'O'
    
    def get_move(self, board):
        """Get move from human input."""
        while True:
            try:
                print(f"\nPlayer {self.symbol}'s turn")
                print("Enter move as 'big_index small_index' (e.g., '4 6')")
                print("Or enter 'q' to quit")
                
                user_input = input("Move: ").strip()
                
                if user_input.lower() == 'q':
                    return None
                
                parts = user_input.split()
                if len(parts) != 2:
                    print("Invalid input. Please enter two numbers separated by space.")
                    continue
                
                big_index = int(parts[0])
                small_index = int(parts[1])
                
                if not (0 <= big_index <= 8 and 0 <= small_index <= 8):
                    print("Indices must be between 0 and 8.")
                    continue
                
                if not board.is_valid_move(big_index, small_index):
                    print("Invalid move. Please try again.")
                    continue
                
                return (big_index, small_index)
                
            except ValueError:
                print("Invalid input. Please enter two numbers separated by space.")
            except KeyboardInterrupt:
                return None


def play_game(player1_type="human", player2_type="human"):
    """
    Play a game of Ultimate Tic Tac Toe.
    
    Args:
        player1_type: "human" or "random"
        player2_type: "human" or "random"
    """
    board = UltimateTTTBoard()
    
    # Create players
    if player1_type == "human":
        player1 = HumanPlayer(PLAYER_1)
    else:
        player1 = RandomPlayer(PLAYER_1)
    
    if player2_type == "human":
        player2 = HumanPlayer(PLAYER_2)
    else:
        player2 = RandomPlayer(PLAYER_2)
    
    print("Ultimate Tic Tac Toe")
    print("=" * 30)
    print("Board indices:")
    print("0 | 1 | 2")
    print("---------")
    print("3 | 4 | 5")
    print("---------")
    print("6 | 7 | 8")
    print("\nSmall cell indices within each board:")
    print("0 | 1 | 2")
    print("---------")
    print("3 | 4 | 5")
    print("---------")
    print("6 | 7 | 8")
    print("\nEnter moves as 'big_index small_index'")
    
    while not board.game_over:
        board.display_board()
        
        current_player = player1 if board.current_player == PLAYER_1 else player2
        
        move = current_player.get_move(board)
        if move is None:
            print("Game cancelled.")
            return
        
        big_index, small_index = move
        success = board.make_move(big_index, small_index)
        
        if not success:
            print("Invalid move. Please try again.")
            continue
    
    # Game over
    board.display_board()
    
    # Save game if requested
    save_game = input("\nSave this game? (y/n): ").strip().lower()
    if save_game == 'y':
        save_game_to_file(board)


def save_game_to_file(board):
    """Save the game to a file."""
    try:
        import json
        from datetime import datetime
        
        game_data = {
            'timestamp': datetime.now().isoformat(),
            'winner': board.winner,
            'move_count': board.move_count,
            'move_history': board.move_history,
            'final_board': board.board.tolist(),
            'small_board_wins': board.small_board_wins.tolist()
        }
        
        filename = f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'saved_games', filename)
        
        with open(filepath, 'w') as f:
            json.dump(game_data, f, indent=2)
        
        print(f"Game saved to {filepath}")
        
    except Exception as e:
        print(f"Failed to save game: {e}")


def main():
    """Main function to start the game."""
    print("Ultimate Tic Tac Toe")
    print("=" * 30)
    print("Choose game mode:")
    print("1. Human vs Human")
    print("2. Human vs Random")
    print("3. Random vs Human")
    print("4. Random vs Random")
    
    while True:
        try:
            choice = input("\nEnter choice (1-4): ").strip()
            if choice == '1':
                play_game("human", "human")
                break
            elif choice == '2':
                play_game("human", "random")
                break
            elif choice == '3':
                play_game("random", "human")
                break
            elif choice == '4':
                play_game("random", "random")
                break
            else:
                print("Invalid choice. Please enter 1, 2, 3, or 4.")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
    
    play_again = input("\nPlay again? (y/n): ").strip().lower()
    if play_again == 'y':
        main()


if __name__ == "__main__":
    main()
