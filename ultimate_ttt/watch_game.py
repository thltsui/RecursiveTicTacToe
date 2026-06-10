#!/usr/bin/env python3
"""
Watch Ultimate Tic Tac Toe games with enhanced board visualization.
"""

import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from game.board import UltimateTTTBoard
from game.config import PLAYER_1, PLAYER_2
from play.play_random import RandomAgent
import random

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def display_board_enhanced(board):
    """Display the board with enhanced visualization."""
    print("\n" + "="*60)
    print("                    ULTIMATE TIC TAC TOE")
    print("="*60)
    
    # Display the big board with small boards
    for big_row in range(3):
        for small_row in range(3):
            line = ""
            for big_col in range(3):
                big_index = big_row * 3 + big_col
                for small_col in range(3):
                    small_index = small_row * 3 + small_col
                    cell = board.board[big_index][small_index]
                    
                    if cell == 0:  # EMPTY
                        line += "·"
                    elif cell == PLAYER_1:
                        line += "X"
                    else:  # PLAYER_2
                        line += "O"
                    
                    if small_col < 2:
                        line += " "
                
                if big_col < 2:
                    line += " │ "
            print(line)
        
        if big_row < 2:
            print("─" * 60)
    
    # Display small board wins
    print("\n" + "─" * 60)
    print("SMALL BOARD WINS:")
    for i in range(3):
        row = ""
        for j in range(3):
            index = i * 3 + j
            if board.small_board_wins[index] == 0:
                row += "·"
            elif board.small_board_wins[index] == PLAYER_1:
                row += "X"
            else:
                row += "O"
            if j < 2:
                row += " "
        print(row)
    
    # Game info
    print("\n" + "─" * 60)
    current_player = "X" if board.current_player == PLAYER_1 else "O"
    print(f"Current Player: {current_player}")
    print(f"Move Count: {board.move_count}")
    
    if board.next_board != -1:
        print(f"Next Move Must Be In Board: {board.next_board}")
    else:
        print("Next Move Can Be Anywhere")
    
    if board.game_over:
        if board.winner != 0:
            winner = "X" if board.winner == PLAYER_1 else "O"
            print(f"\n🎉 GAME OVER! Player {winner} wins!")
        else:
            print(f"\n🤝 GAME OVER! It's a draw!")
    
    print("="*60)

def watch_random_game(delay=1.0, verbose=True):
    """Watch a random game with delays between moves."""
    print("🎮 Starting Ultimate Tic Tac Toe Game...")
    print("Two random AI players will battle it out!")
    print(f"⏱️  Delay between moves: {delay} seconds")
    print("\nPress Ctrl+C to stop watching...")
    
    # Create agents
    agent1 = RandomAgent(PLAYER_1)
    agent2 = RandomAgent(PLAYER_2)
    
    # Create board
    board = UltimateTTTBoard()
    move_count = 0
    
    try:
        while not board.game_over and move_count < 81:
            clear_screen()
            display_board_enhanced(board)
            
            if verbose:
                print(f"\n🤔 Player {board.current_player} is thinking...")
            
            # Get current agent
            current_agent = agent1 if board.current_player == PLAYER_1 else agent2
            
            # Get valid moves
            valid_moves = board.get_valid_moves()
            if not valid_moves:
                print("No valid moves available!")
                break
            
            # Get move from agent
            move = current_agent.get_move(board)
            if move is None:
                move = random.choice(valid_moves)
            
            big_index, small_index = move
            
            # Make the move
            success = board.make_move(big_index, small_index)
            if not success:
                print(f"Invalid move attempted: {move}")
                continue
            
            move_count += 1
            
            if verbose:
                player_symbol = "X" if board.current_player == PLAYER_2 else "O"  # Show who just moved
                print(f"✅ Player {player_symbol} moved to board {big_index}, cell {small_index}")
                print(f"📍 Next move must be in board: {board.next_board}")
            
            # Wait before next move
            time.sleep(delay)
        
        # Show final board
        clear_screen()
        display_board_enhanced(board)
        
        # Final statistics
        print(f"\n📊 GAME STATISTICS:")
        print(f"Total Moves: {board.move_count}")
        print(f"Winner: {'X' if board.winner == PLAYER_1 else 'O' if board.winner == PLAYER_2 else 'Draw'}")
        
        # Count filled cells
        filled_cells = sum(1 for big_board in board.board for cell in big_board if cell != 0)
        print(f"Filled Cells: {filled_cells}/81 ({filled_cells/81:.1%})")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Game stopped by user.")
        print("Final board state:")
        display_board_enhanced(board)

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Watch Ultimate Tic Tac Toe games')
    parser.add_argument('--delay', '-d', type=float, default=1.0,
                       help='Delay between moves in seconds (default: 1.0)')
    parser.add_argument('--fast', '-f', action='store_true',
                       help='Fast mode (0.5 second delay)')
    parser.add_argument('--slow', '-s', action='store_true',
                       help='Slow mode (2.0 second delay)')
    
    args = parser.parse_args()
    
    # Set delay based on arguments
    if args.fast:
        delay = 0.5
    elif args.slow:
        delay = 2.0
    else:
        delay = args.delay
    
    print("🎯 Ultimate Tic Tac Toe Game Watcher")
    print("=" * 50)
    
    watch_random_game(delay=delay)

if __name__ == "__main__":
    main()
