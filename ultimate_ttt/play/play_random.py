"""
Simulate Ultimate Tic Tac Toe games using random agents.
"""

import random
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.board import UltimateTTTBoard
from game.config import PLAYER_1, PLAYER_2


class RandomAgent:
    """Random agent that makes random valid moves."""
    
    def __init__(self, player_id):
        self.player_id = player_id
    
    def get_move(self, board):
        """Get a random valid move."""
        valid_moves = board.get_valid_moves()
        if valid_moves:
            return random.choice(valid_moves)
        return None


def simulate_game(agent1=None, agent2=None, verbose=False):
    """
    Simulate a single game.
    
    Args:
        agent1: Agent for player 1 (default: RandomAgent)
        agent2: Agent for player 2 (default: RandomAgent)
        verbose: Whether to print game progress
        
    Returns:
        Dictionary with game results
    """
    if agent1 is None:
        agent1 = RandomAgent(PLAYER_1)
    if agent2 is None:
        agent2 = RandomAgent(PLAYER_2)
    
    board = UltimateTTTBoard()
    
    if verbose:
        print("Starting new game...")
    
    while not board.game_over:
        current_agent = agent1 if board.current_player == PLAYER_1 else agent2
        
        move = current_agent.get_move(board)
        if move is None:
            # No valid moves available
            break
        
        big_index, small_index = move
        success = board.make_move(big_index, small_index)
        
        if not success:
            if verbose:
                print(f"Invalid move attempted: {move}")
            continue
        
        if verbose:
            print(f"Player {board.current_player} moved to board {big_index}, cell {small_index}")
    
    result = {
        'winner': board.winner,
        'move_count': board.move_count,
        'move_history': board.move_history.copy(),
        'final_board': board.board.copy(),
        'small_board_wins': board.small_board_wins.copy(),
        'game_over': board.game_over
    }
    
    if verbose:
        if board.winner != 0:
            print(f"Game over! Player {board.winner} wins in {board.move_count} moves.")
        else:
            print(f"Game over! Draw after {board.move_count} moves.")
    
    return result


def simulate_multiple_games(num_games, agent1=None, agent2=None, verbose=False):
    """
    Simulate multiple games and collect statistics.
    
    Args:
        num_games: Number of games to simulate
        agent1: Agent for player 1
        agent2: Agent for player 2
        verbose: Whether to print progress
        
    Returns:
        Dictionary with simulation results
    """
    results = []
    wins_player1 = 0
    wins_player2 = 0
    draws = 0
    
    for i in range(num_games):
        if verbose and (i + 1) % 100 == 0:
            print(f"Completed {i + 1}/{num_games} games...")
        
        result = simulate_game(agent1, agent2, verbose=False)
        results.append(result)
        
        if result['winner'] == PLAYER_1:
            wins_player1 += 1
        elif result['winner'] == PLAYER_2:
            wins_player2 += 1
        else:
            draws += 1
    
    # Calculate statistics
    total_games = len(results)
    avg_moves = np.mean([r['move_count'] for r in results])
    
    stats = {
        'total_games': total_games,
        'wins_player1': wins_player1,
        'wins_player2': wins_player2,
        'draws': draws,
        'win_rate_player1': wins_player1 / total_games,
        'win_rate_player2': wins_player2 / total_games,
        'draw_rate': draws / total_games,
        'avg_moves': avg_moves,
        'results': results
    }
    
    if verbose:
        print(f"\nSimulation Results ({num_games} games):")
        print(f"Player 1 wins: {wins_player1} ({stats['win_rate_player1']:.2%})")
        print(f"Player 2 wins: {wins_player2} ({stats['win_rate_player2']:.2%})")
        print(f"Draws: {draws} ({stats['draw_rate']:.2%})")
        print(f"Average moves per game: {avg_moves:.1f}")
    
    return stats


def save_simulation_results(stats, filename=None):
    """
    Save simulation results to a file.
    
    Args:
        stats: Simulation statistics dictionary
        filename: Optional filename (default: auto-generated)
    """
    try:
        import json
        from datetime import datetime
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"simulation_{timestamp}.json"
        
        filepath = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'data', 'saved_games', 
            filename
        )
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_stats = {}
        for key, value in stats.items():
            if key == 'results':
                # Handle results list specially
                serializable_results = []
                for result in value:
                    serializable_result = {}
                    for k, v in result.items():
                        if isinstance(v, np.ndarray):
                            serializable_result[k] = v.tolist()
                        else:
                            serializable_result[k] = v
                    serializable_results.append(serializable_result)
                serializable_stats[key] = serializable_results
            else:
                serializable_stats[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(serializable_stats, f, indent=2)
        
        print(f"Simulation results saved to {filepath}")
        return filepath
        
    except Exception as e:
        print(f"Failed to save simulation results: {e}")
        return None


def main():
    """Main function to run simulations."""
    print("Ultimate Tic Tac Toe Random Simulation")
    print("=" * 40)
    
    try:
        num_games = int(input("Enter number of games to simulate: "))
        if num_games <= 0:
            print("Number of games must be positive.")
            return
        
        verbose = input("Show progress? (y/n): ").strip().lower() == 'y'
        
        print(f"\nSimulating {num_games} games...")
        stats = simulate_multiple_games(num_games, verbose=verbose)
        
        # Ask if user wants to save results
        save = input("\nSave simulation results? (y/n): ").strip().lower()
        if save == 'y':
            save_simulation_results(stats)
        
        # Ask if user wants to run another simulation
        again = input("\nRun another simulation? (y/n): ").strip().lower()
        if again == 'y':
            main()
            
    except ValueError:
        print("Invalid input. Please enter a valid number.")
    except KeyboardInterrupt:
        print("\nSimulation cancelled.")


if __name__ == "__main__":
    main()
