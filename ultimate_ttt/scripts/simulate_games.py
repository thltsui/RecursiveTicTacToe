"""
Script to generate multiple game histories for training purposes.
"""

import os
import sys
import argparse
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from play.play_random import simulate_multiple_games, save_simulation_results
from game.board import UltimateTTTBoard
from game.utils import board_to_state_tensor


def generate_training_data(num_games, output_dir=None, save_format='both'):
    """
    Generate training data from multiple games.
    
    Args:
        num_games: Number of games to simulate
        output_dir: Output directory for saved files
        save_format: Format to save ('json', 'npy', 'both')
        
    Returns:
        Dictionary with training data
    """
    print(f"Generating {num_games} games for training data...")
    
    # Simulate games
    stats = simulate_multiple_games(num_games, verbose=True)
    
    # Prepare training data
    training_data = {
        'games': [],
        'total_games': num_games,
        'timestamp': datetime.now().isoformat()
    }
    
    for i, result in enumerate(stats['results']):
        # Create board and replay moves to get state sequence
        board = UltimateTTTBoard()
        state_sequence = []
        
        for big_index, small_index, player in result['move_history']:
            # Get current state tensor
            state_tensor = board.get_state_tensor()
            state_sequence.append(state_tensor.copy())
            
            # Make the move
            board.make_move(big_index, small_index)
        
        # Add final state
        final_state = board.get_state_tensor()
        state_sequence.append(final_state)
        
        game_data = {
            'game_id': i,
            'move_history': result['move_history'],
            'state_sequence': state_sequence,
            'winner': result['winner'],
            'move_count': result['move_count'],
            'final_board': result['final_board'],
            'small_board_wins': result['small_board_wins']
        }
        
        training_data['games'].append(game_data)
    
    # Save training data
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'data', 'saved_games'
        )
    
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if save_format in ['json', 'both']:
        # Save as JSON
        import json
        
        # Convert numpy arrays to lists for JSON
        json_data = {}
        for key, value in training_data.items():
            if key == 'games':
                json_games = []
                for game in value:
                    json_game = {}
                    for k, v in game.items():
                        if isinstance(v, np.ndarray):
                            json_game[k] = v.tolist()
                        else:
                            json_game[k] = v
                    json_games.append(json_game)
                json_data[key] = json_games
            else:
                json_data[key] = value
        
        json_filename = f"training_data_{timestamp}.json"
        json_filepath = os.path.join(output_dir, json_filename)
        
        with open(json_filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Training data saved as JSON: {json_filepath}")
    
    if save_format in ['npy', 'both']:
        # Save as numpy arrays
        npy_filename = f"training_data_{timestamp}.npy"
        npy_filepath = os.path.join(output_dir, npy_filename)
        
        # Extract state sequences and move histories
        state_sequences = []
        move_histories = []
        winners = []
        
        for game in training_data['games']:
            state_sequences.append(np.array(game['state_sequence']))
            move_histories.append(np.array(game['move_history']))
            winners.append(game['winner'])
        
        np.savez_compressed(
            npy_filepath,
            state_sequences=state_sequences,
            move_histories=move_histories,
            winners=winners,
            metadata=training_data
        )
        
        print(f"Training data saved as NPY: {npy_filepath}")
    
    # Save summary statistics
    summary_filename = f"training_summary_{timestamp}.txt"
    summary_filepath = os.path.join(output_dir, summary_filename)
    
    with open(summary_filepath, 'w') as f:
        f.write(f"Training Data Summary\n")
        f.write(f"=====================\n\n")
        f.write(f"Generated: {training_data['timestamp']}\n")
        f.write(f"Total Games: {training_data['total_games']}\n\n")
        f.write(f"Game Statistics:\n")
        f.write(f"Player 1 Wins: {stats['wins_player1']} ({stats['win_rate_player1']:.2%})\n")
        f.write(f"Player 2 Wins: {stats['wins_player2']} ({stats['win_rate_player2']:.2%})\n")
        f.write(f"Draws: {stats['draws']} ({stats['draw_rate']:.2%})\n")
        f.write(f"Average Moves: {stats['avg_moves']:.1f}\n\n")
        f.write(f"Data Format:\n")
        f.write(f"- State sequences: {len(training_data['games'])} games\n")
        f.write(f"- Each game has {len(training_data['games'][0]['state_sequence'])} states\n")
        f.write(f"- State tensor shape: {training_data['games'][0]['state_sequence'][0].shape}\n")
    
    print(f"Summary saved: {summary_filepath}")
    
    return training_data


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Generate Ultimate Tic Tac Toe training data')
    parser.add_argument('--num-games', '-n', type=int, default=1000,
                       help='Number of games to simulate (default: 1000)')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                       help='Output directory for saved files')
    parser.add_argument('--format', '-f', choices=['json', 'npy', 'both'], default='both',
                       help='Output format (default: both)')
    
    args = parser.parse_args()
    
    print("Ultimate Tic Tac Toe Training Data Generator")
    print("=" * 50)
    
    try:
        training_data = generate_training_data(
            num_games=args.num_games,
            output_dir=args.output_dir,
            save_format=args.format
        )
        
        print(f"\nSuccessfully generated training data for {args.num_games} games!")
        print("Files saved in:", args.output_dir or "data/saved_games/")
        
    except KeyboardInterrupt:
        print("\nGeneration cancelled.")
    except Exception as e:
        print(f"Error generating training data: {e}")


if __name__ == "__main__":
    main()
