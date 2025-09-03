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
        
        # Validate the game result
        if not result.get('is_complete', False):
            print(f"Warning: Game {i} may not be complete (completion rate: {result.get('completion_rate', 0):.2%})")
        
        for big_index, small_index, player in result['move_history']:
            # Get current state tensor
            state_tensor = board.get_state_tensor()
            state_sequence.append(state_tensor.copy())
            
            # Make the move
            board.make_move(big_index, small_index)
        
        # Add final state
        final_state = board.get_state_tensor()
        state_sequence.append(final_state)
        
        # Verify the replay matches the original result
        replay_winner = board.winner
        replay_moves = board.move_count
        if replay_winner != result['winner'] or replay_moves != result['move_count']:
            print(f"Warning: Game {i} replay mismatch - Original: winner={result['winner']}, moves={result['move_count']}, Replay: winner={replay_winner}, moves={replay_moves}")
        
        game_data = {
            'game_id': i,
            'move_history': result['move_history'],
            'winner': result['winner'],
            'move_count': result['move_count'],
            'final_board': result['final_board'].tolist() if hasattr(result['final_board'], 'tolist') else result['final_board'],
            'small_board_wins': result['small_board_wins'].tolist() if hasattr(result['small_board_wins'], 'tolist') else result['small_board_wins'],
            'completion_rate': result.get('completion_rate', 0),
            'is_complete': result.get('is_complete', False),
            'replay_verified': (replay_winner == result['winner'] and replay_moves == result['move_count'])
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
            print(f"Processing key: {key}, type: {type(value)}")
            if key == 'games':
                json_games = []
                for i, game in enumerate(value):
                    print(f"Processing game {i}")
                    json_game = {}
                    for k, v in game.items():
                        try:
                            if isinstance(v, np.ndarray):
                                print(f"Converting numpy array for key '{k}', shape: {v.shape}, dtype: {v.dtype}")
                                json_game[k] = v.tolist()
                            elif isinstance(v, np.integer):
                                json_game[k] = int(v)
                            elif isinstance(v, np.floating):
                                json_game[k] = float(v)
                            elif isinstance(v, np.bool_):
                                json_game[k] = bool(v)
                            else:
                                json_game[k] = v
                        except Exception as e:
                            print(f"Error serializing key '{k}' with value {v} (type: {type(v)}): {e}")
                            # Try to convert to string as fallback
                            json_game[k] = str(v)
                    json_games.append(json_game)
                json_data[key] = json_games
            else:
                try:
                    if isinstance(value, np.ndarray):
                        json_data[key] = value.tolist()
                    elif isinstance(value, np.integer):
                        json_data[key] = int(value)
                    elif isinstance(value, np.floating):
                        json_data[key] = float(value)
                    elif isinstance(value, np.bool_):
                        json_data[key] = bool(value)
                    else:
                        json_data[key] = value
                except Exception as e:
                    print(f"Error serializing metadata key '{key}' with value {value} (type: {type(value)}): {e}")
                    json_data[key] = str(value)
        
        json_filename = f"training_data_{timestamp}.json"
        json_filepath = os.path.join(output_dir, json_filename)
        
        with open(json_filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Training data saved as JSON: {json_filepath}")
    
    if save_format in ['npy', 'both']:
        # Save as numpy arrays
        npy_filename = f"training_data_{timestamp}.npy"
        npy_filepath = os.path.join(output_dir, npy_filename)
        
        # Extract move histories and other data
        move_histories = []
        winners = []
        final_boards = []
        small_board_wins = []
        
        # Find maximum move count for padding
        max_moves = max(len(game['move_history']) for game in training_data['games'])
        
        for game in training_data['games']:
            # Pad move history to max length with -1 (invalid move)
            move_history = game['move_history']
            if len(move_history) < max_moves:
                padded_history = move_history + [(-1, -1, -1)] * (max_moves - len(move_history))
            else:
                padded_history = move_history
            
            move_histories.append(np.array(padded_history))
            winners.append(game['winner'])
            final_boards.append(np.array(game['final_board']))
            small_board_wins.append(np.array(game['small_board_wins']))
        
        np.savez_compressed(
            npy_filepath,
            move_histories=np.array(move_histories),
            winners=np.array(winners),
            final_boards=np.array(final_boards),
            small_board_wins=np.array(small_board_wins),
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
        f.write(f"Completion Statistics:\n")
        f.write(f"Complete Games: {stats.get('complete_games', 'N/A')}\n")
        f.write(f"Completion Rate: {stats.get('completion_rate', 'N/A'):.2%}\n")
        f.write(f"Average Completion Rate: {stats.get('avg_completion_rate', 'N/A'):.2%}\n\n")
        f.write(f"Data Format:\n")
        f.write(f"- Games: {len(training_data['games'])} games\n")
        f.write(f"- Move histories: {len(training_data['games'])} games\n")
        f.write(f"- Final boards: 9x9 arrays\n")
        f.write(f"- Small board wins: 9-element arrays\n")
        f.write(f"- Replay verified: {sum(1 for g in training_data['games'] if g.get('replay_verified', False))}/{len(training_data['games'])} games\n")
    
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
