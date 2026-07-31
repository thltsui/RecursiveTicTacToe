import os
import sys
import glob
from collections import defaultdict

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from importlib import import_module
network_mod = import_module('02_network.network')
board_mod = import_module('01_game.board')
search_mod = import_module('03_mcts.search')
rules_mod = import_module('01_game.rules')
trainer_mod = import_module('04_training.trainer')

def main():
    # Find latest checkpoint
    checkpoint_dir = 'checkpoints/large_v5_exp_temp/'
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, 'checkpoint_iter*.pt')))
    if not checkpoints:
        print("No checkpoints found!")
        return
    latest_cp = checkpoints[-1]
    print(f"Loading latest checkpoint: {latest_cp}")

    # Load network
    device = 'cpu'
    net = network_mod.UltimateTTTNetwork(channels=192, num_blocks=10).to(device)
    meta = trainer_mod.load_checkpoint(latest_cp, net)
    net.eval()
    print(f"Loaded model from iteration {meta['iteration']}")
    
    # Let's inspect the policy prior at the root board first
    state = board_mod.create_initial_state()
    output = net.predict(state, device=device)
    legal_mask = rules_mod.get_legal_move_mask(state)
    from importlib import import_module
    policy_head_mod = import_module('02_network.policy_head')
    probs = policy_head_mod.apply_legal_mask(output.policy_logits, legal_mask)
    
    print("\nTop 5 moves proposed by the neural network from the empty board (Prior Policy):")
    probs_list = [(i, probs[i].item()) for i in range(81) if probs[i].item() > 0.001]
    probs_list.sort(key=lambda x: x[1], reverse=True)
    for move, p in probs_list[:5]:
        print(f"  Move {move}: {p*100:.2f}%")

    print(f"\nValue evaluation of empty board (from Player 1's perspective): {output.win_value.item():.3f}")

    num_games = 100
    num_sims = 100
    print(f"\nPlaying {num_games} self-play games (temperature active, dirichlet off) to measure diversity...")

    unique_openings = defaultdict(int)
    unique_first_moves = defaultdict(int)

    for i in tqdm(range(num_games), desc="Playing games"):
        state = board_mod.create_initial_state()
        history = []
        move_count = 0
        
        while not state.is_terminal and move_count < 10:
            root = search_mod.run_mcts(
                state, net,
                num_simulations=num_sims,
                dirichlet_epsilon=0.0,
                device=device
            )
            
            # Use the exact training temperature schedule
            temp = 2.0 * (0.94 ** state.move_count)
            if temp < 0.15:
                temp = 0.0
                
            move = search_mod.select_move(root, temperature=temp)
            history.append(move)
            state = rules_mod.apply_move(state, move)
            move_count += 1
            
        unique_first_moves[history[0]] += 1
        opening_key = tuple(history[:5])  # First 5 plies
        unique_openings[opening_key] += 1

    print("\n--- Diversity Results ---")
    print(f"Total games played: {num_games}")
    print(f"Number of unique first moves chosen: {len(unique_first_moves)}")
    print(f"Number of unique 5-ply opening sequences chosen: {len(unique_openings)}")
    
    print("\nMost frequent first moves:")
    for m, c in sorted(unique_first_moves.items(), key=lambda x: -x[1])[:5]:
        print(f"  Move {m}: played {c} times")

    print("\nMost frequent 5-ply opening lines:")
    for seq, c in sorted(unique_openings.items(), key=lambda x: -x[1])[:5]:
        print(f"  Sequence {seq}: played {c} times")
        
if __name__ == '__main__':
    main()
