import os
import sys
import glob

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from importlib import import_module
network_mod = import_module('02_network.network')
board_mod = import_module('01_game.board')
search_mod = import_module('03_mcts.search')
rules_mod = import_module('01_game.rules')
trainer_mod = import_module('04_training.trainer')
policy_head_mod = import_module('02_network.policy_head')

def main():
    checkpoint_dir = 'checkpoints/large_v5_exp_temp/'
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, 'checkpoint_iter*.pt')))
    latest_cp = checkpoints[-1]
    
    device = 'cpu'
    net = network_mod.UltimateTTTNetwork(channels=192, num_blocks=10).to(device)
    meta = trainer_mod.load_checkpoint(latest_cp, net)
    net.eval()
    
    print(f"Loaded model from iteration {meta['iteration']}")
    print("Generating 3 random mid-game states (Move 35) to test network diversity...\n")
    
    for game_idx in range(3):
        state = board_mod.create_initial_state()
        
        # Play 35 moves to reach mid-game
        for _ in range(35):
            if state.is_terminal:
                break
            # Generate realistic play using MCTS with temp=1.0
            root = search_mod.run_mcts(state, net, num_simulations=50, dirichlet_epsilon=0.0, device=device)
            move = search_mod.select_move(root, temperature=1.0)
            state = rules_mod.apply_move(state, move)
            
        if state.is_terminal:
            print(f"Game {game_idx+1}: Ended early at move {state.move_count}. Skipping.")
            continue
            
        print(f"--- Game {game_idx+1} at Move {state.move_count} ---")
        
        # Get raw network policy
        output = net.predict(state, device=device)
        legal_mask = rules_mod.get_legal_move_mask(state)
        probs = policy_head_mod.apply_legal_mask(output.policy_logits, legal_mask)
        
        probs_list = [(i, probs[i].item()) for i in range(81) if probs[i].item() > 0.001]
        probs_list.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Top 5 moves proposed by network:")
        for move, p in probs_list[:5]:
            print(f"  Move {move}: {p*100:.2f}%")
            
        print(f"Value evaluation: {output.win_value.item():.3f}")
        
        # Calculate how many moves get at least 2% probability
        viable_moves = sum(1 for p in probs_list if p[1] > 0.02)
        legal_count = sum(legal_mask)
        # Run MCTS to see actual Q-values
        root = search_mod.run_mcts(state, net, num_simulations=400, dirichlet_epsilon=0.0, device=device)
        
        print("\nMCTS Child Nodes (Visits and Q-values):")
        q_values = []
        for move_idx, child in root.children.items():
            if root.N.get(move_idx, 0) > 0:
                q = root.Q[move_idx]
                n = root.N[move_idx]
                q_values.append((move_idx, q, n))
                print(f"  Move {move_idx}: Q-value = {q:+.3f} (Visits: {n})")
                
        # Check if all Q-values are exactly -1 or +1
        if all(q <= -0.95 for _, q, _ in q_values):
            print("  -> ALERT: Q-values have collapsed to -1 (Current player believes all paths lead to a loss).")
        elif all(q >= 0.95 for _, q, _ in q_values):
            print("  -> ALERT: Q-values have collapsed to +1 (Current player believes all paths lead to a win).")
        else:
            print("  -> Q-values are distributed and not collapsed.")
        print("\n" + "-"*40 + "\n")

if __name__ == '__main__':
    main()
