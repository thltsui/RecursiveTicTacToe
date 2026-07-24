import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from importlib import import_module
network_mod = import_module('02_network.network')
self_play_mod = import_module('04_training.self_play')

def main():
    device = 'cpu'
    
    net = network_mod.UltimateTTTNetwork(channels=192, num_blocks=10)
    net.to(device)
    
    import glob
    checkpoints = sorted(glob.glob("checkpoints/large_v3_pure_self_play/checkpoint_iter*.pt"))
    if not checkpoints:
        print("No checkpoints found.")
        return
        
    checkpoint_path = checkpoints[-1]
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        net.load_state_dict(checkpoint['network_state_dict'])
        print(f"Loaded {checkpoint_path}")
    else:
        print(f"Checkpoint {checkpoint_path} not found.")
        return
        
    net.eval()
    
    num_games = 20
    wins = 0
    losses = 0
    draws = 0
    
    print(f"Evaluating {checkpoint_path} against random player for {num_games} games (num_simulations=200)...")
    
    with torch.no_grad():
        for i in range(num_games):
            net_p = 1 if i < (num_games // 2) else -1
            record = self_play_mod.play_vs_random_game(net, num_simulations=800, device=device, network_player=net_p)
            
            # Record winner from the perspective of the network
            if record.winner == net_p:
                wins += 1
            elif record.winner == -net_p:
                losses += 1
            else:
                draws += 1
                
            print(f"Game {i+1}/{num_games}: Network as {'P1' if net_p==1 else 'P2'}, Winner: {record.winner}, Length: {record.game_length}")
            
    total = wins + losses + draws
    win_rate = wins / total
    print(f"\nResults:")
    print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")
    print(f"Win Rate: {win_rate:.3f}")

if __name__ == "__main__":
    main()
