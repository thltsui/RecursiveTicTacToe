import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from importlib import import_module
network_mod = import_module('02_network.network')
arena_mod = import_module('06_evaluation.arena')

def main():
    device = 'cpu'
    
    net_old = network_mod.UltimateTTTNetwork(channels=192, num_blocks=10)
    net_old.to(device)
    ckpt_old_path = "checkpoints/pretrain_value.pt"
    checkpoint_old = torch.load(ckpt_old_path, map_location=device, weights_only=True)
    net_old.load_state_dict(checkpoint_old['network_state_dict'])
    net_old.eval()
    print(f"Loaded old net: {ckpt_old_path}")

    net_new = network_mod.UltimateTTTNetwork(channels=192, num_blocks=10)
    net_new.to(device)
    ckpt_new_path = "checkpoints/large_v3_pure_self_play/checkpoint_iter00005_elo0.pt"
    checkpoint_new = torch.load(ckpt_new_path, map_location=device, weights_only=True)
    net_new.load_state_dict(checkpoint_new['network_state_dict'])
    net_new.eval()
    print(f"Loaded new net: {ckpt_new_path}")
    
    num_games = 20
    num_simulations = 20
    
    print(f"\nRunning Arena: New vs Old ({num_games} games, {num_simulations} sims/move)")
    with torch.no_grad():
        result = arena_mod.run_arena(
            network_new=net_new,
            network_old=net_old,
            num_games=num_games,
            num_simulations=num_simulations,
            device=device
        )
        
    print(f"\nResults (New vs Old):")
    print(f"New Wins: {result.wins}, Old Wins: {result.losses}, Draws: {result.draws}")
    print(f"New Win Rate: {result.win_rate:.3f}")

if __name__ == "__main__":
    main()
