#!/usr/bin/env python3
"""Evaluate v3 pure self-play model vs v2 fixed model."""

import sys, os

sys.path.insert(0, os.path.dirname(__file__))

import torch
from importlib import import_module

arena_mod = import_module('06_evaluation.arena')
network_mod = import_module('02_network.network')

device = 'cpu'
num_sims = 200

def load_latest_checkpoint(ckpt_dir, channels=192, num_blocks=10):
    ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pt') and 'checkpoint' in f])
    if not ckpts:
        raise ValueError(f"No checkpoints found in {ckpt_dir}")
    latest = ckpts[-1]
    ckpt_path = os.path.join(ckpt_dir, latest)
    print(f"Loading {ckpt_path}...")
    cp = torch.load(ckpt_path, weights_only=False, map_location=device)
    net = network_mod.UltimateTTTNetwork(channels=channels, num_blocks=num_blocks)
    net.load_state_dict(cp['network_state_dict'])
    net.eval()
    print(f"  Iteration: {cp.get('iteration', '?')}")
    return net

def eval_head_to_head(net_a, net_b, name_a="v3", name_b="v2", num_games=20):
    wins_a, wins_b, draws = 0, 0, 0
    
    for i in range(num_games):
        # Alternate who goes first
        if i % 2 == 0:
            print(f"  Game {i+1}/{num_games}: {name_a} vs {name_b} (going second)")
            r = arena_mod.play_single_game(net_a, net_b, num_sims, device)
            if r == 1:
                wins_a += 1
                res = f"{name_a} wins"
            elif r == -1:
                wins_b += 1
                res = f"{name_b} wins"
            else:
                draws += 1
                res = "Draw"
        else:
            print(f"  Game {i+1}/{num_games}: {name_b} vs {name_a} (going second)")
            r = arena_mod.play_single_game(net_b, net_a, num_sims, device)
            if r == 1:
                wins_b += 1
                res = f"{name_b} wins"
            elif r == -1:
                wins_a += 1
                res = f"{name_a} wins"
            else:
                draws += 1
                res = "Draw"
                
        print(f"    Result: {res} | Current Score: {name_a} {wins_a} - {name_b} {wins_b} - Draws {draws}", flush=True)

    print("\n" + "=" * 60)
    print(f"  FINAL SCORE:")
    print(f"  {name_a}: {wins_a} wins")
    print(f"  {name_b}: {wins_b} wins")
    print(f"  Draws: {draws}")
    total_games = wins_a + wins_b + draws
    if wins_a + wins_b > 0:
        win_rate_a = wins_a / total_games
        print(f"  {name_a} win rate (including draws): {win_rate_a:.3f}")
    print("=" * 60)

def main():
    v3_dir = 'checkpoints/large_v3_pure_self_play'
    v2_dir = 'checkpoints/large_v2_fixed'
    
    print(f"Loading models for head-to-head evaluation...")
    print("---------------------------------------------")
    try:
        net_v3 = load_latest_checkpoint(v3_dir)
        net_v2 = load_latest_checkpoint(v2_dir)
    except Exception as e:
        print(f"Error loading models: {e}")
        return
        
    print("\n" + "=" * 60)
    print("  EVALUATION: v3 (Pure Self Play) vs v2 (Mixed)")
    print("=" * 60)
    eval_head_to_head(net_v3, net_v2, name_a="v3_pure", name_b="v2_mixed", num_games=20)

if __name__ == '__main__':
    main()
