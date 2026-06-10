#!/usr/bin/env .venv/bin/python3
"""Arena evaluation — Compare checkpoints head-to-head.

Usage:
    python arena_eval.py                     # Compare latest vs best_model
    python arena_eval.py --num-games 50      # Fewer games for faster eval
    python arena_eval.py --num-sims 200      # Fewer sims for speed
    python arena_eval.py --checkpoint1 path  # Custom first checkpoint
    python arena_eval.py --checkpoint2 path  # Custom second checkpoint
"""

import sys
import os
import argparse
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from importlib import import_module

network_mod = import_module('02_network.network')
arena_mod = import_module('06_evaluation.arena')


def infer_network_config(state_dict):
    """Infer channels and num_blocks from a checkpoint state dict."""
    first_key = [k for k in state_dict if 'conv.weight' in k and 'input' in k][0]
    channels = state_dict[first_key].shape[0]
    num_blocks = sum(1 for k in state_dict if '.conv1.weight' in k)
    return channels, num_blocks


def load_network_from_checkpoint(path, device='cpu'):
    """Load a network from a checkpoint file."""
    checkpoint = torch.load(path, weights_only=False, map_location=device)
    state_dict = checkpoint['network_state_dict']
    channels, num_blocks = infer_network_config(state_dict)
    net = network_mod.UltimateTTTNetwork(channels=channels, num_blocks=num_blocks)
    net.load_state_dict(state_dict)
    net.eval()
    net.to(device)
    iteration = checkpoint.get('iteration', '?')
    elo = checkpoint.get('elo', '?')
    return net, iteration, elo, channels, num_blocks


def find_checkpoints(directory='checkpoints/'):
    """Find best model and latest checkpoint."""
    best_path = os.path.join(directory, 'best_model.pt')
    pts = sorted([f for f in os.listdir(directory) if f.endswith('.pt') and f != 'best_model.pt' and os.path.isfile(os.path.join(directory, f))])
    latest_path = os.path.join(directory, pts[-1]) if pts else None
    return best_path if os.path.exists(best_path) else None, latest_path


def main():
    parser = argparse.ArgumentParser(description='Arena evaluation of checkpoints')
    parser.add_argument('--num-games', type=int, default=20,
                        help='Number of games in arena (default: 20)')
    parser.add_argument('--num-sims', type=int, default=400,
                        help='MCTS simulations per move (default: 400)')
    parser.add_argument('--checkpoint1', type=str, default=None,
                        help='Path to first checkpoint (default: best_model.pt)')
    parser.add_argument('--checkpoint2', type=str, default=None,
                        help='Path to second checkpoint (default: latest)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run on (default: cpu)')
    args = parser.parse_args()

    # Resolve checkpoints
    best_path, latest_path = find_checkpoints('checkpoints/')

    cp1_path = args.checkpoint1 or best_path
    cp2_path = args.checkpoint2 or latest_path

    if not cp1_path or not os.path.exists(cp1_path):
        print(f"❌ Cannot find checkpoint1: {cp1_path}")
        sys.exit(1)
    if not cp2_path or not os.path.exists(cp2_path):
        print(f"❌ Cannot find checkpoint2: {cp2_path}")
        sys.exit(1)

    print("=" * 64)
    print("  ULTIMATE TIC-TAC-TOE — Arena Evaluation")
    print("=" * 64)

    # Load networks
    net1, iter1, elo1, ch1, blk1 = load_network_from_checkpoint(cp1_path, args.device)
    net2, iter2, elo2, ch2, blk2 = load_network_from_checkpoint(cp2_path, args.device)

    print(f"\n  {cp1_path.split('/')[-1]}:")
    print(f"    Iteration: {iter1}, Elo: {elo1}, Network: {ch1}ch × {blk1}blocks")
    print(f"\n  {cp2_path.split('/')[-1]}:")
    print(f"    Iteration: {iter2}, Elo: {elo2}, Network: {ch2}ch × {blk2}blocks")

    # Warn if network architectures differ
    if ch1 != ch2 or blk1 != blk2:
        print(f"\n  ⚠️  Network architectures differ — comparison may not be fair!")
        print(f"     Net1: {ch1}ch×{blk1}blocks vs Net2: {ch2}ch×{blk2}blocks")

    # Run arena in both directions
    print(f"\n  Arena: {args.num_games} games, {args.num_sims} simulations/move")
    print(f"  Running...\n")

    t0 = time.time()

    # First direction: net1 (new) vs net2 (old)
    print(f"  Round 1: {cp1_path.split('/')[-1]} as challenger vs {cp2_path.split('/')[-1]}:")
    result = arena_mod.run_arena(
        net1, net2,
        num_games=args.num_games,
        num_simulations=args.num_sims,
        device=args.device,
    )
    print(f"    W: {result.wins:>3d}  L: {result.losses:>3d}  D: {result.draws:>3d}")
    print(f"    Win rate: {result.win_rate:.3f}")

    # Second direction: net2 (new) vs net1 (old)
    print(f"\n  Round 2: {cp2_path.split('/')[-1]} as challenger vs {cp1_path.split('/')[-1]}:")
    result2 = arena_mod.run_arena(
        net2, net1,
        num_games=args.num_games,
        num_simulations=args.num_sims,
        device=args.device,
    )
    print(f"    W: {result2.wins:>3d}  L: {result2.losses:>3d}  D: {result2.draws:>3d}")
    print(f"    Win rate: {result2.win_rate:.3f}")

    elapsed = time.time() - t0

    # Summary
    print(f"\n{'─' * 64}")
    print(f"  SUMMARY")
    print(f"  {'─' * 60}")
    print(f"  Round 1 — {cp1_path.split('/')[-1]} vs {cp2_path.split('/')[-1]}:")
    print(f"    Win rate: {result.win_rate:.3f}  "
          f"({'NEW is better' if result.new_is_better else 'NOT better'} at ≥{result.win_rate_threshold:.0%})")
    print(f"  Round 2 — {cp2_path.split('/')[-1]} vs {cp1_path.split('/')[-1]}:")
    print(f"    Win rate: {result2.win_rate:.3f}  "
          f"({'NEW is better' if result2.new_is_better else 'NOT better'} at ≥{result2.win_rate_threshold:.0%})")
    print(f"  {'─' * 60}")
    combined_win_rate = (result.win_rate + (1 - result2.win_rate)) / 2
    print(f"  Combined effective win rate (Net1): {combined_win_rate:.3f}")
    print(f"  Total time: {elapsed:.0f}s  ({elapsed/60:.1f}min)")
    print(f"  Arena games: {args.num_games * 2}")
    print(f"{'─' * 64}")

    # Compare with training metrics
    metrics_path = os.path.join('checkpoints', 'training_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        # Find matching metrics entries
        m1 = [m for m in metrics if m['iteration'] == iter1]
        m2 = [m for m in metrics if m['iteration'] == iter2]
        if m1 and m2:
            m1, m2 = m1[0], m2[0]
            print(f"\n  Training context:")
            print(f"    {cp1_path.split('/')[-1]} at iter {m1['iteration']}: "
                  f"loss={m1['loss_total']:.3f}, policy={m1['loss_policy']:.3f}, "
                  f"value={m1['loss_value']:.3f}")
            print(f"    {cp2_path.split('/')[-1]} at iter {m2['iteration']}: "
                  f"loss={m2['loss_total']:.3f}, policy={m2['loss_policy']:.3f}, "
                  f"value={m2['loss_value']:.3f}")


if __name__ == '__main__':
    main()