#!/usr/bin/env python3
"""Quick arena test — just a few games to gauge speed and progress."""

import sys, os, time, json

sys.path.insert(0, os.path.dirname(__file__))

import torch
from importlib import import_module

arena_mod = import_module('06_evaluation.arena')
network_mod = import_module('02_network.network')


def load_net(path, device='cpu'):
    cp = torch.load(path, weights_only=False, map_location=device)
    sd = cp['network_state_dict']
    first_key = [k for k in sd if 'conv.weight' in k and 'input_' in k][0]
    channels = sd[first_key].shape[0]
    num_blocks = sum(1 for k in sd if '.conv1.weight' in k)
    net = network_mod.UltimateTTTNetwork(channels=channels, num_blocks=num_blocks)
    net.load_state_dict(sd)
    net.eval()
    net.to(device)
    return net, cp.get('iteration', '?'), channels, num_blocks


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-games', type=int, default=10)
    parser.add_argument('--num-sims', type=int, default=200)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    print("=" * 60)
    print("  ULTIMATE TTT — Arena Evaluation")
    print("=" * 60)
    print(f"  Games: {args.num_games} | Sims: {args.num_sims} | Device: {args.device}")

    print("\n  Loading networks...", flush=True)
    t0 = time.time()

    net_best, iter_best, ch1, blk1 = load_net('checkpoints/best_model.pt', args.device)
    net_latest, iter_latest, ch2, blk2 = load_net('checkpoints/checkpoint_iter00210_elo1.pt', args.device)

    print(f"  best_model.pt:        iter {iter_best} | {ch1}ch x {blk1}blocks")
    print(f"  checkpoint_iter00210: iter {iter_latest} | {ch2}ch x {blk2}blocks")
    print(f"  Loaded in {time.time()-t0:.1f}s\n", flush=True)

    # Round 1: latest as challenger vs best
    print(f"  Round 1 — iter {iter_latest} (challenger) vs iter {iter_best} (best)")
    print(f"  {'─' * 56}", flush=True)
    r1 = arena_mod.run_arena(net_latest, net_best, args.num_games, args.num_sims, device=args.device)
    print(f"  W: {r1.wins}  L: {r1.losses}  D: {r1.draws}  Win rate: {r1.win_rate:.3f}")
    print(f"  Better? {'YES' if r1.new_is_better else 'NO'} (threshold >= {r1.win_rate_threshold:.0%})\n", flush=True)

    # Round 2: best as challenger vs latest
    print(f"  Round 2 — iter {iter_best} (challenger) vs iter {iter_latest} (best)")
    print(f"  {'─' * 56}", flush=True)
    r2 = arena_mod.run_arena(net_best, net_latest, args.num_games, args.num_sims, device=args.device)
    print(f"  W: {r2.wins}  L: {r2.losses}  D: {r2.draws}  Win rate: {r2.win_rate:.3f}")
    print(f"  Better? {'YES' if r2.new_is_better else 'NO'} (threshold >= {r2.win_rate_threshold:.0%})\n", flush=True)

    # Summary
    combined = (r1.win_rate + (1 - r2.win_rate)) / 2
    print(f"  {'═' * 56}")
    print(f"  SUMMARY")
    print(f"  {'═' * 56}")
    print(f"  Combined win rate (iter {iter_latest}): {combined:.3f}")
    print(f"  Round 1: iter {iter_latest} vs iter {iter_best}: {r1.wins}W / {r1.losses}L / {r1.draws}D (wr={r1.win_rate:.3f})")
    print(f"  Round 2: iter {iter_best} vs iter {iter_latest}: {r2.wins}W / {r2.losses}L / {r2.draws}D (wr={r2.win_rate:.3f})")
    print(f"  Total games: {args.num_games * 2}")

    # Compare training metrics
    metrics_path = 'checkpoints/training_metrics.json'
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        m1 = next((m for m in metrics if m['iteration'] == iter_best), None)
        m2 = next((m for m in metrics if m['iteration'] == iter_latest), None)
        if m1 and m2:
            print(f"\n  Training metrics comparison:")
            print(f"    iter {iter_best}: loss={m1['loss_total']:.3f} | policy={m1['loss_policy']:.3f} | value={m1['loss_value']:.3f}")
            print(f"    iter {iter_latest}: loss={m2['loss_total']:.3f} | policy={m2['loss_policy']:.3f} | value={m2['loss_value']:.3f}")


if __name__ == '__main__':
    main()