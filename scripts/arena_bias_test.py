#!/usr/bin/env python3
"""Diagnose first-player advantage by having best_model play itself."""

import sys, os, time

sys.path.insert(0, os.path.dirname(__file__))

import torch
from importlib import import_module

arena_mod = import_module('06_evaluation.arena')
network_mod = import_module('02_network.network')


def main():
    num_games = 50
    num_sims = 200
    device = 'cpu'

    print("Loading best_model.pt...", flush=True)
    cp = torch.load('checkpoints/best_model.pt', weights_only=False, map_location=device)
    sd = cp['network_state_dict']
    channels, num_blocks = 128, 8
    net = network_mod.UltimateTTTNetwork(channels=channels, num_blocks=num_blocks)
    net.load_state_dict(sd)
    net.eval()
    net.to(device)
    print(f"Loaded: 128ch x 8blocks (iter {cp.get('iteration', '?')})", flush=True)
    print(f"\nPlaying {num_games} games: net (P1) vs net (P2)\n", flush=True)

    wins, losses, draws = 0, 0, 0
    t0 = time.time()

    for i in range(num_games):
        result = arena_mod.play_single_game(net, net, num_sims, device)
        if result == 1:
            wins += 1
        elif result == -1:
            losses += 1
        else:
            draws += 1

        elapsed = time.time() - t0
        rate = (i + 1) / elapsed * 60
        print(f"  Game {i+1:2d}/{num_games}: P1 {wins}W / {losses}L / {draws}D  "
              f"(wr={wins/(i+1):.3f})  [{elapsed:.0f}s @ {rate:.1f} games/min]",
              flush=True)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"FIRST-PLAYER ADVANTAGE TEST RESULTS")
    print(f"{'=' * 60}")
    print(f"  Games: {num_games} (200 sims/move)")
    print(f"  Player 1: {wins}W / {losses}L / {draws}D")
    print(f"  Win rate as P1: {wins/(wins+losses+draws):.3f}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"\n  Interpretation:")
    print(f"    wr ~ 0.500: no advantage — model is symmetric")
    print(f"    wr > 0.550: significant first-player advantage exists")
    print(f"    wr > 0.700: strong first-player advantage")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()