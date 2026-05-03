#!/usr/bin/env python3
"""Quick eval of the large model vs random and vs original best_model.pt."""

import sys, os, random

sys.path.insert(0, os.path.dirname(__file__))

import torch
from importlib import import_module

board_mod = import_module('01_game.board')
rules_mod = import_module('01_game.rules')
search_mod = import_module('03_mcts.search')
network_mod = import_module('02_network.network')
arena_mod = import_module('06_evaluation.arena')

device = 'cpu'
num_sims = 200


def eval_vs_random(net, num_games=20):
    wins, losses, draws = 0, 0, 0
    for i in range(num_games):
        state = board_mod.create_initial_state()
        while not state.is_terminal:
            if state.current_player == 1:
                root = search_mod.run_mcts(state, net, num_simulations=num_sims,
                                           dirichlet_epsilon=0.0, device=device)
                move = search_mod.select_move(root, temperature=0.0)
            else:
                legal = rules_mod.get_legal_moves(state)
                move = random.choice(legal)
            state = rules_mod.apply_move(state, move)
        if state.winner == 1:
            wins += 1
        elif state.winner == -1:
            losses += 1
        else:
            draws += 1
        print(f"  [Random] Game {i+1}/{num_games}: {'W' if state.winner==1 else 'L' if state.winner==-1 else 'D'}"
              f"  ({wins}W/{losses}L/{draws}D)", flush=True)
    print(f"\n  RESULT vs Random: {wins}W / {losses}L / {draws}D  (wr={wins/(wins+losses+draws):.3f})", flush=True)
    return wins, losses, draws


def eval_vs_best_model(net, net_best, num_games=20):
    wins, losses, draws = 0, 0, 0
    for i in range(num_games):
        r = arena_mod.play_single_game(net, net_best, num_sims, device)
        if r == 1:
            wins += 1
        elif r == -1:
            losses += 1
        else:
            draws += 1
        print(f"  [vs Best] Game {i+1}/{num_games}: {'W' if r==1 else 'L' if r==-1 else 'D'}"
              f"  ({wins}W/{losses}L/{draws}D)", flush=True)
    print(f"\n  RESULT vs Best: {wins}W / {losses}L / {draws}D  (wr={wins/(wins+losses+draws):.3f})", flush=True)
    return wins, losses, draws


def main():
    # Find the latest checkpoint in large_192ch_10blocks
    ckpt_dir = 'checkpoints/large_192ch_10blocks'
    ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pt') and f != 'best_model.pt'])
    latest = ckpts[-1] if ckpts else None
    if not latest:
        print("No checkpoints found!")
        return
    ckpt_path = os.path.join(ckpt_dir, latest)
    print(f"Loading: {ckpt_path}", flush=True)

    cp = torch.load(ckpt_path, weights_only=False, map_location=device)
    net = network_mod.UltimateTTTNetwork(channels=192, num_blocks=10)
    net.load_state_dict(cp['network_state_dict'])
    net.eval()
    print(f"  Iteration: {cp.get('iteration', '?')}, Elo: {cp.get('elo', '?')}", flush=True)

    # Load original best_model.pt
    print("\nLoading original best_model.pt (128ch x 8blocks)...", flush=True)
    cp_best = torch.load('checkpoints/best_model.pt', weights_only=False, map_location=device)
    net_best = network_mod.UltimateTTTNetwork(channels=128, num_blocks=8)
    net_best.load_state_dict(cp_best['network_state_dict'])
    net_best.eval()
    print(f"  Iteration: {cp_best.get('iteration', '?')}, Elo: {cp_best.get('elo', '?')}", flush=True)

    print("\n" + "=" * 60)
    print("  EVALUATION 1: Large Model vs Random Opponent")
    print("=" * 60)
    eval_vs_random(net, num_games=20)

    print("\n" + "=" * 60)
    print("  EVALUATION 2: Large Model vs Original Best Model")
    print("=" * 60)
    eval_vs_best_model(net, net_best, num_games=20)


if __name__ == '__main__':
    main()