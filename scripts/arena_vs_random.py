#!/usr/bin/env python3
"""Evaluate best_model.pt against a purely random opponent (no MCTS, just sampling)."""

import sys, os, time, random, argparse

# Ensure project root is on path regardless of where script is run from
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
from importlib import import_module

board_mod = import_module('01_game.board')
rules_mod = import_module('01_game.rules')
network_mod = import_module('02_network.network')
search_mod = import_module('03_mcts.search')


def play_vs_random(network, num_simulations, device, verbose=False):
    """Play one game: network (player1) vs random player (player2).

    Network uses MCTS with temperature=0 (greedy).
    Random player samples uniformly from legal moves.

    Returns:
        1 if network wins, -1 if random wins, 0 if draw.
    """
    state = board_mod.create_initial_state()

    while not state.is_terminal:
        if state.current_player == 1:
            # Network's turn
            root = search_mod.run_mcts(
                state, network,
                num_simulations=num_simulations,
                dirichlet_epsilon=0.0,
                device=device,
            )
            move = search_mod.select_move(root, temperature=0.0)
        else:
            # Random player's turn
            legal_moves = rules_mod.get_legal_moves(state)
            if not legal_moves:
                # Should not happen in non-terminal states, but safety check
                break
            move = random.choice(legal_moves)

        state = rules_mod.apply_move(state, move)

    return state.winner if state.winner is not None else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--games', type=int, default=50)
    parser.add_argument('--sims', type=int, default=200)
    args = parser.parse_args()

    num_games = args.games
    num_sims = args.sims
    device = 'cpu'

    # Resolve checkpoint path
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint_path = args.checkpoint
    else:
        v3_dir = os.path.join(PROJECT_ROOT, 'checkpoints/large_v3_pure_self_play')
        if os.path.isdir(v3_dir):
            ckpts = sorted([f for f in os.listdir(v3_dir) if f.endswith('.pt') and 'checkpoint' in f])
            checkpoint_path = os.path.join(v3_dir, ckpts[-1]) if ckpts else None
        else:
            checkpoint_path = None

    if not checkpoint_path:
        print("ERROR: No checkpoint found."); sys.exit(1)

    print(f"Loading {checkpoint_path}...", flush=True)
    cp = torch.load(checkpoint_path, weights_only=False, map_location=device)
    sd = cp['network_state_dict']

    # Auto-detect network size from checkpoint weights
    # input_conv weight shape is (channels, 7, 3, 3)
    channels = sd['input_conv.weight'].shape[0]
    num_blocks = sum(1 for k in sd if k.startswith('trunk.') and k.endswith('.conv1.weight'))

    net = network_mod.UltimateTTTNetwork(channels=channels, num_blocks=num_blocks)
    net.load_state_dict(sd)
    net.eval()
    net.to(device)
    print(f"Loaded: {channels}ch x {num_blocks}blocks (iter {cp.get('iteration', '?')})", flush=True)
    print(f"\nPlaying {num_games} games: model (P1) vs random (P2)\n", flush=True)

    wins, losses, draws = 0, 0, 0
    t0 = time.time()

    for i in range(num_games):
        result = play_vs_random(net, num_sims, device)
        if result == 1:
            wins += 1
        elif result == -1:
            losses += 1
        else:
            draws += 1

        elapsed = time.time() - t0
        rate = (i + 1) / elapsed * 60
        print(f"  Game {i+1:2d}/{num_games}: AI {wins}W / {losses}L / {draws}D  "
              f"(wr={wins/(i+1):.3f})  [{elapsed:.0f}s @ {rate:.1f} games/min]",
              flush=True)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"RESULTS: best_model vs Random Opponent")
    print(f"{'=' * 60}")
    print(f"  Games: {num_games} ({num_sims} sims/move)")
    print(f"  AI wins:  {wins}")
    print(f"  Losses:   {losses}")
    print(f"  Draws:    {draws}")
    print(f"  Win rate: {wins/(wins+losses+draws):.3f}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"\n  Interpretation:")
    print(f"    wr ~ 1.000: AI is extremely strong, crushes random play")
    print(f"    wr ~ 0.900: AI has solid strategy but still makes mistakes")
    print(f"    wr ~ 0.500: AI barely better than random")
    print(f"    wr ~ 0.000: AI is terrible")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()