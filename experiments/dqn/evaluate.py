"""Evaluation utilities for the DQN agent.

Can be run standalone to evaluate a saved checkpoint:
    python experiments/dqn/evaluate.py --checkpoint path/to/dqn.pt
"""
from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module

_board = import_module("01_game.board")
_rules = import_module("01_game.rules")

create_initial_state = _board.create_initial_state
get_legal_moves      = _rules.get_legal_moves
apply_move           = _rules.apply_move


def evaluate_vs_random(
    agent,
    n_games: int = 500,
    seed: int    = 42,
) -> tuple[float, float, float]:
    """Evaluate the DQN agent against a uniform-random opponent.

    Plays n_games//2 as X and n_games//2 as O for an unbiased estimate.

    Returns:
        (win_rate, draw_rate, loss_rate)
    """
    rng = random.Random(seed)
    wins = draws = losses = 0

    for game_idx in range(n_games):
        agent_player = 1 if game_idx % 2 == 0 else -1
        state = create_initial_state()

        while not state.is_terminal:
            legal = get_legal_moves(state)
            if state.current_player == agent_player:
                move = agent.select_action(state)
            else:
                move = rng.choice(legal)
            state = apply_move(state, move)

        if state.winner == agent_player:
            wins += 1
        elif state.winner == 0:
            draws += 1
        else:
            losses += 1

    total = n_games
    return wins / total, draws / total, losses / total


def print_eval(agent, n_games: int = 500) -> None:
    saved_eps     = agent.epsilon
    agent.epsilon = 0.0
    wr, dr, lr    = evaluate_vs_random(agent, n_games=n_games)
    agent.epsilon = saved_eps
    print(f"Greedy eval over {n_games} games:")
    print(f"  Win rate:  {wr:.1%}")
    print(f"  Draw rate: {dr:.1%}")
    print(f"  Loss rate: {lr:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to a saved .pt checkpoint")
    parser.add_argument("--n-games", type=int, default=500)
    args = parser.parse_args()

    from agent import DQNAgent
    agent = DQNAgent()
    agent.load(args.checkpoint)
    agent.epsilon = 0.0
    print(f"Loaded checkpoint: {args.checkpoint}")
    print_eval(agent, n_games=args.n_games)
