"""Evaluation utilities for the Tabular Q-agent.

Can be run standalone:
    python tabular_q/evaluate.py   (runs a quick 500-game eval of a fresh agent)
"""
from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from importlib import import_module

_board = import_module("01_game.board")
_rules = import_module("01_game.rules")

create_initial_state = _board.create_initial_state
get_legal_moves      = _rules.get_legal_moves
apply_move           = _rules.apply_move


def evaluate_vs_random(
    agent,
    n_games: int = 500,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Evaluate the agent against a uniformly-random opponent.

    The agent plays as X (current_player=1) for the first half of games
    and as O (current_player=-1) for the second half, to get an unbiased
    estimate regardless of first-mover advantage.

    Returns:
        (win_rate, draw_rate, loss_rate)
    """
    rng = random.Random(seed)
    wins = draws = losses = 0

    for game_idx in range(n_games):
        # Alternate which colour the agent plays
        agent_player = 1 if game_idx % 2 == 0 else -1
        state = create_initial_state()

        while not state.is_terminal:
            legal = get_legal_moves(state)
            if state.current_player == agent_player:
                move = agent.select_action(state, legal)
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
    print(f"  Q-table entries: {agent.table_size():,}")


if __name__ == "__main__":
    # Quick smoke test: untrained agent should be ~50% win rate
    sys.path.insert(0, os.path.dirname(__file__))
    from agent import TabularQAgent
    fresh = TabularQAgent(epsilon=0.0)
    print("Untrained agent (random behaviour, epsilon=0 → all Q=0 → first legal move):")
    print_eval(fresh, n_games=200)
