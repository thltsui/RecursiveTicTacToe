"""Training loop for the Tabular Q-Learning agent.

Usage:
    python tabular_q/train.py                    # defaults
    python tabular_q/train.py --episodes 10000   # more training
    python tabular_q/train.py --eval-every 500

Imports from 01_game via importlib (package name starts with digit).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

# ── repo root on path ─────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from importlib import import_module

_board = import_module("01_game.board")
_rules = import_module("01_game.rules")

GameState       = _board.GameState
create_initial_state = _board.create_initial_state
get_legal_moves = _rules.get_legal_moves
apply_move      = _rules.apply_move

sys.path.insert(0, os.path.dirname(__file__))
from agent import TabularQAgent  # noqa: E402
from evaluate import evaluate_vs_random  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Single training game
# ──────────────────────────────────────────────────────────────────────────────

def play_training_game(
    agent: TabularQAgent,
    sub_board_bonus: float = 0.02,
) -> int | None:
    """Run one game; both players are the same agent (self-play).

    Returns the winner: 1, -1, or 0 (draw).
    """
    state = create_initial_state()

    # Store (state, move, current_player) so we can assign rewards at the end.
    trajectory: list[tuple] = []

    while not state.is_terminal:
        legal = get_legal_moves(state)
        move  = agent.select_action(state, legal)

        prev_state = state
        state = apply_move(state, move)

        # Intermediate shaping: small bonus for winning a sub-board.
        intermediate = 0.0
        sb = move // 9
        if state.sub_board_results[sb] == prev_state.current_player:
            intermediate = sub_board_bonus

        trajectory.append((prev_state, move, prev_state.current_player, intermediate))

    # Assign terminal rewards and update backwards through trajectory.
    winner = state.winner   # 1, -1, or 0

    # Update from most-recent move backward (standard episodic Q-learning).
    final_state = state
    for i in reversed(range(len(trajectory))):
        s, m, player, shaping = trajectory[i]

        if i == len(trajectory) - 1:
            # Terminal transition
            if winner == player:
                reward = 1.0
            elif winner == -player:
                reward = -1.0
            else:
                reward = 0.0
            reward += shaping
            agent.update(s, m, reward, final_state, [], done=True)
        else:
            # Non-terminal: next state is the state at trajectory[i+1]
            next_s, next_m, _, _ = trajectory[i + 1]
            next_legal = get_legal_moves(next_s)
            agent.update(s, m, shaping, next_s, next_legal, done=False)

    return winner


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(
    episodes: int = 5_000,
    eval_every: int = 500,
    epsilon_start: float = 0.4,
    epsilon_end: float   = 0.05,
    alpha: float = 0.1,
    gamma: float = 0.99,
    seed: int   = 0,
    out_dir: str = "tabular_q/results",
) -> TabularQAgent:

    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    agent = TabularQAgent(epsilon=epsilon_start, alpha=alpha, gamma=gamma)
    log: list[dict] = []

    t0 = time.time()
    for ep in range(1, episodes + 1):
        # Linear epsilon decay
        agent.epsilon = epsilon_start - (epsilon_start - epsilon_end) * (ep / episodes)
        play_training_game(agent)

        if ep % eval_every == 0:
            # Greedy eval
            saved_eps     = agent.epsilon
            agent.epsilon = 0.0
            wr, dr, lr    = evaluate_vs_random(agent, n_games=300)
            agent.epsilon = saved_eps

            elapsed = time.time() - t0
            entry = {
                "episode": ep,
                "win_rate": round(wr, 4),
                "draw_rate": round(dr, 4),
                "loss_rate": round(lr, 4),
                "table_size": agent.table_size(),
                "elapsed_s": round(elapsed, 1),
            }
            log.append(entry)
            print(
                f"Ep {ep:6d} | "
                f"W {wr:.1%}  D {dr:.1%}  L {lr:.1%} | "
                f"table {agent.table_size():,} entries | "
                f"{elapsed:.0f}s"
            )

    # Save log
    log_path = os.path.join(out_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nLog saved to {log_path}")

    return agent


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tabular Q-agent on UTTT")
    parser.add_argument("--episodes",    type=int,   default=5_000)
    parser.add_argument("--eval-every",  type=int,   default=500)
    parser.add_argument("--alpha",       type=float, default=0.1)
    parser.add_argument("--gamma",       type=float, default=0.99)
    parser.add_argument("--eps-start",   type=float, default=0.4)
    parser.add_argument("--eps-end",     type=float, default=0.05)
    parser.add_argument("--seed",        type=int,   default=0)
    parser.add_argument("--out-dir",     type=str,   default="tabular_q/results")
    args = parser.parse_args()

    train(
        episodes      = args.episodes,
        eval_every    = args.eval_every,
        alpha         = args.alpha,
        gamma         = args.gamma,
        epsilon_start = args.eps_start,
        epsilon_end   = args.eps_end,
        seed          = args.seed,
        out_dir       = args.out_dir,
    )
