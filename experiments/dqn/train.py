"""DQN training loop for Ultimate Tic-Tac-Toe.

The agent trains by playing against a random opponent (alternating colours).
Both the agent's own moves AND the opponent's moves are stored in the replay
buffer — the opponent's transitions are stored from the opponent's perspective
(board is always encoded from the current player's viewpoint, so this is
automatically handled by encode_state).

Usage:
    python experiments/dqn/train.py
    python experiments/dqn/train.py --episodes 30000 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module

_board = import_module("01_game.board")
_rules = import_module("01_game.rules")

create_initial_state = _board.create_initial_state
get_legal_moves      = _rules.get_legal_moves
apply_move           = _rules.apply_move

from agent import DQNAgent          # noqa: E402
from evaluate import evaluate_vs_random  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Single training game
# ──────────────────────────────────────────────────────────────────────────────

def play_training_game(
    agent: DQNAgent,
    sub_board_bonus: float = 0.05,
) -> int | None:
    """Play one game and push all transitions to the agent's replay buffer.

    The agent plays as both colours in alternate games (handled externally
    by calling this function repeatedly). Within a single game the agent
    acts for every move — we treat this as self-play against itself, which
    avoids the need for a second agent object.

    Returns: winner (1, -1, or 0).
    """
    state = create_initial_state()
    transitions: list[tuple] = []   # (prev_state, move, prev_player, shaping)

    while not state.is_terminal:
        prev = state
        move = agent.select_action(state)
        state = apply_move(state, move)

        # Sub-board shaping reward
        shaping = 0.0
        if state.sub_board_results[move // 9] == prev.current_player:
            shaping = sub_board_bonus

        transitions.append((prev, move, prev.current_player, shaping))
        agent.train_step()

    winner = state.winner

    # Push terminal transitions
    for prev_s, m, player, shaping in transitions:
        if winner == player:
            terminal_r = 1.0
        elif winner == -player:
            terminal_r = -1.0
        else:
            terminal_r = 0.0
        agent.store(prev_s, m, terminal_r + shaping, state, done=True)

    return winner


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(
    episodes: int             = 20_000,
    eval_every: int           = 1_000,
    epsilon_start: float      = 0.5,
    epsilon_end: float        = 0.05,
    lr: float                 = 1e-3,
    gamma: float              = 0.99,
    batch_size: int           = 256,
    target_update_freq: int   = 500,
    buffer_capacity: int      = 100_000,
    seed: int                 = 0,
    device: str               = "cpu",
    checkpoint_dir: str       = "experiments/dqn/checkpoints",
    out_dir: str              = "experiments/dqn/results",
) -> DQNAgent:

    random.seed(seed)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    agent = DQNAgent(
        lr=lr, gamma=gamma, epsilon=epsilon_start,
        batch_size=batch_size, target_update_freq=target_update_freq,
        buffer_capacity=buffer_capacity, device=device,
    )
    log: list[dict] = []
    t0 = time.time()

    for ep in range(1, episodes + 1):
        # Linear epsilon decay
        agent.epsilon = epsilon_start - (epsilon_start - epsilon_end) * (ep / episodes)
        play_training_game(agent)

        if ep % eval_every == 0:
            saved_eps     = agent.epsilon
            agent.epsilon = 0.0
            wr, dr, lr_   = evaluate_vs_random(agent, n_games=300)
            agent.epsilon = saved_eps

            elapsed = time.time() - t0
            entry = {
                "episode": ep,
                "win_rate":  round(wr,  4),
                "draw_rate": round(dr,  4),
                "loss_rate": round(lr_, 4),
                "buffer_size": len(agent.buffer),
                "grad_steps": agent._grad_steps,
                "elapsed_s": round(elapsed, 1),
            }
            log.append(entry)
            print(
                f"Ep {ep:7d} | "
                f"W {wr:.1%}  D {dr:.1%}  L {lr_:.1%} | "
                f"buf {len(agent.buffer):,}  steps {agent._grad_steps:,} | "
                f"{elapsed:.0f}s"
            )

            # Save checkpoint every 5 eval points
            if (ep // eval_every) % 5 == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"dqn_ep{ep:07d}.pt")
                agent.save(ckpt_path)

    log_path = os.path.join(out_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nLog saved to {log_path}")

    final_ckpt = os.path.join(checkpoint_dir, "dqn_final.pt")
    agent.save(final_ckpt)
    print(f"Final checkpoint: {final_ckpt}")

    return agent


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN agent on UTTT")
    parser.add_argument("--episodes",           type=int,   default=20_000)
    parser.add_argument("--eval-every",         type=int,   default=1_000)
    parser.add_argument("--lr",                 type=float, default=1e-3)
    parser.add_argument("--gamma",              type=float, default=0.99)
    parser.add_argument("--batch-size",         type=int,   default=256)
    parser.add_argument("--target-update-freq", type=int,   default=500)
    parser.add_argument("--buffer-capacity",    type=int,   default=100_000)
    parser.add_argument("--eps-start",          type=float, default=0.5)
    parser.add_argument("--eps-end",            type=float, default=0.05)
    parser.add_argument("--seed",               type=int,   default=0)
    parser.add_argument("--device",             type=str,   default="cpu")
    parser.add_argument("--checkpoint-dir",     type=str,   default="experiments/dqn/checkpoints")
    parser.add_argument("--out-dir",            type=str,   default="experiments/dqn/results")
    args = parser.parse_args()

    train(
        episodes           = args.episodes,
        eval_every         = args.eval_every,
        lr                 = args.lr,
        gamma              = args.gamma,
        batch_size         = args.batch_size,
        target_update_freq = args.target_update_freq,
        buffer_capacity    = args.buffer_capacity,
        epsilon_start      = args.eps_start,
        epsilon_end        = args.eps_end,
        seed               = args.seed,
        device             = args.device,
        checkpoint_dir     = args.checkpoint_dir,
        out_dir            = args.out_dir,
    )
