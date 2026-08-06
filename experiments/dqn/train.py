"""DQN training loop for Ultimate Tic-Tac-Toe.

The DQN trains against a uniform-random opponent and alternates colours between
episodes. Only the DQN's decisions are stored. Each replay transition spans from
one DQN turn to its next turn (after the opponent response), keeping both states
in the learning agent's perspective so ordinary one-step DQN bootstrapping
applies without a zero-sum sign flip.

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

import torch

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
    agent_player: int,
    rng: random.Random,
    sub_board_bonus: float = 0.05,
    train_every: int = 4,
    learning_starts: int = 1_000,
) -> int | None:
    """Train for one game against a random opponent.

    A stored transition begins on the DQN's turn and ends after the opponent
    response, when it is the DQN's turn again. This makes the next-state Q-value
    use the same player perspective as the current state.

    Returns: winner (1, -1, or 0).
    """
    if agent_player not in (-1, 1):
        raise ValueError("agent_player must be 1 or -1")

    state = create_initial_state()

    while not state.is_terminal:
        # When the DQN is O, the random opponent makes the opening move.
        if state.current_player != agent_player:
            state = apply_move(state, rng.choice(get_legal_moves(state)))
            if state.is_terminal:
                break

        prev = state
        move = agent.select_action(state)
        state = apply_move(state, move)

        reward = 0.0
        if state.sub_board_results[move // 9] == prev.current_player:
            reward += sub_board_bonus

        if state.is_terminal:
            if state.winner == agent_player:
                reward += 1.0
            agent.observe(
                prev,
                move,
                reward,
                state,
                done=True,
                train_every=train_every,
                learning_starts=learning_starts,
            )
            break

        # Advance through the random opponent's response so the replay
        # transition ends on the DQN's perspective again.
        opponent_prev = state
        opponent_move = rng.choice(get_legal_moves(state))
        state = apply_move(state, opponent_move)
        if state.sub_board_results[opponent_move // 9] == opponent_prev.current_player:
            reward -= sub_board_bonus

        done = state.is_terminal
        if done and state.winner == -agent_player:
            reward -= 1.0

        agent.observe(
            prev,
            move,
            reward,
            state,
            done=done,
            train_every=train_every,
            learning_starts=learning_starts,
        )

    return state.winner


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(
    episodes: int             = 20_000,
    eval_every: int           = 1_000,
    eval_games: int           = 300,
    epsilon_start: float      = 0.5,
    epsilon_end: float        = 0.05,
    lr: float                 = 1e-4,
    gamma: float              = 0.99,
    batch_size: int           = 32,
    target_update_freq: int   = 500,
    buffer_capacity: int      = 100_000,
    train_every: int          = 4,
    learning_starts: int      = 1_000,
    seed: int                 = 0,
    device: str               = "cpu",
    checkpoint_dir: str       = "experiments/dqn/checkpoints",
    out_dir: str              = "experiments/dqn/results",
) -> DQNAgent:

    random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)
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
        agent_player = 1 if ep % 2 == 1 else -1
        play_training_game(
            agent,
            agent_player=agent_player,
            rng=rng,
            train_every=train_every,
            learning_starts=learning_starts,
        )

        if ep % eval_every == 0:
            saved_eps     = agent.epsilon
            agent.epsilon = 0.0
            wr, dr, lr_   = evaluate_vs_random(agent, n_games=eval_games)
            agent.epsilon = saved_eps

            elapsed = time.time() - t0
            entry = {
                "episode": ep,
                "win_rate":  round(wr,  4),
                "draw_rate": round(dr,  4),
                "loss_rate": round(lr_, 4),
                "buffer_size": len(agent.buffer),
                "grad_steps": agent._grad_steps,
                "environment_steps": agent._environment_steps,
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
    run_record = {
        "config": {
            "episodes": episodes,
            "eval_every": eval_every,
            "eval_games": eval_games,
            "epsilon_start": epsilon_start,
            "epsilon_end": epsilon_end,
            "learning_rate": lr,
            "gamma": gamma,
            "batch_size": batch_size,
            "target_update_freq": target_update_freq,
            "buffer_capacity": buffer_capacity,
            "train_every": train_every,
            "learning_starts": learning_starts,
            "seed": seed,
            "device": device,
            "torch_version": torch.__version__,
        },
        "results": log,
    }
    with open(log_path, "w") as f:
        json.dump(run_record, f, indent=2)
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
    parser.add_argument("--eval-games",         type=int,   default=300)
    parser.add_argument("--lr",                 type=float, default=1e-4)
    parser.add_argument("--gamma",              type=float, default=0.99)
    parser.add_argument("--batch-size",         type=int,   default=32)
    parser.add_argument("--target-update-freq", type=int,   default=500)
    parser.add_argument("--buffer-capacity",    type=int,   default=100_000)
    parser.add_argument("--train-every",        type=int,   default=4)
    parser.add_argument("--learning-starts",    type=int,   default=1_000)
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
        eval_games         = args.eval_games,
        lr                 = args.lr,
        gamma              = args.gamma,
        batch_size         = args.batch_size,
        target_update_freq = args.target_update_freq,
        buffer_capacity    = args.buffer_capacity,
        train_every        = args.train_every,
        learning_starts    = args.learning_starts,
        epsilon_start      = args.eps_start,
        epsilon_end        = args.eps_end,
        seed               = args.seed,
        device             = args.device,
        checkpoint_dir     = args.checkpoint_dir,
        out_dir            = args.out_dir,
    )
