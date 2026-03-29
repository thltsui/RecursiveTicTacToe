"""Launch training on Mac Mini M2.

Optimized settings:
    - 64ch/4 blocks (~160K params) — fast iteration, good enough to learn tactics
    - CPU for MCTS self-play (faster than MPS for batch=1)
    - MPS for batched gradient updates
    - ~1 minute per iteration
"""

import sys
import os
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))

from importlib import import_module


def main():
    trainer_mod = import_module('04_training.trainer')
    network_mod = import_module('02_network.network')

    TrainingConfig = trainer_mod.TrainingConfig

    config = TrainingConfig(
        # Network — smaller for fast iteration on M2
        channels=64,
        num_blocks=4,

        # Self-play — CPU is faster for batch=1 MCTS inference
        device='cpu',           # MCTS runs on CPU
        games_per_iteration=25,
        num_simulations=50,     # lighter search, still learns
        temperature_threshold=15,

        # Training
        batch_size=128,
        batches_per_iteration=50,
        learning_rate=0.002,
        weight_decay=1e-4,
        grad_clip_norm=1.0,

        # Loss weights (KataGo defaults)
        lambda_value=1.0,
        lambda_score=0.5,
        lambda_ownership=0.5,
        lambda_opp=0.15,

        # Evaluation
        arena_every_n=10,       # arena every 10 iterations
        arena_games=20,         # quick arena
        win_rate_threshold=0.55,

        # Buffer
        buffer_capacity=100_000,

        # Checkpointing
        checkpoint_dir='checkpoints/',
        seed=42,
    )

    # Print estimated timing
    net = network_mod.UltimateTTTNetwork(channels=config.channels,
                                          num_blocks=config.num_blocks)
    n_params = sum(p.numel() for p in net.parameters())
    del net

    print("=" * 60)
    print("  Ultimate TTT — AlphaZero Training")
    print("=" * 60)
    print(f"  Device:       {config.device} (MCTS) + MPS (training)")
    print(f"  Network:      {config.channels}ch x {config.num_blocks} blocks ({n_params:,} params)")
    print(f"  Self-play:    {config.games_per_iteration} games x {config.num_simulations} sims/move")
    print(f"  Training:     {config.batches_per_iteration} batches x B={config.batch_size}")
    print(f"  Arena:        every {config.arena_every_n} iters, {config.arena_games} games")
    print(f"  Buffer:       {config.buffer_capacity:,} capacity")
    print(f"  Estimated:    ~1-2 min/iteration")
    print("=" * 60)
    print("  Ctrl+C to stop (checkpoint auto-saved)")
    print("=" * 60)
    print()

    trainer_mod.train(config)


if __name__ == '__main__':
    main()
