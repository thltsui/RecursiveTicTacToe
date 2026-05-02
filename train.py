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
        # Network — large: 192ch x 10 blocks (~6.78M params)
        channels=192,
        num_blocks=10,

        # Self-play — 100% pure self-play with anti-draw value shaping
        device='cpu',
        games_per_iteration=20,
        num_simulations=200,
        temperature_threshold=30,  # AlphaZero standard: temp=1 for first 30 moves
        self_play_with_best=False,

        # Training
        batch_size=256,
        batches_per_iteration=100,  # auto-capped if buffer is small
        learning_rate=0.001,
        weight_decay=1e-4,
        grad_clip_norm=1.0,
        lr_decay_every_n=50,
        lr_decay_gamma=0.5,

        # Loss weights
        lambda_value=1.0,
        lambda_score=0.5,
        lambda_ownership=0.5,
        lambda_opp=0.15,

        # Evaluation
        arena_every_n=10,       # arena every 10 iterations to catch issues early
        arena_games=50,
        win_rate_threshold=0.55,

        # Buffer
        buffer_capacity=200_000,

        # Checkpointing — fresh start with pure self-play
        checkpoint_dir='checkpoints/large_v3_pure_self_play/',
        checkpoint_every_n=5,
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
    print(f"  Checkpoint:   every {config.checkpoint_every_n} iters")
    print("=" * 60)
    print("  Ctrl+C to stop (checkpoint auto-saved)")
    print("=" * 60)
    print()

    trainer_mod.train(config)


if __name__ == '__main__':
    main()
