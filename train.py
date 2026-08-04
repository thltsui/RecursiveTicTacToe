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

        # Self-play mix
        device='cpu',  # CPU used for MCTS to avoid MPS overhead for batch=1
        num_self_play=10,
        num_vs_random=5,
        num_vs_best=5,
        num_simulations=1200,
        temp_initial=2.0,
        temp_decay_rate=0.94,
        temp_min=0.15,
        dirichlet_alpha=0.3,       # Explicitly set for tuning
        dirichlet_epsilon=0.35,    # Explicitly set for tuning

        # Early-ply exploration boost + forced opening diversity
        # (added after diagnosing opening-move search collapse: some openings
        # received search visits in <5% of late-training games and were never
        # once MCTS's top choice, letting an early belief compound unchecked)
        dirichlet_epsilon_boost=0.55,   # root epsilon for the first few plies
        dirichlet_boost_plies=5,        # boost applies to moves 0-4
        forced_opening_fraction=0.2,    # 20% of self-play games/iteration forced
                                         # through the center-sub-board opening pool

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
        max_iterations=1000,
        early_stopping_patience=15,

        # Buffer
        buffer_capacity=200_000,

        # Checkpointing — fresh start with pure self-play
        checkpoint_dir='checkpoints/large_v5_exp_temp/',
        checkpoint_every_n=5,
        pretrain_checkpoint=None,  # Start from scratch due to architecture change
        train_device='mps',        # MPS explicitly used for batched training updates
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
    print(f"  Self-play:    {config.num_self_play} SP + {config.num_vs_random} vs Rand + {config.num_vs_best} vs Best ({config.num_simulations} sims/move)")
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
