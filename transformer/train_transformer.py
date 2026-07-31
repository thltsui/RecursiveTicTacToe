import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from importlib import import_module

# Import the transformer network
from transformer_network import TransformerTTTNetwork

def main():
    trainer_mod = import_module('04_training.trainer')
    network_mod = import_module('02_network.network')

    # Monkey patch the network module to use our Transformer
    network_mod.UltimateTTTNetwork = TransformerTTTNetwork
    
    # We use num_blocks as num_layers to reuse the config seamlessly
    TrainingConfig = trainer_mod.TrainingConfig
    config = TrainingConfig(
        channels=128,
        num_blocks=4, # 4 Transformer layers
        
        # Self-play mix
        device='cpu',
        num_self_play=10,
        num_vs_random=5,
        num_vs_best=5,
        num_simulations=200,
        temp_initial=2.0,
        temp_decay_rate=0.94,
        temp_min=0.15,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.35,

        # Early-ply exploration boost
        dirichlet_epsilon_boost=0.55,
        dirichlet_boost_plies=5,
        forced_opening_fraction=0.2,

        # Training
        batch_size=256,
        batches_per_iteration=100,
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
        arena_every_n=10,
        arena_games=50,
        win_rate_threshold=0.55,
        max_iterations=1000,
        early_stopping_patience=15,

        # Buffer
        buffer_capacity=200_000,

        # Checkpointing
        checkpoint_dir='checkpoints/transformer_v1/',
        checkpoint_every_n=5,
        pretrain_checkpoint=None,
        train_device='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'),
        seed=42,
    )
    
    # Print estimated timing and parameters
    net = TransformerTTTNetwork(channels=config.channels, num_blocks=config.num_blocks)
    n_params = sum(p.numel() for p in net.parameters())
    del net

    print("=" * 60)
    print("  Ultimate TTT — AlphaZero Training (TRANSFORMER)")
    print("=" * 60)
    print(f"  Device:       {config.device} (MCTS) + {config.train_device} (training)")
    print(f"  Network:      {config.channels}ch x {config.num_blocks} layers ({n_params:,} params)")
    print("=" * 60)
    
    trainer_mod.train(config)

if __name__ == '__main__':
    main()
