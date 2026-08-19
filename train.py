"""Launch the AlphaZero loop from one validated JSON configuration.

Loading or validating a configuration does not start training.  A run begins
only when this file is executed and reaches ``trainer.train(config)``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from importlib import import_module


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(
    PROJECT_ROOT, "configs", "training", "lite_transformer.json"
)
sys.path.insert(0, PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Ultimate Tic-Tac-Toe from a validated JSON config."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Training JSON path (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Validate and print the canonical config, then exit without training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer_mod = import_module("04_training.trainer")
    model_factory_mod = import_module("02_network.model_factory")

    config = trainer_mod.load_training_config(args.config)
    if args.print_config:
        print(json.dumps(config.to_dict(), indent=2))
        return

    # Construction here is only for a truthful pre-run summary. The trainer
    # creates the actual model after the user explicitly invokes this command.
    net = model_factory_mod.create_network(config.model_config())
    num_params = sum(parameter.numel() for parameter in net.parameters())
    del net

    print("=" * 64)
    print("  Ultimate TTT — JSON-configured AlphaZero training")
    print("=" * 64)
    print(f"  Config:       {os.path.abspath(args.config)}")
    print(
        f"  Model:        {config.architecture} {config.channels}ch x "
        f"{config.num_blocks} ({num_params:,} parameters)"
    )
    print(
        f"  Self-play:    {config.num_self_play} games + "
        f"{config.num_reanalyzed} reanalysed + {config.num_vs_best} vs best"
    )
    print(
        f"  Search:       {config.num_simulations} simulations/move, "
        f"batch {config.self_play_batch_size} games"
    )
    print(
        f"  Optimizer:    {config.batches_per_iteration} batches x "
        f"{config.batch_size} positions"
    )
    print(f"  Checkpoints:  {config.checkpoint_dir}")
    print("  Ctrl+C stops the run and writes a final checkpoint.")
    print("=" * 64)

    trainer_mod.train(config)


if __name__ == "__main__":
    main()
