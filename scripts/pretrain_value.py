#!/usr/bin/env python3
"""Legacy random-policy representation bootstrap (disabled by default).

Usage:
    python scripts/pretrain_value.py [--games N] [--batches N] [--checkpoint PATH]

Random-vs-random outcomes estimate a different policy than MCTS self-play and
must not be interpreted as calibrated strong-play W/D/L probabilities.  The
default training loop no longer uses this path.  The acknowledgement flag is
required to run the old representation-learning experiment intentionally.

This script implements the two legacy pre-AlphaZero stages:

    Stage 0  Generate --games random-vs-random games and populate a ReplayBuffer.
    Stage 1  Train with lambda_policy=0 (value/score/ownership only) for --batches
             gradient steps, then save the pretrain checkpoint.

The checkpoint is then passed to TrainingConfig(pretrain_checkpoint=PATH) so that
the normal train() loop warm-starts from a value function with positive SNR.
"""

import sys
import os
import argparse
import time

# Ensure project root is on sys.path regardless of CWD
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.optim as optim
from importlib import import_module
from tqdm import tqdm

# Load modules via importlib (digit-prefixed directories are not valid Python identifiers)
network_mod        = import_module('02_network.network')
self_play_mod      = import_module('04_training.self_play')
replay_buffer_mod  = import_module('04_training.replay_buffer')
loss_mod           = import_module('04_training.loss')
trainer_mod        = import_module('04_training.trainer')

UltimateTTTNetwork         = network_mod.UltimateTTTNetwork
play_random_vs_random_game = self_play_mod.play_random_vs_random_game
ReplayBuffer               = replay_buffer_mod.ReplayBuffer
TrainingBatch              = replay_buffer_mod.TrainingBatch
compute_total_loss         = loss_mod.compute_total_loss
save_checkpoint            = trainer_mod.save_checkpoint


def generate_random_games(num_games: int) -> ReplayBuffer:
    """Generate random-vs-random games and return a populated ReplayBuffer.

    Args:
        num_games: Number of complete games to generate.

    Returns:
        ReplayBuffer containing all positions from all games.
    """
    # Capacity: assume average game length of 80 moves, both players recorded.
    # 5000 games x 80 moves = 400 000 positions; use 500 000 to be safe.
    buffer = ReplayBuffer(capacity=max(num_games * 100, 500_000))

    t0 = time.time()
    total_positions = 0

    for i in range(num_games):
        record = play_random_vs_random_game()
        buffer.add_game(record)
        total_positions += record.game_length

        if (i + 1) % 100 == 0 or (i + 1) == num_games:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(
                f"  [{i+1}/{num_games}] games  |  {total_positions} positions  "
                f"|  {rate:.1f} games/s  |  {elapsed:.0f}s elapsed"
            )

    print(f"\nBuffer: {len(buffer)} positions from {num_games} games.")
    return buffer


def pretrain_value_head(
    network: torch.nn.Module,
    buffer: ReplayBuffer,
    num_batches: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    lambda_score: float = 1.0,
    lambda_ownership: float = 1.0,
) -> list:
    """Train the value head (only) on the random-play buffer.

    Policy and opponent-policy losses are suppressed (lambda_policy=0, lambda_opp=0).
    The score and ownership auxiliaries are included because they share the value-head
    trunk and provide richer per-position supervision.

    Args:
        network: Full UltimateTTTNetwork (all parameters accessible, but only
                 value-head-related parameters receive meaningful gradient).
        buffer: ReplayBuffer populated with random-play positions.
        num_batches: Number of gradient update steps.
        batch_size: Positions per batch.
        learning_rate: Adam learning rate.
        device: Compute device ('cpu', 'mps', 'cuda').
        lambda_score: Weight for score auxiliary loss.
        lambda_ownership: Weight for ownership auxiliary loss.

    Returns:
        List of per-batch metric dicts for logging.
    """
    optimizer = optim.Adam(network.parameters(), lr=learning_rate, weight_decay=1e-4)
    network.to(device)
    network.train()

    metrics_log = []

    for step in tqdm(range(num_batches), desc="Pretraining value head"):
        batch = buffer.sample(batch_size)

        # Move batch tensors to device
        states    = batch.state_tensors.to(device)
        pol_tgt   = batch.policy_targets.to(device)
        opp_tgt   = batch.opp_policy_targets.to(device)
        wdl_tgt   = batch.wdl_targets.to(device)
        val_tgt   = batch.value_targets.to(device)
        score_tgt = batch.score_targets.to(device)
        own_tgt   = batch.ownership_targets.to(device)
        masks     = batch.legal_masks.to(device)
        opp_masks = batch.opp_legal_masks.to(device)

        device_batch = TrainingBatch(
            state_tensors=states,
            policy_targets=pol_tgt,
            opp_policy_targets=opp_tgt,
            wdl_targets=wdl_tgt,
            value_targets=val_tgt,
            score_targets=score_tgt,
            ownership_targets=own_tgt,
            legal_masks=masks,
            opp_legal_masks=opp_masks,
        )

        output = network(states)

        breakdown = compute_total_loss(
            output,
            device_batch,
            lambda_policy=0.0,              # suppress policy loss entirely
            lambda_value=3.0,               # emphasise value (primary objective)
            lambda_score=lambda_score,
            lambda_ownership=lambda_ownership,
            lambda_opp=0.0,                 # suppress opponent-policy loss
        )

        optimizer.zero_grad()
        breakdown.total.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
        optimizer.step()

        metrics_log.append({
            'step': step,
            'loss_total': breakdown.total.item(),
            'loss_value': breakdown.value.item(),
            'loss_score': breakdown.score.item(),
            'loss_ownership': breakdown.ownership.item(),
        })

        if (step + 1) % 50 == 0:
            tqdm.write(
                f"  step {step+1:4d}  |  total={breakdown.total.item():.4f}  "
                f"value={breakdown.value.item():.4f}  "
                f"score={breakdown.score.item():.4f}  "
                f"ownership={breakdown.ownership.item():.4f}"
            )

    return metrics_log


def main():
    parser = argparse.ArgumentParser(description="Pretrain value head on random-play data.")
    parser.add_argument(
        '--allow-random-policy-bootstrap',
        action='store_true',
        help=(
            "Acknowledge that these are random-policy, uncalibrated targets. "
            "They must not be mixed into the strong-play replay buffer."
        ),
    )
    parser.add_argument('--games',      type=int,   default=5_000,
                        help="Number of random-vs-random games to generate (default: 5000).")
    parser.add_argument('--batches',    type=int,   default=500,
                        help="Gradient update steps for value pretraining (default: 500).")
    parser.add_argument('--batch-size', type=int,   default=256,
                        help="Positions per gradient step (default: 256).")
    parser.add_argument('--lr',         type=float, default=1e-3,
                        help="Adam learning rate (default: 0.001).")
    parser.add_argument('--channels',   type=int,   default=128,
                        help="Network channels (must match main training config).")
    parser.add_argument('--num-blocks', type=int,   default=8,
                        help="Number of residual blocks (must match main training config).")
    parser.add_argument('--device',     type=str,   default='cpu',
                        help="Compute device: cpu | mps | cuda (default: cpu).")
    parser.add_argument('--checkpoint', type=str,
                        default=os.path.join(PROJECT_ROOT, 'checkpoints', 'pretrain_value.pt'),
                        help="Output checkpoint path.")
    parser.add_argument('--seed',       type=int,   default=42)
    args = parser.parse_args()

    if not args.allow_random_policy_bootstrap:
        parser.error(
            "disabled by ADR-0001: random-game outcomes are not strong-play "
            "W/D/L targets; pass --allow-random-policy-bootstrap only for the "
            "legacy representation experiment"
        )

    torch.manual_seed(args.seed)

    print("=" * 60)
    print("  Ultimate TTT -- Value Head Pretraining")
    print("=" * 60)
    print(f"  Network:    {args.channels}ch x {args.num_blocks} blocks")
    print(f"  Games:      {args.games:,} random-vs-random")
    print(f"  Batches:    {args.batches} gradient steps  (batch size {args.batch_size})")
    print(f"  Device:     {args.device}")
    print(f"  Checkpoint: {args.checkpoint}")
    print("=" * 60)
    print()

    # Stage 0: generate random games
    print("Stage 0: Generating random-vs-random games...")
    buffer = generate_random_games(args.games)

    if len(buffer) < args.batch_size:
        print(f"ERROR: buffer has {len(buffer)} positions, need at least {args.batch_size}.")
        sys.exit(1)

    # Stage 1: pretrain value head
    print(f"\nStage 1: Pretraining value head ({args.batches} steps)...")
    network = UltimateTTTNetwork(channels=args.channels, num_blocks=args.num_blocks)
    n_params = sum(p.numel() for p in network.parameters())
    print(f"  Parameters: {n_params:,}")

    metrics = pretrain_value_head(
        network=network,
        buffer=buffer,
        num_batches=args.batches,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
    )

    # Save checkpoint to specified path directly
    os.makedirs(os.path.dirname(os.path.abspath(args.checkpoint)), exist_ok=True)
    network.to('cpu')
    torch.save({
        'network_state_dict': network.state_dict(),
        'iteration': 0,
        'elo': 0.0,
        'pretrain_games': args.games,
        'pretrain_batches': args.batches,
        'timestamp': time.time(),
    }, args.checkpoint)

    print(f"\nSaved pretrain checkpoint: {args.checkpoint}")

    # Report final losses
    if metrics:
        last = metrics[-1]
        print(f"\nFinal losses (step {last['step']+1}):")
        print(f"  value     = {last['loss_value']:.4f}   (target: < 0.5)")
        print(f"  score     = {last['loss_score']:.4f}")
        print(f"  ownership = {last['loss_ownership']:.4f}")
        print(f"  total     = {last['loss_total']:.4f}")

    print("\nTo use this checkpoint, add to TrainingConfig in train.py:")
    print(f"  pretrain_checkpoint='{args.checkpoint}'")
    print("\nDone.")


if __name__ == '__main__':
    main()
