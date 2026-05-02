"""Episode 14 — The Training Loop — Putting It All Together

Concept taught: The AlphaZero training cycle. How self-play and training
interleave. Learning rate scheduling. Gradient clipping. Checkpointing.

# Training Diagnostics — What Healthy Training Looks Like
#
# | Loss          | At iter 0      | Healthy progress      | Red flags            |
# |---------------|----------------|-----------------------|----------------------|
# | L_policy      | ~4.4 (log 81)  | Steady decrease       | Plateau before i50   |
# | L_value       | ~0.5-1.0       | Slow, noisy decrease  | Wild oscillation     |
# | L_score       | ~0.3-0.5       | Correlated w/ L_value | Opposite to L_value  |
# | L_ownership   | ~0.6-0.7       | Fast early drop       | Not moving at all    |
# | L_opp_policy  | ~4.4           | Tracks L_policy       | Diverging            |
#
# Elo:
# - First 10 iters: fast 1000->1200. Then ~20-30/iter. Flat for 10+ = stuck.
#
# Warning signs:
# 1. L_policy increases after decrease -> reduce lr
# 2. L_value stuck at ~1.0 -> check target signs
# 3. Policy entropy hits 0 -> collapsed
# 4. Arena win rate always < 0.5 -> check buffer
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@dataclass
class TrainingConfig:
    """Configuration for the full training pipeline.

    All hyperparameters with justified defaults.
    """
    # Self-play
    games_per_iteration:   int   = 100
    num_simulations:       int   = 800
    temperature_threshold: int   = 30
    self_play_with_best:   bool  = True

    # Training
    batch_size:            int   = 256
    batches_per_iteration: int   = 100
    learning_rate:         float = 0.001
    weight_decay:          float = 1e-4
    grad_clip_norm:        float = 1.0
    lr_decay_every_n:      int   = 50
    lr_decay_gamma:        float = 0.5

    # Loss weights
    lambda_value:          float = 1.0
    lambda_score:          float = 0.5
    lambda_ownership:      float = 0.5
    lambda_opp:            float = 0.15

    # Evaluation
    arena_every_n:         int   = 10
    arena_games:           int   = 100
    win_rate_threshold:    float = 0.55

    # Buffer
    buffer_capacity:       int   = 500_000

    # Checkpointing
    checkpoint_dir:        str   = 'checkpoints/'
    checkpoint_every_n:    int   = 10          # save checkpoint every N iterations regardless of arena
    device:                str   = 'cpu'       # device for MCTS self-play
    train_device:          str   = ''          # device for gradient updates (default: same as device)

    # Network
    channels:              int   = 128
    num_blocks:            int   = 8

    # Reproducibility
    seed:                  int   = 42


def train(config: TrainingConfig) -> None:
    """Main training loop. Runs indefinitely until interrupted.

    Logs all metrics to stdout and saves loss curves as JSON.
    Saves checkpoint after every iteration with improved network.

    Args:
        config: Training configuration.
    """
    from importlib import import_module
    network_mod = import_module('02_network.network')
    self_play_mod = import_module('04_training.self_play')
    replay_buffer_mod = import_module('04_training.replay_buffer')
    loss_mod = import_module('04_training.loss')

    UltimateTTTNetwork = network_mod.UltimateTTTNetwork
    play_self_play_game = self_play_mod.play_self_play_game
    generate_mixed_batch = self_play_mod.generate_mixed_batch
    ReplayBuffer = replay_buffer_mod.ReplayBuffer

    # Set seed
    torch.manual_seed(config.seed)

    # Resolve train device
    train_device = config.train_device if config.train_device else config.device

    # Create network
    network = UltimateTTTNetwork(
        channels=config.channels, num_blocks=config.num_blocks
    ).to(config.device)

    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.lr_decay_every_n,
        gamma=config.lr_decay_gamma,
    )

    buffer = ReplayBuffer(capacity=config.buffer_capacity)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Try to load persisted buffer
    buffer_path = os.path.join(config.checkpoint_dir, 'replay_buffer.pt')
    if os.path.exists(buffer_path):
        try:
            buffer = ReplayBuffer.load_from_file(buffer_path)
            print(f"  Loaded replay buffer: {len(buffer)} positions")
        except Exception as e:
            print(f"  Failed to load buffer: {e} — starting with empty buffer")

    metrics_log: list[dict] = []
    total_games = 0
    best_network_state = clone_state_dict(network.state_dict())
    last_arena_win_rate = 0.0
    best_random_win_rate = 0.0   # track best win rate vs random opponent
    best_checkpoint_path = os.path.join(config.checkpoint_dir, 'best_model.pt')

    # Auto-resume from latest checkpoint
    import glob
    checkpoints = sorted(glob.glob(os.path.join(config.checkpoint_dir, '*.pt')))
    iteration = 0
    if checkpoints:
        latest_cp = checkpoints[-1]
        try:
            meta = load_checkpoint(latest_cp, network, optimizer)
            iteration = meta['iteration']
            best_network_state = clone_state_dict(network.state_dict())
            last_arena_win_rate = float(meta.get('elo', 0.0))
            # Load existing metrics log if available
            metrics_path = os.path.join(config.checkpoint_dir, 'training_metrics.json')
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    metrics_log = json.load(f)
                total_games = metrics_log[-1].get('games_played', 0) if metrics_log else 0
            print(f"  Resumed from: {latest_cp} (iteration {iteration})")
        except Exception as e:
            print(f"  Failed to load checkpoint: {e} — starting fresh")
            iteration = 0

    # If an explicit best checkpoint exists, prefer it for self-play gating.
    if os.path.exists(best_checkpoint_path):
        best_net = UltimateTTTNetwork(
            channels=config.channels, num_blocks=config.num_blocks
        ).to(config.device)
        best_meta = load_checkpoint(best_checkpoint_path, best_net)
        best_network_state = clone_state_dict(best_net.state_dict())
        last_arena_win_rate = float(best_meta.get('elo', last_arena_win_rate))
        print(f"  Loaded best model from: {best_checkpoint_path}")

    print(f"Starting training with config:")
    print(f"  MCTS device: {config.device} | Train device: {train_device}")
    print(f"  Network: {config.channels}ch x {config.num_blocks}blocks")
    num_params = sum(p.numel() for p in network.parameters())
    print(f"  Parameters: {num_params:,}")
    print()
    try:
        while True:
            iteration += 1
            iter_start = time.time()

            # 1. SELF-PLAY — 100% pure self-play
            # We break the draw equilibrium via anti-draw value shaping and
            # increased Dirichlet noise / temperature exploration.
            network.eval()
            num_self = config.games_per_iteration
            print(f"[Iter {iteration}] Self-play: {num_self}x pure self-play")
            records = generate_mixed_batch(
                network=network,
                num_self_play=num_self,
                num_vs_random=0,
                num_simulations=config.num_simulations,
                temperature_threshold=config.temperature_threshold,
                device=config.device,
            )
            for record in records:
                buffer.add_game(record)
                total_games += 1

            print(f"  Buffer size: {len(buffer)}")

            # 2. TRAIN — move to train_device (e.g. MPS) for batched gradient updates
            if len(buffer) < config.batch_size:
                print(f"  Buffer too small for training ({len(buffer)} < {config.batch_size})")
                continue

            # Fix: cap batches to prevent overfitting when buffer is small
            # Allow at most ~2 epochs per iteration over the buffer
            max_batches_by_buffer = max(1, (len(buffer) * 2) // config.batch_size)
            effective_batches = min(config.batches_per_iteration, max_batches_by_buffer)
            if effective_batches < config.batches_per_iteration:
                print(f"  Batches capped: {effective_batches} (buffer {len(buffer)} too small for {config.batches_per_iteration})")

            if train_device != config.device:
                network.to(train_device)
            network.train()
            total_loss_accum = 0.0
            component_accum = {k: 0.0 for k in
                               ['policy', 'value', 'score', 'ownership', 'opp_policy']}

            for step in tqdm(range(effective_batches), desc="Training"):
                batch = buffer.sample(config.batch_size)
                loss_breakdown = train_step(network, optimizer, batch, config, train_device)

                total_loss_accum += loss_breakdown.total.item()
                for k in component_accum:
                    component_accum[k] += getattr(loss_breakdown, k).item()

            # Move back to MCTS device for next self-play round
            if train_device != config.device:
                network.to(config.device)

            avg_factor = effective_batches
            metrics = {
                'iteration': iteration,
                'games_played': total_games,
                'loss_total': total_loss_accum / avg_factor,
                'loss_policy': component_accum['policy'] / avg_factor,
                'loss_value': component_accum['value'] / avg_factor,
                'loss_score': component_accum['score'] / avg_factor,
                'loss_ownership': component_accum['ownership'] / avg_factor,
                'loss_opp_policy': component_accum['opp_policy'] / avg_factor,
                'buffer_size': len(buffer),
                'learning_rate': optimizer.param_groups[0]['lr'],
                'time_seconds': time.time() - iter_start,
            }

            # 3. EVALUATE vs random opponent (every N iterations)
            # We measure win rate against random — this is our true metric of improvement.
            # No more arena vs best_model (which was losing to random!).
            if iteration % config.arena_every_n == 0:
                try:
                    import random as rng
                    board_mod = import_module('01_game.board')
                    rules_mod = import_module('01_game.rules')
                    search_mod = import_module('03_mcts.search')

                    network.eval()
                    num_games = min(config.arena_games, 50)  # cap at 50 to keep it fast
                    wins, loses, draws = 0, 0, 0
                    for g in range(num_games):
                        state = board_mod.create_initial_state()
                        while not state.is_terminal:
                            if state.current_player == 1:
                                root = search_mod.run_mcts(
                                    state, network, num_simulations=config.num_simulations,
                                    dirichlet_epsilon=0.0, device=config.device)
                                move = search_mod.select_move(root, temperature=0.0)
                            else:
                                legal = rules_mod.get_legal_moves(state)
                                move = rng.choice(legal)
                            state = rules_mod.apply_move(state, move)
                        if state.winner == 1:
                            wins += 1
                        elif state.winner == -1:
                            loses += 1
                        else:
                            draws += 1

                    win_rate = wins / (wins + loses + draws) if (wins + loses + draws) > 0 else 0.0
                    metrics['arena_win_rate'] = win_rate
                    metrics['arena_wins'] = wins
                    metrics['arena_losses'] = loses
                    metrics['arena_draws'] = draws
                    last_arena_win_rate = win_rate
                    print(f"  Vs Random: {wins}W/{loses}L/{draws}D "
                          f"(win_rate={win_rate:.3f})")

                    # 4. CHECKPOINT if new network beats best_model.pt at 55%+ vs random
                    # Load the stored best_model.pt's win rate vs random as baseline
                    if win_rate > 0.55 and win_rate > best_random_win_rate:
                        best_network_state = clone_state_dict(network.state_dict())
                        best_random_win_rate = win_rate
                        path = save_checkpoint(network, optimizer, iteration,
                                               win_rate, config.checkpoint_dir)
                        save_checkpoint(network, optimizer, iteration,
                                        win_rate, config.checkpoint_dir,
                                        filename_override='best_model.pt')
                        print(f"  New best network (wr vs random={win_rate:.3f})! Saved to {path}")
                except Exception as e:
                    print(f"  Eval vs random skipped: {e}")
                    import traceback; traceback.print_exc()

            # Periodic checkpoint save (regardless of arena)
            if iteration % config.checkpoint_every_n == 0:
                path = save_checkpoint(network, optimizer, iteration,
                                       last_arena_win_rate, config.checkpoint_dir)
                print(f"  Periodic checkpoint saved to {path}")
                # Persist replay buffer alongside checkpoint
                buffer.save_to_file(buffer_path)
                print(f"  Buffer saved: {len(buffer)} positions")

            metrics_log.append(metrics)
            print(f"[Iter {iteration}] Loss: {metrics['loss_total']:.4f} "
                  f"(P:{metrics['loss_policy']:.3f} V:{metrics['loss_value']:.3f}) "
                  f"Time: {metrics['time_seconds']:.1f}s")

            # Save metrics
            metrics_path = os.path.join(config.checkpoint_dir, 'training_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics_log, f, indent=2)

            scheduler.step()

    except KeyboardInterrupt:
        print(f"\nTraining interrupted at iteration {iteration}")
        path = save_checkpoint(network, optimizer, iteration, last_arena_win_rate, config.checkpoint_dir)
        print(f"Final checkpoint saved to {path}")


def train_step(
    network: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: 'TrainingBatch',
    config: TrainingConfig,
    device: str,
) -> 'LossBreakdown':
    """Single gradient update step.

    Steps:
        1. Move batch to device.
        2. Forward pass.
        3. Compute loss.
        4. Backward pass.
        5. Clip gradients.
        6. Optimizer step.

    Args:
        network: The neural network.
        optimizer: Optimizer instance.
        batch: Training batch.
        config: Training config (for loss weights).
        device: Compute device.

    Returns:
        LossBreakdown with all components (detached, for logging).
    """
    from importlib import import_module
    loss_mod = import_module('04_training.loss')

    # 1. Move batch to device
    states = batch.state_tensors.to(device)       # (B, 7, 9, 9)
    policy_tgt = batch.policy_targets.to(device)   # (B, 81)
    opp_tgt = batch.opp_policy_targets.to(device)  # (B, 81)
    value_tgt = batch.value_targets.to(device)     # (B, 1)
    score_tgt = batch.score_targets.to(device)     # (B, 1)
    own_tgt = batch.ownership_targets.to(device)   # (B, 9)
    masks = batch.legal_masks.to(device)           # (B, 81)
    opp_masks = batch.opp_legal_masks.to(device)   # (B, 81)

    # Rebuild batch on device
    from .replay_buffer import TrainingBatch
    device_batch = TrainingBatch(
        state_tensors=states,
        policy_targets=policy_tgt,
        opp_policy_targets=opp_tgt,
        value_targets=value_tgt,
        score_targets=score_tgt,
        ownership_targets=own_tgt,
        legal_masks=masks,
        opp_legal_masks=opp_masks,
    )

    # 2. Forward pass
    output = network(states)

    # 3. Compute loss
    breakdown = loss_mod.compute_total_loss(
        output, device_batch,
        lambda_value=config.lambda_value,
        lambda_score=config.lambda_score,
        lambda_ownership=config.lambda_ownership,
        lambda_opp=config.lambda_opp,
    )

    # 4. Backward pass
    optimizer.zero_grad()
    breakdown.total.backward()

    # 5. Clip gradients
    torch.nn.utils.clip_grad_norm_(network.parameters(), config.grad_clip_norm)

    # 6. Optimizer step
    optimizer.step()

    return LossBreakdown(
        total=breakdown.total.detach(),
        policy=breakdown.policy,
        value=breakdown.value,
        score=breakdown.score,
        ownership=breakdown.ownership,
        opp_policy=breakdown.opp_policy,
    )


# Re-export for convenience
from .loss import LossBreakdown


def clone_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return a deep-cloned state dict safe for long-lived snapshots."""
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def save_checkpoint(
    network: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    elo: float,
    checkpoint_dir: str,
    filename_override: str = "",
) -> str:
    """Save network and optimizer state.

    Filename format: checkpoint_iter{iteration:05d}_elo{elo:.0f}.pt

    Args:
        network: Network to save.
        optimizer: Optimizer to save.
        iteration: Current iteration number.
        elo: Current Elo rating.
        checkpoint_dir: Directory to save to.

    Returns:
        Path to saved checkpoint file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = filename_override if filename_override else f"checkpoint_iter{iteration:05d}_elo{elo:.0f}.pt"
    path = os.path.join(checkpoint_dir, filename)

    torch.save({
        'network_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
        'elo': elo,
        'timestamp': time.time(),
    }, path)

    return path


def load_checkpoint(
    path: str,
    network: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    """Load network (and optionally optimizer) from checkpoint.

    Args:
        path: Path to checkpoint file.
        network: Network to load weights into.
        optimizer: Optional optimizer to load state into.

    Returns:
        Dict with 'iteration', 'elo', 'timestamp'.
    """
    checkpoint = torch.load(path, weights_only=False)
    network.load_state_dict(checkpoint['network_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return {
        'iteration': checkpoint.get('iteration', 0),
        'elo': checkpoint.get('elo', 1000.0),
        'timestamp': checkpoint.get('timestamp', 0),
    }


if __name__ == "__main__":
    print("=== Episode 14: Trainer ===\n")

    from importlib import import_module
    network_mod = import_module('02_network.network')
    from .replay_buffer import ReplayBuffer, TrainingBatch
    from .self_play import MoveRecord, GameRecord

    # Create small network
    net = network_mod.UltimateTTTNetwork(channels=32, num_blocks=2)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    config = TrainingConfig(batch_size=4, channels=32, num_blocks=2)

    # Create fake batch
    B = 4
    batch = TrainingBatch(
        state_tensors=torch.randn(B, 7, 9, 9),
        policy_targets=torch.softmax(torch.randn(B, 81), dim=-1),
        opp_policy_targets=torch.softmax(torch.randn(B, 81), dim=-1),
        value_targets=torch.zeros(B, 1),
        score_targets=torch.zeros(B, 1),
        ownership_targets=torch.rand(B, 9).round(),
        legal_masks=torch.ones(B, 81),
        opp_legal_masks=torch.ones(B, 81),
    )

    # Run a training step
    net.train()
    breakdown = train_step(net, optimizer, batch, config, 'cpu')
    print(f"Train step loss: {breakdown.total.item():.4f}")

    # Test checkpoint save/load
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_checkpoint(net, optimizer, 1, 1000.0, tmpdir)
        print(f"Saved checkpoint to: {path}")

        net2 = network_mod.UltimateTTTNetwork(channels=32, num_blocks=2)
        meta = load_checkpoint(path, net2)
        print(f"Loaded: iteration={meta['iteration']}, elo={meta['elo']}")

        # Verify weights match
        for (n1, p1), (n2, p2) in zip(net.named_parameters(), net2.named_parameters()):
            assert torch.allclose(p1, p2), f"Mismatch in {n1}"
        print("Checkpoint roundtrip: PASSED")

    print("\n=== Episode 14 PASSED ===")
