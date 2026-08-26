#!/usr/bin/env python3
"""Build the combined, non-training release report for a candidate model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
import time
from importlib import import_module

import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

model_factory_mod = import_module('02_network.model_factory')
replay_mod = import_module('04_training.replay_buffer')
calibration_mod = import_module('06_evaluation.calibration')
arena_mod = import_module('06_evaluation.arena')
board_mod = import_module('01_game.board')
search_mod = import_module('03_mcts.search')


def _load(path: str, device: str):
    checkpoint = torch.load(path, weights_only=False, map_location='cpu')
    network, config = model_factory_mod.create_network_from_checkpoint(checkpoint)
    return network.to(device).eval(), config, checkpoint


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _sync(device: str) -> None:
    if device == 'mps' and torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.synchronize()


def _calibration_report(network, records, batch_size: int, device: str) -> dict:
    probabilities = []
    targets = []
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            chunk = records[start:start + batch_size]
            states = torch.stack([record.state_tensor for record in chunk]).to(device)
            output = network(states)
            probabilities.append(output.wdl_probs.detach().cpu())
            targets.extend(record.wdl_target for record in chunk)
    probs = torch.cat(probabilities)
    target_tensor = torch.tensor(targets, dtype=torch.long)
    return {
        'is_calibrated': bool(network.value_head.wdl_is_calibrated.item()),
        'positions': len(records),
        'negative_log_likelihood': float(
            torch.nn.functional.nll_loss(
                probs.clamp_min(1e-12).log(), target_tensor
            ).item()
        ),
        'brier_score': calibration_mod.wdl_brier_score(probs, target_tensor),
        'classwise_ece': calibration_mod.classwise_expected_calibration_error(
            probs, target_tensor
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a candidate without changing its weights."
    )
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--held-out-buffer', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--arena-games', type=int, default=100)
    parser.add_argument('--arena-simulations', type=int, default=200)
    parser.add_argument('--hard-simulations', type=int, default=400)
    parser.add_argument('--latency-repeats', type=int, default=3)
    parser.add_argument('--max-positions', type=int, default=50_000)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    counts = (
        args.arena_games,
        args.arena_simulations,
        args.hard_simulations,
        args.latency_repeats,
        args.max_positions,
        args.batch_size,
    )
    if min(counts) <= 0:
        parser.error("game, simulation, repeat, position, and batch counts must be positive")

    candidate, model_config, candidate_checkpoint = _load(
        args.candidate, args.device
    )
    baseline, _, _ = _load(args.baseline, args.device)
    buffer = replay_mod.ReplayBuffer.load_from_file(args.held_out_buffer)
    records = list(buffer.records)
    if not records:
        parser.error("held-out replay buffer is empty")
    random.Random(args.seed).shuffle(records)
    records = records[:args.max_positions]

    calibration = _calibration_report(
        candidate, records, args.batch_size, args.device
    )
    evaluation_fingerprint = _sha256_file(args.held_out_buffer)
    fitted_fingerprint = candidate_checkpoint.get('wdl_calibration', {}).get(
        'calibration_buffer_sha256'
    )
    calibration['evaluation_buffer_sha256'] = evaluation_fingerprint
    calibration['evaluation_is_independent'] = bool(
        fitted_fingerprint and fitted_fingerprint != evaluation_fingerprint
    )

    latency_samples = []
    initial_state = board_mod.create_initial_state()
    for _ in range(args.latency_repeats):
        start = time.perf_counter()
        search_mod.run_mcts(
            initial_state,
            candidate,
            num_simulations=args.hard_simulations,
            dirichlet_epsilon=0.0,
            device=args.device,
        )
        _sync(args.device)
        latency_samples.append((time.perf_counter() - start) * 1000.0)

    arena = arena_mod.run_arena(
        network_new=candidate,
        network_old=baseline,
        num_games=args.arena_games,
        num_simulations=args.arena_simulations,
        device=args.device,
    )
    arena_score = (arena.wins + 0.5 * arena.draws) / args.arena_games
    parameters = sum(parameter.numel() for parameter in candidate.parameters())
    report = {
        'candidate': os.path.abspath(args.candidate),
        'baseline': os.path.abspath(args.baseline),
        'model_config': model_config.to_dict(),
        'parameters': parameters,
        'mcts': {
            str(args.hard_simulations): {
                'median_ms': statistics.median(latency_samples),
                'min_ms': min(latency_samples),
                'max_ms': max(latency_samples),
            }
        },
        'arena': {
            'games': args.arena_games,
            'simulations': args.arena_simulations,
            'wins': arena.wins,
            'losses': arena.losses,
            'draws': arena.draws,
            'score': arena_score,
        },
        'calibration': calibration,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
