#!/usr/bin/env python3
"""Measure architecture and MCTS latency without updating model weights."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from importlib import import_module

import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

trainer_mod = import_module('04_training.trainer')
model_factory_mod = import_module('02_network.model_factory')
board_mod = import_module('01_game.board')
search_mod = import_module('03_mcts.search')


def _synchronize(device: str) -> None:
    if device == 'mps' and torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.synchronize()


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return ordered[index]


def _load_model(args):
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, weights_only=False, map_location='cpu')
        return model_factory_mod.create_network_from_checkpoint(checkpoint)
    training_config = trainer_mod.load_training_config(args.config)
    model_config = training_config.model_config()
    return model_factory_mod.create_network(model_config), model_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark inference and search only; this script never trains."
    )
    parser.add_argument(
        '--config',
        default=os.path.join(PROJECT_ROOT, 'configs', 'training', 'lite_transformer.json'),
    )
    parser.add_argument('--checkpoint')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--warmup-runs', type=int, default=20)
    parser.add_argument('--forward-runs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--mcts-simulations', default='100,200,400')
    parser.add_argument('--mcts-repeats', type=int, default=3)
    args = parser.parse_args()

    if min(args.warmup_runs, args.forward_runs, args.batch_size, args.mcts_repeats) <= 0:
        parser.error("run counts, repeats, and batch size must be positive")
    simulations = [int(value) for value in args.mcts_simulations.split(',') if value]
    if not simulations or min(simulations) <= 0:
        parser.error("mcts-simulations must contain positive integers")

    network, model_config = _load_model(args)
    network = network.to(args.device).eval()
    sample = torch.zeros(args.batch_size, 7, 9, 9, device=args.device)

    with torch.inference_mode():
        for _ in range(args.warmup_runs):
            network(sample)
        _synchronize(args.device)

        forward_ms = []
        for _ in range(args.forward_runs):
            start = time.perf_counter()
            network(sample)
            _synchronize(args.device)
            forward_ms.append((time.perf_counter() - start) * 1000.0)

    mcts_results: dict[str, dict[str, float]] = {}
    state = board_mod.create_initial_state()
    for simulation_count in simulations:
        samples = []
        for _ in range(args.mcts_repeats):
            start = time.perf_counter()
            search_mod.run_mcts(
                state,
                network,
                num_simulations=simulation_count,
                dirichlet_epsilon=0.0,
                device=args.device,
            )
            _synchronize(args.device)
            samples.append((time.perf_counter() - start) * 1000.0)
        mcts_results[str(simulation_count)] = {
            'median_ms': statistics.median(samples),
            'min_ms': min(samples),
            'max_ms': max(samples),
        }

    parameters = sum(parameter.numel() for parameter in network.parameters())
    report = {
        'model_config': model_config.to_dict(),
        'device': args.device,
        'parameters': parameters,
        'parameter_bytes_fp32': parameters * 4,
        'forward': {
            'batch_size': args.batch_size,
            'runs': args.forward_runs,
            'median_ms': statistics.median(forward_ms),
            'p95_ms': _percentile(forward_ms, 0.95),
            'mean_ms': statistics.fmean(forward_ms),
        },
        'mcts': mcts_results,
    }
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
