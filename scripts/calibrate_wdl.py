#!/usr/bin/env python3
"""Temperature-scale a native W/D/L checkpoint on a held-out replay buffer.

This is post-hoc calibration, not network training.  The input replay buffer
must be held out from gradient updates and representative of deployment states.

Example (run explicitly after training):
    python scripts/calibrate_wdl.py \
        --checkpoint checkpoints/native_wdl.pt \
        --held-out-buffer checkpoints/calibration_buffer.pt \
        --output checkpoints/native_wdl_calibrated.pt
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from importlib import import_module

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

network_mod = import_module('02_network.network')
replay_mod = import_module('04_training.replay_buffer')
calibration_mod = import_module('06_evaluation.calibration')


def _is_transformer(state_dict: dict) -> bool:
    return 'pos_embed' in state_dict or any('patch_embed' in key for key in state_dict)


def _load_network(checkpoint: dict):
    state_dict = checkpoint['network_state_dict']
    if _is_transformer(state_dict):
        transformer_mod = import_module('transformer.transformer_network')
        channels = state_dict['pos_embed'].shape[-1]
        num_blocks = sum(
            key.endswith('.self_attn.in_proj_weight')
            and 'transformer.layers.' in key
            for key in state_dict
        ) or 4
        network = transformer_mod.TransformerTTTNetwork(channels, num_blocks)
    else:
        input_weights = [
            value for key, value in state_dict.items() if key.endswith('input_conv.weight')
        ]
        if not input_weights:
            raise ValueError("cannot infer CNN channels from checkpoint")
        channels = input_weights[0].shape[0]
        num_blocks = sum(
            key.endswith('.conv1.weight') and 'trunk.' in key for key in state_dict
        )
        network = network_mod.UltimateTTTNetwork(channels, num_blocks)

    network.load_state_dict(state_dict)
    if not bool(network.value_head.wdl_is_native.item()):
        raise ValueError(
            "legacy scalar checkpoints cannot produce meaningful draw "
            "probabilities; train a native three-output W/D/L head first"
        )
    network.eval()
    return network


def _collect_logits(network, records, batch_size: int):
    logits = []
    targets = []
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            chunk = records[start:start + batch_size]
            states = torch.stack([record.state_tensor for record in chunk])
            output = network(states)
            logits.append(output.wdl_logits.cpu())
            targets.extend(record.wdl_target for record in chunk)
    return torch.cat(logits), torch.tensor(targets, dtype=torch.long)


def _metrics(probs: torch.Tensor, targets: torch.Tensor) -> dict:
    return {
        'negative_log_likelihood': float(
            torch.nn.functional.nll_loss(probs.clamp_min(1e-12).log(), targets).item()
        ),
        'brier_score': calibration_mod.wdl_brier_score(probs, targets),
        'classwise_ece': calibration_mod.classwise_expected_calibration_error(
            probs, targets
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Temperature-scale W/D/L logits on a held-out replay buffer."
    )
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--held-out-buffer', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-positions', type=int, default=50_000)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    if os.path.exists(args.output) and not args.overwrite:
        parser.error("output exists; choose a new path or pass --overwrite")
    if args.max_positions <= 0 or args.batch_size <= 0:
        parser.error("max-positions and batch-size must be positive")

    checkpoint = torch.load(args.checkpoint, weights_only=False, map_location='cpu')
    network = _load_network(checkpoint)
    buffer = replay_mod.ReplayBuffer.load_from_file(args.held_out_buffer)
    records = list(buffer.records)
    if not records:
        parser.error("held-out replay buffer is empty")

    random.Random(args.seed).shuffle(records)
    records = records[:args.max_positions]
    logits, targets = _collect_logits(network, records, args.batch_size)

    before_probs = torch.softmax(logits, dim=-1)
    temperature = calibration_mod.fit_temperature(logits, targets)
    after_probs = torch.softmax(logits / temperature, dim=-1)
    calibration_mod.set_model_temperature(network, temperature)

    report = {
        'positions': len(records),
        'temperature': temperature,
        'before': _metrics(before_probs, targets),
        'after': _metrics(after_probs, targets),
    }
    checkpoint['network_state_dict'] = network.state_dict()
    checkpoint['wdl_calibration'] = report
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(checkpoint, args.output)

    print(json.dumps(report, indent=2))
    print(f"Saved calibrated checkpoint: {args.output}")


if __name__ == '__main__':
    main()
