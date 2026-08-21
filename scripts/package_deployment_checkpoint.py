#!/usr/bin/env python3
"""Build a small inference-only checkpoint with release evidence attached."""

from __future__ import annotations

import argparse
import json
import os

import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strip training state and package a checkpoint for deployment."
    )
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--evaluation-report', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument(
        '--release-status',
        choices=('accepted', 'experimental_portfolio_trial'),
        required=True,
    )
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    if os.path.exists(args.output) and not args.overwrite:
        parser.error("output exists; choose another path or pass --overwrite")

    checkpoint = torch.load(
        args.checkpoint, weights_only=False, map_location='cpu'
    )
    with open(args.evaluation_report, encoding='utf-8') as handle:
        report = json.load(handle)
    required = {'network_state_dict', 'model_config', 'wdl_calibration'}
    missing = sorted(required - set(checkpoint))
    if missing:
        parser.error(f"checkpoint is missing deployment fields: {', '.join(missing)}")

    packaged = {
        key: value
        for key, value in checkpoint.items()
        if key not in {'optimizer_state_dict', 'scheduler_state_dict'}
    }
    packaged['release_status'] = args.release_status
    packaged['deployment_evaluation'] = {
        'parameters': report.get('parameters'),
        'mcts': report.get('mcts'),
        'arena': report.get('arena'),
        'calibration': report.get('calibration'),
    }

    destination = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    temporary = f"{destination}.tmp"
    torch.save(packaged, temporary)
    os.replace(temporary, destination)
    print(
        f"Packaged {destination}: {os.path.getsize(destination):,} bytes, "
        f"status={args.release_status}"
    )


if __name__ == '__main__':
    main()
