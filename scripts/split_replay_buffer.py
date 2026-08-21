#!/usr/bin/env python3
"""Create deterministic, disjoint calibration/evaluation replay buffers."""

from __future__ import annotations

import argparse
import os
import random

import torch


def _save_subset(path: str, source: str, records: list[dict], seed: int) -> None:
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    temporary = f"{destination}.tmp"
    torch.save(
        {
            'capacity': len(records),
            'position': 0,
            'records': records,
            'split_metadata': {
                'source': os.path.abspath(source),
                'seed': seed,
                'positions': len(records),
            },
        },
        temporary,
    )
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Split records not used by a candidate into disjoint calibration "
            "and evaluation buffers."
        )
    )
    parser.add_argument('--source', required=True)
    parser.add_argument('--start-index', required=True, type=int)
    parser.add_argument('--calibration-output', required=True)
    parser.add_argument('--evaluation-output', required=True)
    parser.add_argument('--calibration-positions', required=True, type=int)
    parser.add_argument(
        '--evaluation-positions',
        type=int,
        default=0,
        help="Positions for evaluation; zero uses every remaining record.",
    )
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.start_index < 0:
        parser.error("start-index must be non-negative")
    if args.calibration_positions <= 0 or args.evaluation_positions < 0:
        parser.error("split sizes must be positive (evaluation may be zero)")
    if os.path.abspath(args.calibration_output) == os.path.abspath(
        args.evaluation_output
    ):
        parser.error("calibration and evaluation outputs must differ")

    source = torch.load(args.source, weights_only=False, map_location='cpu')
    records = list(source.get('records', []))
    if args.start_index >= len(records):
        parser.error(
            f"start-index {args.start_index} is outside {len(records)} records"
        )

    unseen = records[args.start_index:]
    random.Random(args.seed).shuffle(unseen)
    evaluation_positions = args.evaluation_positions or (
        len(unseen) - args.calibration_positions
    )
    required = args.calibration_positions + evaluation_positions
    if required > len(unseen):
        parser.error(
            f"requested {required} split records but only {len(unseen)} are available"
        )

    calibration = unseen[:args.calibration_positions]
    evaluation = unseen[
        args.calibration_positions:args.calibration_positions + evaluation_positions
    ]
    _save_subset(
        args.calibration_output, args.source, calibration, args.seed
    )
    _save_subset(args.evaluation_output, args.source, evaluation, args.seed)
    print(
        f"Saved {len(calibration)} calibration and {len(evaluation)} "
        f"evaluation positions from {len(unseen)} unseen records."
    )


if __name__ == '__main__':
    main()
