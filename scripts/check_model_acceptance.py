#!/usr/bin/env python3
"""Apply versioned release thresholds to a combined evaluation report."""

from __future__ import annotations

import argparse
import json
import os
import sys
from importlib import import_module


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
acceptance_mod = import_module('06_evaluation.acceptance')


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check latency, strength, and W/D/L calibration release gates."
    )
    parser.add_argument('--report', required=True)
    parser.add_argument(
        '--thresholds',
        default=os.path.join(
            PROJECT_ROOT,
            'configs',
            'evaluation',
            'lite_transformer_acceptance.json',
        ),
    )
    args = parser.parse_args()

    with open(args.report, encoding='utf-8') as handle:
        report = json.load(handle)
    with open(args.thresholds, encoding='utf-8') as handle:
        thresholds = acceptance_mod.AcceptanceThresholds.from_dict(json.load(handle))

    failures = acceptance_mod.evaluate_acceptance(report, thresholds)
    result = {'passed': not failures, 'failures': failures}
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
