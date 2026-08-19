"""Deterministic release gates for a trained and calibrated candidate model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class AcceptanceThresholds:
    max_parameters: int
    hard_simulations: int
    max_hard_search_median_ms: float
    min_arena_games: int
    min_arena_score: float
    min_calibration_positions: int
    max_negative_log_likelihood: float
    max_brier_score: float
    max_classwise_ece: float

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AcceptanceThresholds":
        unknown = sorted(set(raw) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown acceptance threshold(s): {', '.join(unknown)}")
        missing = sorted(set(cls.__dataclass_fields__) - set(raw))
        if missing:
            raise ValueError(f"missing acceptance threshold(s): {', '.join(missing)}")
        thresholds = cls(**raw)
        if min(
            thresholds.max_parameters,
            thresholds.hard_simulations,
            thresholds.min_arena_games,
            thresholds.min_calibration_positions,
        ) <= 0:
            raise ValueError("count-based acceptance thresholds must be positive")
        if thresholds.max_hard_search_median_ms <= 0:
            raise ValueError("max_hard_search_median_ms must be positive")
        if not 0.0 <= thresholds.min_arena_score <= 1.0:
            raise ValueError("min_arena_score must be in [0, 1]")
        return thresholds


def evaluate_acceptance(
    report: Mapping[str, Any], thresholds: AcceptanceThresholds
) -> list[str]:
    """Return human-readable failures; an empty list means the model passes."""
    failures: list[str] = []

    parameters = int(report.get('parameters', -1))
    if parameters < 0:
        failures.append('missing parameters')
    elif parameters > thresholds.max_parameters:
        failures.append(
            f"parameters {parameters} exceed {thresholds.max_parameters}"
        )

    mcts = report.get('mcts', {}).get(str(thresholds.hard_simulations), {})
    hard_ms = mcts.get('median_ms')
    if hard_ms is None:
        failures.append(
            f"missing {thresholds.hard_simulations}-simulation MCTS median"
        )
    elif float(hard_ms) > thresholds.max_hard_search_median_ms:
        failures.append(
            f"hard search median {float(hard_ms):.1f}ms exceeds "
            f"{thresholds.max_hard_search_median_ms:.1f}ms"
        )

    arena = report.get('arena', {})
    games = int(arena.get('games', -1))
    score = arena.get('score')
    if games < thresholds.min_arena_games:
        failures.append(
            f"arena games {games} are below {thresholds.min_arena_games}"
        )
    if score is None:
        failures.append('missing arena score')
    elif float(score) < thresholds.min_arena_score:
        failures.append(
            f"arena score {float(score):.3f} is below {thresholds.min_arena_score:.3f}"
        )

    calibration = report.get('calibration', {})
    if calibration.get('is_calibrated') is not True:
        failures.append('candidate checkpoint is not marked W/D/L calibrated')
    if calibration.get('evaluation_is_independent') is not True:
        failures.append(
            'calibration gate must use a buffer different from temperature fitting'
        )
    positions = int(calibration.get('positions', -1))
    if positions < thresholds.min_calibration_positions:
        failures.append(
            f"calibration positions {positions} are below "
            f"{thresholds.min_calibration_positions}"
        )

    scalar_gates = (
        ('negative_log_likelihood', thresholds.max_negative_log_likelihood),
        ('brier_score', thresholds.max_brier_score),
    )
    for name, maximum in scalar_gates:
        value = calibration.get(name)
        if value is None:
            failures.append(f"missing calibration {name}")
        elif float(value) > maximum:
            failures.append(
                f"calibration {name} {float(value):.4f} exceeds {maximum:.4f}"
            )

    ece = calibration.get('classwise_ece', {})
    for outcome in ('win', 'draw', 'loss'):
        value = ece.get(outcome)
        if value is None:
            failures.append(f"missing calibration classwise_ece.{outcome}")
        elif float(value) > thresholds.max_classwise_ece:
            failures.append(
                f"calibration ECE {outcome}={float(value):.4f} exceeds "
                f"{thresholds.max_classwise_ece:.4f}"
            )

    return failures
