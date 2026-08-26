"""Tests for deterministic release acceptance gates."""

from importlib import import_module


acceptance_mod = import_module('06_evaluation.acceptance')


def _thresholds():
    return acceptance_mod.AcceptanceThresholds(
        max_parameters=500_000,
        hard_simulations=400,
        max_hard_search_median_ms=2500.0,
        min_arena_games=100,
        min_arena_score=0.5,
        min_calibration_positions=10_000,
        max_negative_log_likelihood=1.0,
        max_brier_score=0.6,
        max_classwise_ece=0.1,
    )


def _passing_report():
    return {
        'parameters': 350_000,
        'mcts': {'400': {'median_ms': 1800.0}},
        'arena': {'games': 100, 'score': 0.52},
        'calibration': {
            'is_calibrated': True,
            'evaluation_is_independent': True,
            'positions': 12_000,
            'negative_log_likelihood': 0.9,
            'brier_score': 0.5,
            'classwise_ece': {'win': 0.05, 'draw': 0.08, 'loss': 0.06},
        },
    }


def test_complete_report_passes():
    assert acceptance_mod.evaluate_acceptance(_passing_report(), _thresholds()) == []


def test_uncalibrated_slow_model_fails_with_reasons():
    report = _passing_report()
    report['parameters'] = 800_000
    report['mcts']['400']['median_ms'] = 3000.0
    report['calibration']['is_calibrated'] = False
    report['calibration']['evaluation_is_independent'] = False

    failures = acceptance_mod.evaluate_acceptance(report, _thresholds())

    assert any('parameters' in failure for failure in failures)
    assert any('hard search' in failure for failure in failures)
    assert any('not marked' in failure for failure in failures)
    assert any('different from temperature fitting' in failure for failure in failures)
