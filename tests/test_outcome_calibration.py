"""Tests for zero-sum outcome semantics and W/D/L calibration helpers."""

import os
import sys
from importlib import import_module

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

outcome_mod = import_module('01_game.outcome')
calibration_mod = import_module('06_evaluation.calibration')
self_play_mod = import_module('04_training.self_play')
replay_mod = import_module('04_training.replay_buffer')


@pytest.mark.parametrize(
    "winner,perspective,expected_class,expected_value",
    [
        (1, 1, outcome_mod.WDL_WIN, 1.0),
        (1, -1, outcome_mod.WDL_LOSS, -1.0),
        (-1, -1, outcome_mod.WDL_WIN, 1.0),
        (-1, 1, outcome_mod.WDL_LOSS, -1.0),
        (0, 1, outcome_mod.WDL_DRAW, 0.0),
        (0, -1, outcome_mod.WDL_DRAW, 0.0),
    ],
)
def test_outcome_is_perspective_correct(winner, perspective, expected_class, expected_value):
    assert outcome_mod.outcome_class(winner, perspective) == expected_class
    assert outcome_mod.outcome_value(winner, perspective) == expected_value


def test_legacy_negative_draw_is_normalized():
    klass = outcome_mod.legacy_value_to_class(-0.5)
    assert klass == outcome_mod.WDL_DRAW
    assert outcome_mod.class_value(klass) == 0.0


def test_legacy_replay_buffer_normalizes_negative_draw(tmp_path):
    record = {
        'state_tensor': torch.zeros(7, 9, 9),
        'policy_target': torch.zeros(81),
        'opp_policy_target': torch.zeros(81),
        'opp_legal_mask': torch.zeros(81),
        'legal_mask': torch.ones(81),
        'current_player': 1,
        'value_target': -0.5,
        'score_target': 0.0,
        'ownership_target': torch.zeros(9),
    }
    path = tmp_path / 'legacy_replay.pt'
    torch.save({'capacity': 10, 'position': 1, 'records': [record]}, path)

    loaded = replay_mod.ReplayBuffer.load_from_file(str(path))

    assert loaded.records[0].wdl_target == outcome_mod.WDL_DRAW
    assert loaded.records[0].value_target == 0.0


def test_finalize_targets_never_makes_draw_negative():
    def record(player):
        return self_play_mod.MoveRecord(
            state_tensor=torch.zeros(7, 9, 9),
            policy_target=torch.zeros(81),
            opp_policy_target=torch.zeros(81),
            opp_legal_mask=torch.zeros(81),
            legal_mask=torch.ones(81),
            current_player=player,
        )

    records = [record(1), record(-1)]
    self_play_mod._finalize_move_targets(records, 0, torch.zeros(9).numpy())
    assert [rec.wdl_target for rec in records] == [outcome_mod.WDL_DRAW] * 2
    assert [rec.value_target for rec in records] == [0.0, 0.0]


def test_perfect_probabilities_have_zero_calibration_error():
    probs = torch.eye(3)
    targets = torch.tensor([0, 1, 2])
    assert calibration_mod.wdl_brier_score(probs, targets) == pytest.approx(0.0)
    assert calibration_mod.classwise_expected_calibration_error(probs, targets) == {
        'win': pytest.approx(0.0),
        'draw': pytest.approx(0.0),
        'loss': pytest.approx(0.0),
    }


def test_temperature_fit_does_not_worsen_held_out_nll():
    logits = torch.tensor([
        [6.0, 0.0, 0.0],
        [6.0, 0.0, 0.0],
        [0.0, 6.0, 0.0],
        [0.0, 6.0, 0.0],
        [0.0, 0.0, 6.0],
        [0.0, 0.0, 6.0],
    ])
    targets = torch.tensor([0, 1, 1, 2, 2, 0])
    before = torch.nn.functional.cross_entropy(logits, targets)
    temperature = calibration_mod.fit_temperature(logits, targets)
    after = torch.nn.functional.cross_entropy(logits / temperature, targets)

    assert 0.05 <= temperature <= 20.0
    assert after.item() <= before.item() + 1e-6


def test_random_prefix_is_diversity_only_and_nonterminal():
    state = self_play_mod.sample_random_prefix_state(4, 4)
    assert state.move_count == 4
    assert not state.is_terminal
