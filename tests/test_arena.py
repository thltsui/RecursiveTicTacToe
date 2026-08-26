"""Tests for Episode 19 — arena terminates and returns valid result."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest

from importlib import import_module
network_mod = import_module('02_network.network')
arena_mod = import_module('06_evaluation.arena')


class TestArena:
    def test_arena_terminates(self):
        net1 = network_mod.UltimateTTTNetwork(channels=32, num_blocks=2)
        net2 = network_mod.UltimateTTTNetwork(channels=32, num_blocks=2)
        result = arena_mod.run_arena(net1, net2, num_games=2,
                                      num_simulations=10, device='cpu')
        assert result.wins + result.losses + result.draws == 2

    def test_arena_result_valid(self):
        net1 = network_mod.UltimateTTTNetwork(channels=32, num_blocks=2)
        net2 = network_mod.UltimateTTTNetwork(channels=32, num_blocks=2)
        result = arena_mod.run_arena(net1, net2, num_games=2,
                                      num_simulations=10, device='cpu')
        assert 0.0 <= result.win_rate <= 1.0
        assert 0.0 <= result.score_rate <= 1.0
        assert isinstance(result.new_is_better, bool)

    def test_draw_is_half_a_point_for_promotion(self):
        result = arena_mod.ArenaResult(
            wins=4,
            losses=3,
            draws=3,
            win_rate=0.4,
            win_rate_threshold=0.54,
        )
        assert result.score_rate == pytest.approx(0.55)
        assert result.new_is_better

    def test_single_game(self):
        net1 = network_mod.UltimateTTTNetwork(channels=32, num_blocks=2)
        net2 = network_mod.UltimateTTTNetwork(channels=32, num_blocks=2)
        result = arena_mod.play_single_game(net1, net2, num_simulations=10,
                                             device='cpu')
        assert result in (1, -1, 0)
