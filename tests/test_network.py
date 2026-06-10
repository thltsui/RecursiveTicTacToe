"""Tests for Episodes 4-7 — network forward pass shapes."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import torch

from importlib import import_module
res_mod = import_module('02_network.residual_block')
policy_mod = import_module('02_network.policy_head')
value_mod = import_module('02_network.value_head')
network_mod = import_module('02_network.network')
board_mod = import_module('01_game.board')


class TestResidualBlock:
    def test_output_shape(self):
        block = res_mod.ResidualBlock(channels=64)
        x = torch.randn(2, 64, 9, 9)
        y = block(x)
        assert y.shape == (2, 64, 9, 9)

    def test_not_identity(self):
        block = res_mod.ResidualBlock(channels=64)
        x = torch.randn(2, 64, 9, 9)
        y = block(x)
        assert not torch.allclose(x, y)


class TestPolicyHead:
    def test_output_shapes(self):
        head = policy_mod.PolicyHead(in_channels=64)
        x = torch.randn(2, 64, 9, 9)
        p, opp = head(x)
        assert p.shape == (2, 81)
        assert opp.shape == (2, 81)


class TestApplyLegalMask:
    def test_probs_sum_to_one(self):
        logits = torch.randn(81)
        mask = torch.zeros(81)
        mask[:10] = 1.0
        probs = policy_mod.apply_legal_mask(logits, mask)
        assert abs(probs.sum().item() - 1.0) < 1e-5

    def test_illegal_moves_zero(self):
        logits = torch.randn(81)
        mask = torch.zeros(81)
        mask[:10] = 1.0
        probs = policy_mod.apply_legal_mask(logits, mask)
        assert probs[10:].sum().item() < 1e-6

    def test_empty_mask_raises(self):
        logits = torch.randn(81)
        mask = torch.zeros(81)
        with pytest.raises(ValueError):
            policy_mod.apply_legal_mask(logits, mask)


class TestValueHead:
    def test_output_shapes(self):
        head = value_mod.ValueHead(in_channels=64)
        x = torch.randn(2, 64, 9, 9)
        out = head(x)
        assert out.win_value.shape == (2, 1)
        assert out.score_margin.shape == (2, 1)
        assert out.ownership.shape == (2, 9)

    def test_value_ranges(self):
        head = value_mod.ValueHead(in_channels=64)
        x = torch.randn(4, 64, 9, 9)
        out = head(x)
        assert out.win_value.min() >= -1.0
        assert out.win_value.max() <= 1.0
        assert out.ownership.min() >= 0.0
        assert out.ownership.max() <= 1.0


class TestFullNetwork:
    def test_forward_shapes(self):
        net = network_mod.UltimateTTTNetwork(channels=32, num_blocks=2)
        x = torch.randn(4, 7, 9, 9)
        out = net(x)
        assert out.policy_logits.shape == (4, 81)
        assert out.opp_policy_logits.shape == (4, 81)
        assert out.win_value.shape == (4, 1)
        assert out.score_margin.shape == (4, 1)
        assert out.ownership.shape == (4, 9)

    def test_predict_no_batch(self):
        net = network_mod.UltimateTTTNetwork(channels=32, num_blocks=2)
        state = board_mod.create_initial_state()
        out = net.predict(state)
        assert out.policy_logits.shape == (81,)
        assert out.win_value.shape == (1,)
        assert out.ownership.shape == (9,)

    def test_gradient_flow(self):
        net = network_mod.UltimateTTTNetwork(channels=32, num_blocks=2)
        x = torch.randn(2, 7, 9, 9, requires_grad=True)
        out = net(x)
        loss = out.policy_logits.sum() + out.win_value.sum()
        loss.backward()
        assert x.grad is not None
