"""Tests for Episode 13 — loss functions."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import torch
import numpy as np

from importlib import import_module
loss_mod = import_module('04_training.loss')
network_mod = import_module('02_network.network')
replay_mod = import_module('04_training.replay_buffer')
trainer_mod = import_module('04_training.trainer')


class TestPolicyLoss:
    def test_perfect_prediction_near_zero(self):
        B = 4
        targets = torch.zeros(B, 81)
        targets[:, 0] = 1.0
        logits = torch.full((B, 81), -100.0)
        logits[:, 0] = 100.0
        masks = torch.ones(B, 81)
        loss = loss_mod.policy_loss(logits, targets, masks)
        assert loss.item() < 0.01

    def test_uniform_prediction_log_n(self):
        B = 4
        targets = torch.zeros(B, 81)
        targets[:, 0] = 1.0
        logits = torch.zeros(B, 81)
        masks = torch.ones(B, 81)
        loss = loss_mod.policy_loss(logits, targets, masks)
        expected = np.log(81)
        assert abs(loss.item() - expected) < 0.1


class TestValueLoss:
    def test_perfect_prediction_zero(self):
        preds = torch.ones(4, 1)
        targets = torch.ones(4, 1)
        loss = loss_mod.value_loss(preds, targets)
        assert loss.item() < 1e-6

    def test_worst_case(self):
        preds = torch.ones(4, 1)
        targets = -torch.ones(4, 1)
        loss = loss_mod.value_loss(preds, targets)
        assert loss.item() == pytest.approx(4.0, abs=0.01)


class TestWDLLoss:
    def test_confident_correct_predictions_are_near_zero(self):
        logits = torch.tensor([
            [20.0, -20.0, -20.0],
            [-20.0, 20.0, -20.0],
            [-20.0, -20.0, 20.0],
        ])
        targets = torch.tensor([0, 1, 2])
        assert loss_mod.wdl_loss(logits, targets).item() < 1e-6


class TestTotalLoss:
    def test_all_components_nonnegative(self):
        B = 4
        net = network_mod.UltimateTTTNetwork(channels=32, num_blocks=2)
        x = torch.randn(B, 7, 9, 9)
        out = net(x)

        batch = replay_mod.TrainingBatch(
            state_tensors=x,
            policy_targets=torch.softmax(torch.randn(B, 81), dim=-1),
            opp_policy_targets=torch.softmax(torch.randn(B, 81), dim=-1),
            wdl_targets=torch.ones(B, dtype=torch.long),
            value_targets=torch.zeros(B, 1),
            score_targets=torch.zeros(B, 1),
            ownership_targets=torch.rand(B, 9).round(),
            legal_masks=torch.ones(B, 81),
            opp_legal_masks=torch.ones(B, 81),
        )

        breakdown = loss_mod.compute_total_loss(out, batch)
        assert breakdown.total.item() > 0
        assert breakdown.policy.item() >= 0
        assert breakdown.value.item() >= 0
        assert breakdown.score.item() >= 0
        assert breakdown.ownership.item() >= 0
        assert breakdown.opp_policy.item() >= 0

    def test_backward_pass(self):
        B = 4
        net = network_mod.UltimateTTTNetwork(channels=32, num_blocks=2)
        x = torch.randn(B, 7, 9, 9)
        out = net(x)

        batch = replay_mod.TrainingBatch(
            state_tensors=x,
            policy_targets=torch.softmax(torch.randn(B, 81), dim=-1),
            opp_policy_targets=torch.softmax(torch.randn(B, 81), dim=-1),
            wdl_targets=torch.ones(B, dtype=torch.long),
            value_targets=torch.zeros(B, 1),
            score_targets=torch.zeros(B, 1),
            ownership_targets=torch.rand(B, 9).round(),
            legal_masks=torch.ones(B, 81),
            opp_legal_masks=torch.ones(B, 81),
        )

        breakdown = loss_mod.compute_total_loss(out, batch)
        breakdown.total.backward()
        # Check gradients exist
        for p in net.parameters():
            if p.requires_grad:
                assert p.grad is not None
                break

    def test_gradient_step_invalidates_old_temperature(self):
        batch_size = 2
        net = network_mod.UltimateTTTNetwork(channels=32, num_blocks=2)
        net.value_head.wdl_temperature.fill_(2.5)
        net.value_head.wdl_is_calibrated.fill_(True)
        batch = replay_mod.TrainingBatch(
            state_tensors=torch.randn(batch_size, 7, 9, 9),
            policy_targets=torch.softmax(torch.randn(batch_size, 81), dim=-1),
            opp_policy_targets=torch.softmax(torch.randn(batch_size, 81), dim=-1),
            wdl_targets=torch.ones(batch_size, dtype=torch.long),
            value_targets=torch.zeros(batch_size, 1),
            score_targets=torch.zeros(batch_size, 1),
            ownership_targets=torch.rand(batch_size, 9).round(),
            legal_masks=torch.ones(batch_size, 81),
            opp_legal_masks=torch.ones(batch_size, 81),
        )
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        config = trainer_mod.TrainingConfig(channels=32, num_blocks=2)

        trainer_mod.train_step(net, optimizer, batch, config, 'cpu')

        assert net.value_head.wdl_temperature.item() == pytest.approx(1.0)
        assert not net.value_head.wdl_is_calibrated.item()
        assert net.value_head.wdl_is_native.item()
