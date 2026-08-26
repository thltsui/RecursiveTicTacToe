"""Tests for JSON architecture config and checkpoint metadata."""

from __future__ import annotations

import os
import sys
from importlib import import_module

import pytest
import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

trainer_mod = import_module('04_training.trainer')
model_factory_mod = import_module('02_network.model_factory')


def test_lite_transformer_json_is_complete_and_small():
    path = os.path.join(
        PROJECT_ROOT, 'configs', 'training', 'lite_transformer.json'
    )
    config = trainer_mod.load_training_config(path)
    network = model_factory_mod.create_network(config.model_config())

    assert config.architecture == 'transformer'
    assert config.channels == 96
    assert config.num_blocks == 3
    assert config.position_encoding == 'hierarchical'
    assert config.self_play_batch_size == 8
    assert sum(parameter.numel() for parameter in network.parameters()) < 500_000


def test_training_config_round_trip_preserves_model_definition():
    path = os.path.join(
        PROJECT_ROOT, 'configs', 'training', 'lite_transformer.json'
    )
    original = trainer_mod.load_training_config(path)
    restored = trainer_mod.training_config_from_dict(original.to_dict())
    assert restored == original
    assert restored.model_config() == original.model_config()


def test_training_config_rejects_unknown_fields():
    raw = {
        'model': {
            'architecture': 'transformer',
            'channels': 64,
            'num_layers': 2,
            'num_heads': 4,
            'typo_width': 99,
        }
    }
    with pytest.raises(ValueError, match='typo_width'):
        trainer_mod.training_config_from_dict(raw)


def test_hierarchical_transformer_forward_shapes():
    config = model_factory_mod.ModelConfig(
        architecture='transformer',
        channels=32,
        num_blocks=2,
        num_heads=4,
        ffn_multiplier=2,
        position_encoding='hierarchical',
        value_channels=4,
        value_hidden_size=32,
        value_feature_size=16,
    )
    network = model_factory_mod.create_network(config)
    output = network(torch.randn(3, 7, 9, 9))

    assert output.policy_logits.shape == (3, 81)
    assert output.wdl_logits.shape == (3, 3)
    assert output.wdl_probs.shape == (3, 3)
    assert 'macro_pos_embed' in network.state_dict()
    assert 'cell_pos_embed' in network.state_dict()
    assert 'pos_embed' not in network.state_dict()


def test_checkpoint_stores_and_uses_authoritative_model_config(tmp_path):
    config = model_factory_mod.ModelConfig(
        architecture='transformer',
        channels=32,
        num_blocks=2,
        num_heads=4,
        ffn_multiplier=2,
        dropout=0.0,
        position_encoding='hierarchical',
        value_channels=4,
        value_hidden_size=32,
        value_feature_size=16,
    )
    network = model_factory_mod.create_network(config)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.5
    )
    for _ in range(7):
        optimizer.step()
        scheduler.step()
    path = trainer_mod.save_checkpoint(
        network,
        optimizer,
        iteration=7,
        elo=1234.0,
        checkpoint_dir=str(tmp_path),
        scheduler=scheduler,
    )
    checkpoint = torch.load(path, weights_only=False, map_location='cpu')
    restored, restored_config = model_factory_mod.create_network_from_checkpoint(
        checkpoint
    )

    assert checkpoint['checkpoint_format_version'] == 2
    assert checkpoint['scheduler_state_dict'] == scheduler.state_dict()
    assert restored_config == config
    assert restored.model_config == config.to_dict()
    for name, value in network.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])


def test_checkpoint_restores_scheduler_state(tmp_path):
    config = model_factory_mod.ModelConfig(
        architecture='cnn', channels=16, num_blocks=1
    )
    network = model_factory_mod.create_network(config)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.5
    )
    for _ in range(7):
        optimizer.step()
        scheduler.step()
    path = trainer_mod.save_checkpoint(
        network,
        optimizer,
        iteration=7,
        elo=0.0,
        checkpoint_dir=str(tmp_path),
        scheduler=scheduler,
    )

    restored_network = model_factory_mod.create_network(config)
    restored_optimizer = torch.optim.Adam(restored_network.parameters(), lr=1e-3)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(
        restored_optimizer, step_size=5, gamma=0.5
    )
    metadata = trainer_mod.load_checkpoint(
        path, restored_network, restored_optimizer, restored_scheduler
    )

    assert metadata['scheduler_state_restored'] is True
    assert metadata['scheduler_state_reconstructed'] is False
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    assert restored_optimizer.param_groups[0]['lr'] == pytest.approx(5e-4)


def test_legacy_checkpoint_reconstructs_step_scheduler(tmp_path):
    config = model_factory_mod.ModelConfig(
        architecture='cnn', channels=16, num_blocks=1
    )
    network = model_factory_mod.create_network(config)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    path = trainer_mod.save_checkpoint(
        network,
        optimizer,
        iteration=55,
        elo=0.0,
        checkpoint_dir=str(tmp_path),
    )

    restored_network = model_factory_mod.create_network(config)
    restored_optimizer = torch.optim.Adam(restored_network.parameters(), lr=1e-3)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(
        restored_optimizer, step_size=50, gamma=0.5
    )
    metadata = trainer_mod.load_checkpoint(
        path, restored_network, restored_optimizer, restored_scheduler
    )

    assert metadata['scheduler_state_restored'] is False
    assert metadata['scheduler_state_reconstructed'] is True
    assert restored_scheduler.last_epoch == 55
    assert restored_optimizer.param_groups[0]['lr'] == pytest.approx(5e-4)


def test_resume_archives_metrics_newer_than_checkpoint(tmp_path):
    metrics_path = tmp_path / 'training_metrics.json'
    metrics_path.write_text(
        '[\n'
        '  {"iteration": 14, "games_played": 140},\n'
        '  {"iteration": 15, "games_played": 150},\n'
        '  {"iteration": 16, "games_played": 160}\n'
        ']\n',
        encoding='utf-8',
    )

    durable, total_games, archive_path = trainer_mod.reconcile_metrics_log(
        metrics_path, checkpoint_iteration=15
    )

    assert [entry['iteration'] for entry in durable] == [14, 15]
    assert total_games == 150
    assert archive_path is not None
    assert os.path.exists(archive_path)
    with open(archive_path, encoding='utf-8') as handle:
        assert '"iteration": 16' in handle.read()
    assert '"iteration": 16' not in metrics_path.read_text(encoding='utf-8')


def test_legacy_checkpoint_config_is_inferred_only_when_metadata_is_absent():
    config = model_factory_mod.ModelConfig(
        architecture='cnn', channels=16, num_blocks=1
    )
    network = model_factory_mod.create_network(config)
    legacy = {'network_state_dict': network.state_dict()}

    inferred = model_factory_mod.model_config_from_checkpoint(legacy)

    assert inferred.architecture == 'cnn'
    assert inferred.channels == 16
    assert inferred.num_blocks == 1
