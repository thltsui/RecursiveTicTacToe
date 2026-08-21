"""Focused tests for deployment-facing web behavior."""

from __future__ import annotations

import os
import sys
from importlib import import_module


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

web_mod = import_module('web_app.app')


def test_only_easy_mode_adds_search_noise():
    assert web_mod.exploration_noise_for_difficulty('easy') is True
    assert web_mod.exploration_noise_for_difficulty('medium') is False
    assert web_mod.exploration_noise_for_difficulty('hard') is False


def test_model_info_endpoint_reports_loaded_checkpoint_metadata():
    original = web_mod.network_metadata
    expected = {
        'architecture': 'transformer',
        'channels': 96,
        'layers': 3,
        'iteration': 30,
        'parameters': 346707,
        'wdl_calibrated': True,
        'release_status': 'experimental_portfolio_trial',
    }
    try:
        web_mod.network_metadata = expected
        response = web_mod.app.test_client().get('/api/model_info')
    finally:
        web_mod.network_metadata = original

    assert response.status_code == 200
    assert response.get_json() == expected


def test_footer_is_populated_from_runtime_model_info():
    index_path = os.path.join(PROJECT_ROOT, 'web_app', 'static', 'index.html')
    script_path = os.path.join(PROJECT_ROOT, 'web_app', 'static', 'main.js')
    with open(index_path, encoding='utf-8') as handle:
        index = handle.read()
    with open(script_path, encoding='utf-8') as handle:
        script = handle.read()

    assert 'id="model-info"' in index
    assert '/api/model_info' in script
    assert '192ch × 10 blocks' not in index
