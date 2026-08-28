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


def test_hard_mode_uses_the_deeper_search_budget():
    assert web_mod.DIFFICULTY_SIMS == {
        'easy': 80,
        'medium': 200,
        'hard': 800,
    }
    assert web_mod.ANALYSIS_PREVIEW_SIMS == 100
    assert web_mod.MIN_Q_DISPLAY_VISITS == 10


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


def test_client_refines_preview_analysis_and_hides_sparse_q_values():
    index_path = os.path.join(PROJECT_ROOT, 'web_app', 'static', 'index.html')
    script_path = os.path.join(PROJECT_ROOT, 'web_app', 'static', 'main.js')
    with open(index_path, encoding='utf-8') as handle:
        index = handle.read()
    with open(script_path, encoding='utf-8') as handle:
        script = handle.read()

    assert 'Q (n≥10)' in index
    assert "fetch('/api/analyze'" in script
    assert 'data.analysis.is_preview' in script
    assert 'm.visits < minQDisplayVisits' in script
    assert 'a.mcts_visits[idx] < minQVisits' in script


def test_hard_ai_move_returns_a_preview_with_full_depth_target(monkeypatch):
    calls = []

    def fake_analysis(state, sims, add_noise=False):
        calls.append((sims, add_noise))
        return {
            'total_sims': sims,
            'target_sims': sims,
            'is_preview': False,
            'min_q_display_visits': web_mod.MIN_Q_DISPLAY_VISITS,
        }, object()

    monkeypatch.setattr(web_mod, 'get_analysis', fake_analysis)
    monkeypatch.setattr(web_mod.search_mod, 'select_move', lambda root, temperature: 0)

    result = web_mod.do_ai_move(web_mod.create_initial_state(), 800, 'hard')

    assert calls == [(800, False), (100, False)]
    assert result['analysis']['total_sims'] == 100
    assert result['analysis']['target_sims'] == 800
    assert result['analysis']['is_preview'] is True


def test_analyze_endpoint_honors_full_hard_budget(monkeypatch):
    calls = []

    def fake_analysis(state, sims, add_noise=False):
        calls.append((sims, add_noise))
        return {'total_sims': sims, 'is_preview': False}, None

    monkeypatch.setattr(web_mod, 'get_analysis', fake_analysis)
    state = web_mod.create_initial_state()
    payload = {
        'state': {
            'cells': state.cells.tolist(),
            'sub_board_results': state.sub_board_results.tolist(),
            'active_sub_board': state.active_sub_board,
            'current_player': state.current_player,
            'move_count': state.move_count,
            'is_terminal': state.is_terminal,
            'winner': state.winner,
        },
        'difficulty': 'hard',
    }

    response = web_mod.app.test_client().post('/api/analyze', json=payload)

    assert response.status_code == 200
    assert calls == [(800, False)]
    assert response.get_json()['analysis']['total_sims'] == 800
