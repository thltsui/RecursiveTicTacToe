"""Tests for Episodes 8-10 — MCTS node, search, policy target."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import torch

from importlib import import_module
board_mod = import_module('01_game.board')
rules_mod = import_module('01_game.rules')
node_mod = import_module('03_mcts.node')
search_mod = import_module('03_mcts.search')
policy_target_mod = import_module('03_mcts.policy_target')
network_mod = import_module('02_network.network')


class TestMCTSNode:
    def test_initial_state(self):
        state = board_mod.create_initial_state()
        node = node_mod.MCTSNode(state=state)
        assert node.is_leaf()
        assert not node.is_terminal
        assert node.visit_count == 0

    def test_expansion(self):
        state = board_mod.create_initial_state()
        node = node_mod.MCTSNode(state=state)
        legal = rules_mod.get_legal_moves(state)
        prior = torch.ones(81) / 81.0
        node.expand(prior, legal)
        assert node.is_expanded
        assert len(node.children) == len(legal)

    def test_puct_unexplored_high(self):
        """Unexplored moves should have higher exploration component."""
        state = board_mod.create_initial_state()
        node = node_mod.MCTSNode(state=state)
        legal = rules_mod.get_legal_moves(state)
        prior = torch.ones(81) / 81.0
        node.expand(prior, legal)
        node.visit_count = 100
        node.N[legal[0]] = 50
        node.Q[legal[0]] = 0.0  # neutral Q
        # Move 0 has many visits, move 1 has none
        score_visited = node.puct_score(legal[0])
        score_unvisited = node.puct_score(legal[1])
        assert score_unvisited > score_visited  # exploration bonus dominates

    def test_backup(self):
        state = board_mod.create_initial_state()
        root = node_mod.MCTSNode(state=state)
        legal = rules_mod.get_legal_moves(state)
        prior = torch.ones(81) / 81.0
        root.expand(prior, legal)

        child = root.children[legal[0]]
        child.backup(1.0)

        assert root.N[legal[0]] == 1
        assert root.W[legal[0]] == -1.0
        assert root.visit_count == 1


class TestMCTSSearch:
    def test_run_mcts(self):
        state = board_mod.create_initial_state()
        net = network_mod.UltimateTTTNetwork(channels=32, num_blocks=2)
        root = search_mod.run_mcts(state, net, num_simulations=20)
        visits = root.get_visit_counts()
        assert len(visits) > 0
        assert sum(visits.values()) > 0

    def test_select_move_greedy(self):
        state = board_mod.create_initial_state()
        net = network_mod.UltimateTTTNetwork(channels=32, num_blocks=2)
        root = search_mod.run_mcts(state, net, num_simulations=20)
        move = search_mod.select_move(root, temperature=0.0)
        visits = root.get_visit_counts()
        most_visited = max(visits, key=visits.get)
        assert move == most_visited


class TestPolicyTarget:
    def test_sums_to_one(self):
        visits = {0: 100, 1: 50, 2: 10}
        target = policy_target_mod.compute_policy_target(visits, 3)
        assert abs(target.sum().item() - 1.0) < 1e-5

    def test_pruning(self):
        visits = {0: 100, 1: 50, 2: 1}
        target = policy_target_mod.compute_policy_target(visits, 3, pruning_threshold=0.05)
        assert target[2].item() == 0.0  # 1/151 < 0.05

    def test_empty_visits(self):
        target = policy_target_mod.compute_policy_target({}, 0)
        assert target.sum().item() == 0.0

    def test_forced_playouts(self):
        visits = {0: 100}
        legal = [0, 1, 2]
        forced = policy_target_mod.add_forced_playouts(visits, legal, forced_playout_k=5)
        for m in legal:
            assert forced[m] >= 5
