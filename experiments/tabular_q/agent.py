"""Tabular Q-Learning agent for Ultimate Tic-Tac-Toe.

Imports only from 01_game. No PyTorch dependency.
"""
from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


class TabularQAgent:
    """Q-learning agent using a dictionary as the Q-table.

    State is hashed as (cells bytes, sub_board_results bytes, active_sub_board).
    Q(state, move) is initialised to 0.0 for unseen pairs.

    Args:
        epsilon: Probability of choosing a random legal move.
        alpha:   TD learning rate.
        gamma:   Discount factor for future rewards.
    """

    def __init__(
        self,
        epsilon: float = 0.3,
        alpha: float = 0.1,
        gamma: float = 0.99,
    ) -> None:
        self.epsilon = epsilon
        self.alpha   = alpha
        self.gamma   = gamma
        # (state_key, move_idx) → float
        self.q: dict[tuple, float] = defaultdict(float)

    # ------------------------------------------------------------------
    # State hashing
    # ------------------------------------------------------------------

    def _key(self, state) -> tuple:
        return (
            state.cells.tobytes(),
            state.sub_board_results.tobytes(),
            state.active_sub_board,
        )

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state, legal_moves: list[int]) -> int:
        """Epsilon-greedy action selection over legal moves."""
        if not legal_moves:
            raise ValueError("No legal moves available.")
        if random.random() < self.epsilon:
            return random.choice(legal_moves)
        key = self._key(state)
        return max(legal_moves, key=lambda m: self.q[(key, m)])

    # ------------------------------------------------------------------
    # TD update
    # ------------------------------------------------------------------

    def update(
        self,
        state,
        move: int,
        reward: float,
        next_state,
        next_legal: list[int],
        done: bool,
    ) -> None:
        """Single-step Q-learning (TD(0)) update.

        Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') − Q(s,a)]
        """
        key = self._key(state)
        current_q = self.q[(key, move)]

        if done or not next_legal:
            target = reward
        else:
            next_key  = self._key(next_state)
            best_next = max(self.q[(next_key, m)] for m in next_legal)
            target    = reward + self.gamma * best_next

        self.q[(key, move)] += self.alpha * (target - current_q)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def table_size(self) -> int:
        return len(self.q)
