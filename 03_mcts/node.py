"""Episode 8 — The MCTS Node — Exploration vs Exploitation with UCT

Concept taught: The UCT (Upper Confidence Bound for Trees) formula. Why we need
to balance exploring unknown moves vs exploiting known good moves. How visit counts
and value estimates are maintained per node.

The PUCT variant (used by AlphaZero):
    PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from importlib import import_module


class MCTSNode:
    """A single node in the Monte Carlo search tree.

    Each node represents a game state. It stores statistics about
    how often each action has been explored and what value was observed.

    Attributes:
        state: The GameState this node represents.
        parent: Parent MCTSNode or None if root.
        move_from_parent: The move_idx that led to this node (None for root).
        children: Dict mapping move_idx -> MCTSNode.
        N: Dict[int, int] — visit count per action.
        W: Dict[int, float] — total value per action.
        Q: Dict[int, float] — mean value = W[a] / N[a] (0 if N[a]==0).
        P: Dict[int, float] — prior probability from network.
        visit_count: int — total visits to this node.
        is_expanded: bool — whether children have been created.
        is_terminal: bool — whether this is a terminal game state.
    """

    def __init__(
        self,
        state: 'GameState',
        parent: MCTSNode | None = None,
        move_from_parent: int | None = None,
        prior: float = 0.0,
    ):
        self.state = state
        self.parent = parent
        self.move_from_parent = move_from_parent
        self.prior = prior

        self.children: dict[int, MCTSNode] = {}
        self.N: dict[int, int] = {}
        self.W: dict[int, float] = {}
        self.Q: dict[int, float] = {}
        self.P: dict[int, float] = {}

        self.visit_count: int = 0
        self.is_expanded: bool = False
        self.is_terminal: bool = state.is_terminal

    def puct_score(self, move_idx: int, c_puct: float = 1.0) -> float:
        """Compute PUCT score for a specific action.

        PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))

        Args:
            move_idx: The action to score.
            c_puct: Exploration constant.

        Returns:
            PUCT score (higher = more attractive to visit).
        """
        q_value = self.Q.get(move_idx, 0.0)
        prior = self.P.get(move_idx, 0.0)
        n_action = self.N.get(move_idx, 0)
        n_parent = self.visit_count

        exploration = c_puct * prior * math.sqrt(n_parent) / (1 + n_action)
        return q_value + exploration

    def select_child(self, c_puct: float = 1.0) -> tuple[int, 'MCTSNode']:
        """Select the child with the highest PUCT score.

        Only selects among expanded children (children dict).
        Assumes node is already expanded.

        Args:
            c_puct: Exploration constant.

        Returns:
            (move_idx, child_node) of the best child.

        Raises:
            ValueError: If node has no children.
        """
        if not self.children:
            raise ValueError("Node has no children (not expanded or terminal)")

        best_score = float('-inf')
        best_move = -1
        best_child = None

        for move_idx, child in self.children.items():
            score = self.puct_score(move_idx, c_puct)
            if score > best_score:
                best_score = score
                best_move = move_idx
                best_child = child

        return best_move, best_child

    def expand(self, prior_probs: torch.Tensor, legal_moves: list[int]) -> None:
        """Create child nodes for all legal moves.

        Args:
            prior_probs: Network policy output, shape (81,).
                         Values for illegal moves are ignored.
            legal_moves: List of legal move indices.

        Sets is_expanded = True.
        Creates MCTSNode children with prior probabilities from network.
        Does NOT apply Dirichlet noise here — that happens at the root in search.py.
        """
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from importlib import import_module
        rules_mod = import_module('01_game.rules')
        apply_move = rules_mod.apply_move

        for move_idx in legal_moves:
            p = prior_probs[move_idx].item()
            self.P[move_idx] = p
            self.N[move_idx] = 0
            self.W[move_idx] = 0.0
            self.Q[move_idx] = 0.0

            child_state = apply_move(self.state, move_idx)
            child = MCTSNode(
                state=child_state,
                parent=self,
                move_from_parent=move_idx,
                prior=p,
            )
            self.children[move_idx] = child

        self.is_expanded = True

    def backup(self, value: float) -> None:
        """Backpropagate value up to the root.

        Updates W and N for the edge that led to this node,
        then recursively calls parent.backup(-value) because
        value is always from the perspective of the player who
        just moved, so we negate when going up to parent.

        Args:
            value: Outcome value from this node's perspective (+1 win, -1 loss).
        """
        self.visit_count += 1

        if self.parent is not None and self.move_from_parent is not None:
            move = self.move_from_parent
            self.parent.N[move] = self.parent.N.get(move, 0) + 1
            self.parent.W[move] = self.parent.W.get(move, 0.0) + value
            n = self.parent.N[move]
            self.parent.W[move]
            self.parent.Q[move] = self.parent.W[move] / n

            # Negate value when going up — parent's perspective is opposite
            self.parent.backup(-value)

    def get_visit_counts(self) -> dict[int, int]:
        """Return {move_idx: visit_count} for all visited children."""
        return {move: count for move, count in self.N.items() if count > 0}

    def is_leaf(self) -> bool:
        """Return True if node has not been expanded yet."""
        return not self.is_expanded


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from importlib import import_module

    board_mod = import_module('01_game.board')
    rules_mod = import_module('01_game.rules')

    print("=== Episode 8: MCTS Node ===\n")

    state = board_mod.create_initial_state()
    root = MCTSNode(state=state)

    # Test initial state
    assert root.is_leaf()
    assert not root.is_terminal
    assert root.visit_count == 0

    # Test expansion
    legal = rules_mod.get_legal_moves(state)
    fake_prior = torch.ones(81) / 81.0
    root.expand(fake_prior, legal)

    assert root.is_expanded
    assert len(root.children) == 81
    print(f"Expanded root with {len(root.children)} children")

    # Test PUCT scoring
    score = root.puct_score(0)
    print(f"PUCT score for move 0 (no visits): {score:.4f}")

    # Test select_child
    move, child = root.select_child()
    print(f"Selected child: move={move}")

    # Test backup
    child.backup(1.0)
    assert root.N[move] == 1
    assert root.Q[move] != 0.0
    print(f"After backup: N[{move}]={root.N[move]}, Q[{move}]={root.Q[move]:.4f}")

    print("\n=== Episode 8 PASSED ===")
