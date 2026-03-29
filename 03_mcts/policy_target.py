"""Episode 10 — Policy Target Pruning — Why Raw MCTS Visits Are Noisy Training Targets

Concept taught: The KataGo insight that MCTS visit distributions are optimized for
SEARCH (need exploration) but not for TRAINING (want clean signal). Policy target pruning
removes noise before the network learns from it.

The Problem: During MCTS, we add Dirichlet noise and use temperature > 0. This means
some moves get visited just for exploration, not because they're actually good. If we
train the network to predict these noisy visit counts, we're training on noise.
"""

from __future__ import annotations

import torch


def compute_policy_target(
    visit_counts: dict[int, int],
    num_legal_moves: int,
    pruning_threshold: float = 0.0,
) -> torch.Tensor:
    """Convert MCTS visit counts to clean training target.

    Pruning strategy:
        1. Convert visit counts to probabilities: pi(a) = N(a) / sum(N).
        2. Prune moves with probability below threshold.
        3. Renormalize remaining probabilities to sum to 1.0.

    The threshold removes exploratory low-visit moves that provide
    noisy training signal.

    Args:
        visit_counts: Dict {move_idx: N(a)} from MCTS root node.
        num_legal_moves: Number of legal moves (for context).
        pruning_threshold: Minimum probability to keep a move.
            0.0 means no pruning (use all visits).
            Recommended: 1 / num_simulations — prune single-visit moves.

    Returns:
        Tensor of shape (81,), dtype=float32.
        Probability distribution over all 81 moves.
        Illegal/pruned moves have probability 0.0.
        All values sum to 1.0.
    """
    target = torch.zeros(81, dtype=torch.float32)

    if not visit_counts:
        return target

    total_visits = sum(visit_counts.values())
    if total_visits == 0:
        return target

    # Convert to probabilities
    for move, count in visit_counts.items():
        prob = count / total_visits
        if prob >= pruning_threshold:
            target[move] = prob

    # Renormalize after pruning
    total = target.sum()
    if total > 0:
        target = target / total

    return target


def add_forced_playouts(
    visit_counts: dict[int, int],
    legal_moves: list[int],
    forced_playout_k: int = 2,
) -> dict[int, int]:
    """Ensure every legal move gets a minimum number of visits.

    KataGo improvement: Before pruning, force at least k visits to every
    legal move. This prevents premature convergence where the network
    never explores moves it initially considered unlikely.

    Args:
        visit_counts: Existing visit counts from MCTS.
        legal_moves: All legal moves that should be visited.
        forced_playout_k: Minimum visits per legal move.

    Returns:
        Updated visit counts with forced playouts added.
    """
    updated = dict(visit_counts)
    for move in legal_moves:
        current = updated.get(move, 0)
        if current < forced_playout_k:
            updated[move] = forced_playout_k
    return updated


if __name__ == "__main__":
    print("=== Episode 10: Policy Target Pruning ===\n")

    # Test basic conversion
    visits = {0: 100, 1: 50, 2: 10, 3: 1, 4: 1}
    target = compute_policy_target(visits, num_legal_moves=5)
    print(f"Visit counts: {visits}")
    print(f"Target (no pruning): top 5 values = {target[target > 0].tolist()}")
    assert abs(target.sum().item() - 1.0) < 1e-5, f"Should sum to 1, got {target.sum()}"

    # Test with pruning
    target_pruned = compute_policy_target(visits, num_legal_moves=5, pruning_threshold=0.05)
    print(f"Target (pruned 5%): top values = {target_pruned[target_pruned > 0].tolist()}")
    assert target_pruned[3].item() == 0.0, "Move 3 should be pruned (1/162 < 0.05)"
    assert abs(target_pruned.sum().item() - 1.0) < 1e-5

    # Test forced playouts
    sparse_visits = {0: 100, 1: 50}
    legal = [0, 1, 2, 3, 4]
    forced = add_forced_playouts(sparse_visits, legal, forced_playout_k=2)
    print(f"\nBefore forced: {sparse_visits}")
    print(f"After forced:  {forced}")
    for m in legal:
        assert forced.get(m, 0) >= 2, f"Move {m} should have >= 2 visits"

    # Test empty
    empty_target = compute_policy_target({}, num_legal_moves=0)
    assert empty_target.sum() == 0.0

    print("\n=== Episode 10 PASSED ===")
