"""Episode 18 — The Elo Rating System — Measuring AI Playing Strength

Concept taught: How Elo works mathematically. Why it's the right metric for
comparing AI checkpoints. The K-factor and what it controls.

Elo formula:
    E_A = 1 / (1 + 10^((R_B - R_A) / 400))
    R_new = R_old + K * (actual_score - expected_score)
"""

from __future__ import annotations


def expected_score(rating_a: float, rating_b: float) -> float:
    """Compute expected score for player A against player B.

    Formula: E_A = 1 / (1 + 10^((R_B - R_A) / 400))

    Args:
        rating_a: Player A's Elo rating.
        rating_b: Player B's Elo rating.

    Returns:
        Value in [0, 1] — probability player A wins.
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo(
    rating: float,
    opponent_rating: float,
    actual_score: float,
    k_factor: float = 32.0,
) -> float:
    """Update Elo rating after a game result.

    Formula: R_new = R_old + K * (actual_score - expected_score)

    Args:
        rating: Current player's Elo rating.
        opponent_rating: Opponent's Elo rating.
        actual_score: 1.0 for win, 0.5 for draw, 0.0 for loss.
        k_factor: Controls rating volatility (higher = faster changes).

    Returns:
        Updated Elo rating.
    """
    expected = expected_score(rating, opponent_rating)
    return rating + k_factor * (actual_score - expected)


class EloTracker:
    """Tracks Elo ratings for multiple agents across training.

    Maintains a history of Elo ratings over time for plotting.

    Args:
        initial_rating: Starting Elo for new agents. Default: 1000.
        k_factor: Elo K-factor. Default: 32.
    """

    def __init__(self, initial_rating: float = 1000.0, k_factor: float = 32.0):
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self._ratings: dict[str, float] = {}
        self._history: dict[str, list[tuple[int, float]]] = {}
        self._game_count: int = 0

    def add_agent(self, name: str) -> None:
        """Register a new agent with the initial rating.

        Args:
            name: Unique identifier for the agent.
        """
        self._ratings[name] = self.initial_rating
        self._history[name] = [(0, self.initial_rating)]

    def record_result(self, winner: str, loser: str, draw: bool = False) -> None:
        """Record game result and update both ratings.

        Args:
            winner: Name of the winning agent (ignored if draw=True).
            loser: Name of the losing agent (ignored if draw=True).
            draw: If True, record as a draw.
        """
        self._game_count += 1

        if draw:
            score_w, score_l = 0.5, 0.5
        else:
            score_w, score_l = 1.0, 0.0

        r_w = self._ratings[winner]
        r_l = self._ratings[loser]

        self._ratings[winner] = update_elo(r_w, r_l, score_w, self.k_factor)
        self._ratings[loser] = update_elo(r_l, r_w, score_l, self.k_factor)

        self._history[winner].append((self._game_count, self._ratings[winner]))
        self._history[loser].append((self._game_count, self._ratings[loser]))

    def get_rating(self, name: str) -> float:
        """Get current Elo rating for an agent.

        Args:
            name: Agent identifier.

        Returns:
            Current Elo rating.
        """
        return self._ratings[name]

    def get_history(self, name: str) -> list[tuple[int, float]]:
        """Get (game_number, elo) history for plotting.

        Args:
            name: Agent identifier.

        Returns:
            List of (game_number, elo_rating) tuples.
        """
        return self._history[name]


if __name__ == "__main__":
    print("=== Episode 18: Elo Rating System ===\n")

    # Test expected score
    e = expected_score(1000.0, 1000.0)
    print(f"Equal ratings -> expected score: {e:.3f}")
    assert abs(e - 0.5) < 1e-6

    e_strong = expected_score(1200.0, 1000.0)
    print(f"200 Elo advantage -> expected score: {e_strong:.3f}")
    assert e_strong > 0.7

    # Test update
    new_rating = update_elo(1000.0, 1000.0, 1.0, k_factor=32.0)
    print(f"Win vs equal -> new rating: {new_rating:.1f}")
    assert new_rating > 1000.0

    new_rating_loss = update_elo(1000.0, 1000.0, 0.0, k_factor=32.0)
    print(f"Loss vs equal -> new rating: {new_rating_loss:.1f}")
    assert new_rating_loss < 1000.0

    # Test tracker
    tracker = EloTracker()
    tracker.add_agent("net_v1")
    tracker.add_agent("net_v2")

    for _ in range(10):
        tracker.record_result("net_v2", "net_v1")

    r1 = tracker.get_rating("net_v1")
    r2 = tracker.get_rating("net_v2")
    print(f"\nAfter 10 games (v2 always wins):")
    print(f"  net_v1: {r1:.1f}")
    print(f"  net_v2: {r2:.1f}")
    assert r2 > r1

    history = tracker.get_history("net_v2")
    print(f"  v2 history length: {len(history)}")
    assert len(history) == 11  # initial + 10 games

    print("\n=== Episode 18 PASSED ===")
