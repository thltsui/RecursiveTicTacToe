"""Episode 19 — The Arena — Pitting Networks Against Each Other to Measure Progress

Concept taught: Why loss decrease doesn't always mean the model is actually better.
How to use head-to-head games as the true measure of improvement.
The 55% win rate threshold from AlphaZero.
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@dataclass
class ArenaResult:
    """Result of an arena match between two networks.

    Attributes:
        wins: Number of games won by the new network.
        losses: Number of games lost by the new network.
        draws: Number of drawn games.
        win_rate: wins / (wins + losses + draws).
    """
    wins:     int
    losses:   int
    draws:    int
    win_rate: float
    win_rate_threshold: float = 0.55

    @property
    def new_is_better(self) -> bool:
        """Return True if new network should replace old."""
        return self.win_rate > self.win_rate_threshold


def play_single_game(
    player1_network: 'UltimateTTTNetwork',
    player2_network: 'UltimateTTTNetwork',
    num_simulations: int,
    device: str,
) -> int:
    """Play one game between two networks.

    Both networks use MCTS with temperature=0 (greedy) and no Dirichlet noise
    for deterministic evaluation.

    Args:
        player1_network: Network playing as player 1.
        player2_network: Network playing as player 2.
        num_simulations: MCTS simulations per move.
        device: Compute device.

    Returns:
        1 if player1 wins, -1 if player2 wins, 0 if draw.
    """
    from importlib import import_module
    board_mod = import_module('01_game.board')
    rules_mod = import_module('01_game.rules')
    search_mod = import_module('03_mcts.search')

    state = board_mod.create_initial_state()

    move_count = 0
    while not state.is_terminal:
        # Select network based on current player
        net = player1_network if state.current_player == 1 else player2_network

        # Run MCTS with no exploration noise (arena = deterministic)
        root = search_mod.run_mcts(
            state, net,
            num_simulations=num_simulations,
            dirichlet_epsilon=0.0,  # no noise
            device=device,
        )

        # Temperature = 1.0 for first 4 moves to ensure game variety, then greedy
        temp = 1.0 if move_count < 4 else 0.0
        move = search_mod.select_move(root, temperature=temp)
        state = rules_mod.apply_move(state, move)
        move_count += 1

    return state.winner if state.winner is not None else 0


def run_arena(
    network_new: 'UltimateTTTNetwork',
    network_old: 'UltimateTTTNetwork',
    num_games: int = 100,
    num_simulations: int = 400,
    win_rate_threshold: float = 0.55,
    device: str = 'cpu',
) -> ArenaResult:
    """Pit two networks against each other over num_games games.

    Each network plays as both player 1 and player 2 (alternates).
    Temperature = 0 during arena (greedy play, no exploration noise).
    No Dirichlet noise at root (deterministic evaluation).

    Args:
        network_new: Candidate new network to evaluate.
        network_old: Current best network (baseline).
        num_games: Total games to play (should be even for fair alternation).
        num_simulations: MCTS simulations per move during arena.
        device: Compute device.

    Returns:
        ArenaResult with win/loss/draw counts and win rate.
    """
    wins = 0
    losses = 0
    draws = 0

    for game_i in range(num_games):
        if game_i % 2 == 0:
            # New network plays as player 1
            result = play_single_game(network_new, network_old,
                                       num_simulations, device)
            if result == 1:
                wins += 1
            elif result == -1:
                losses += 1
            else:
                draws += 1
        else:
            # New network plays as player 2
            result = play_single_game(network_old, network_new,
                                       num_simulations, device)
            if result == -1:
                wins += 1  # new network won as player 2
            elif result == 1:
                losses += 1  # new network lost
            else:
                draws += 1

    total = wins + losses + draws
    win_rate = wins / total if total > 0 else 0.0

    return ArenaResult(
        wins=wins,
        losses=losses,
        draws=draws,
        win_rate=win_rate,
        win_rate_threshold=win_rate_threshold,
    )


if __name__ == "__main__":
    from importlib import import_module
    network_mod = import_module('02_network.network')

    print("=== Episode 19: Arena ===\n")

    # Create two small networks
    net1 = network_mod.UltimateTTTNetwork(channels=32, num_blocks=2)
    net2 = network_mod.UltimateTTTNetwork(channels=32, num_blocks=2)

    # Play 2 games (1 as each color) with very few simulations
    result = run_arena(net1, net2, num_games=2, num_simulations=10, device='cpu')

    print(f"Arena result: {result.wins}W / {result.losses}L / {result.draws}D")
    print(f"Win rate: {result.win_rate:.3f}")
    print(f"New is better: {result.new_is_better}")

    assert result.wins + result.losses + result.draws == 2
    assert 0.0 <= result.win_rate <= 1.0

    print("\n=== Episode 19 PASSED ===")
