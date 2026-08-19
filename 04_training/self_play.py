"""Episode 11 — Self-Play — How the AI Generates Its Own Training Data

Concept taught: The self-play loop. How MCTS + network generates game records.
What data we collect per move and why. Temperature scheduling.
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field

import numpy as np
import torch
from importlib import import_module

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

outcome_mod = import_module('01_game.outcome')


@dataclass
class MoveRecord:
    """Data collected for a single move during self-play.

    Attributes:
        state_tensor: Encoded state at time of move, shape (7, 9, 9).
        policy_target: Pruned MCTS visit distribution, shape (81,).
        opp_policy_target: Next player's MCTS target, shape (81,).
        opp_legal_mask: Legal moves for next player's state, shape (81,).
        legal_mask: Legal moves at this state, shape (81,).
        current_player: 1 or -1 (raw game player, before encoding flip).
        wdl_target: Class index ordered win=0, draw=1, loss=2.
        value_target: Zero-sum expectation +1 if winner, -1 if loser, 0 if draw.
        score_target: Final score margin from current player's perspective. Set after game.
        ownership_target: Sub-board ownership from current player's perspective. Shape (9,).
    """
    state_tensor: torch.Tensor          # (7, 9, 9)
    policy_target: torch.Tensor         # (81,)
    opp_policy_target: torch.Tensor     # (81,)
    opp_legal_mask: torch.Tensor        # (81,)
    legal_mask: torch.Tensor            # (81,)
    current_player: int
    wdl_target: int = outcome_mod.WDL_DRAW  # set after game ends
    value_target: float = 0.0           # set after game ends
    score_target: float = 0.0           # set after game ends
    ownership_target: torch.Tensor = field(default_factory=lambda: torch.zeros(9))  # (9,)


@dataclass
class GameRecord:
    """Complete record of a self-play game.

    Attributes:
        moves: List of MoveRecord objects.
        winner: 1, -1, or 0 (draw).
        game_length: Number of moves played.
        final_sub_board_results: Shape (9,) — who won each sub-board.
    """
    moves: list[MoveRecord]
    winner: int
    game_length: int
    final_sub_board_results: np.ndarray = field(
        default_factory=lambda: np.zeros(9, dtype=np.int8)
    )


def _finalize_move_targets(
    move_records: list[MoveRecord],
    winner: int,
    final_results: np.ndarray,
) -> None:
    """Attach one coherent set of zero-sum outcome targets to a game.

    Every data source uses the same W/D/L semantics.  Draw aversion belongs in
    move selection or an antisymmetric auxiliary objective, never in the value
    target consumed by negamax MCTS.
    """
    for rec in move_records:
        cp = rec.current_player
        rec.wdl_target = outcome_mod.outcome_class(winner, cp)
        rec.value_target = outcome_mod.outcome_value(winner, cp)

        own_wins = np.sum(final_results == cp)
        opp_wins = np.sum(final_results == -cp)
        rec.score_target = float(own_wins - opp_wins) / 9.0
        rec.ownership_target = torch.tensor(
            [(1.0 if final_results[i] == cp else 0.0) for i in range(9)],
            dtype=torch.float32,
        )


def play_self_play_game(
    network: 'UltimateTTTNetwork',
    num_simulations: int = 800,
    temp_initial: float = 2.0,
    temp_decay_rate: float = 0.94,
    temp_min: float = 0.15,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.35,
    device: str = 'cpu',
    dirichlet_epsilon_boost: float = 0.55,
    dirichlet_boost_plies: int = 5,
    forced_opening: list[int] | None = None,
) -> GameRecord:
    """Play one complete self-play game and return the game record.

    Temperature schedule (exponential decay):
        - temp = temp_initial * (temp_decay_rate ** move_count)
        - If temp < temp_min, temp snaps to 0.0 (greedy endgame).

    Root Dirichlet-noise schedule:
        - Moves 0..dirichlet_boost_plies-1 use dirichlet_epsilon_boost.
        - Moves after that use the normal dirichlet_epsilon.
        - Set dirichlet_boost_plies=0 to disable boosting (flat dirichlet_epsilon
          throughout, matching old behavior).
        See 03_mcts/search.py's epsilon_for_ply() for the rationale: this
        counteracts PUCT under-exploring low-prior openings early in training,
        which otherwise lets an early, possibly-wrong belief about a move
        compound instead of getting re-examined.

    Forced opening:
        - If forced_opening is given, it's a list of move indices that override
          MCTS's own move selection for the first len(forced_opening) plies of
          the game. Search still runs at those plies (so policy_target reflects
          genuine MCTS visit statistics), but the move actually applied to the
          board is taken from forced_opening instead of select_move()'s output.
          This guarantees the network accumulates real training data on
          specific lines regardless of whether its own search currently
          favors them -- see generate_mixed_batch()'s forced_opening_fraction.

    After game ends, compute and attach value/score/ownership targets to
    each MoveRecord in the game.

    Args:
        network: Current best network (used for both players).
        num_simulations: MCTS simulations per move.
        temp_initial: Initial temperature at move 0.
        temp_decay_rate: Exponential decay multiplier per move.
        temp_min: Minimum temperature before snapping to 0.0.
        device: Compute device.
        dirichlet_epsilon_boost: Root epsilon used during the early-ply boost window.
        dirichlet_boost_plies: Number of opening plies over which the boost applies.
        forced_opening: Optional list of move indices to force at the start of
            the game, in order, overriding MCTS's own move choice for those plies.

    Returns:
        Complete GameRecord with all targets computed.
    """
    from importlib import import_module
    board_mod = import_module('01_game.board')
    rules_mod = import_module('01_game.rules')
    search_mod = import_module('03_mcts.search')
    policy_target_mod = import_module('03_mcts.policy_target')

    create_initial_state = board_mod.create_initial_state
    encode_state = board_mod.encode_state
    get_legal_moves = rules_mod.get_legal_moves
    get_legal_move_mask = rules_mod.get_legal_move_mask
    run_mcts = search_mod.run_mcts
    select_move = search_mod.select_move
    epsilon_for_ply = search_mod.epsilon_for_ply
    compute_policy_target = policy_target_mod.compute_policy_target

    state = create_initial_state()
    move_records: list[MoveRecord] = []
    ply = 0

    while not state.is_terminal:
        # Root Dirichlet epsilon: boosted for the first few plies, then normal.
        ply_epsilon = epsilon_for_ply(
            state.move_count,
            base_epsilon=dirichlet_epsilon,
            boosted_epsilon=dirichlet_epsilon_boost,
            boost_plies=dirichlet_boost_plies,
        )

        # Run MCTS
        root = run_mcts(state, network, num_simulations=num_simulations,
                        dirichlet_alpha=dirichlet_alpha,
                        dirichlet_epsilon=ply_epsilon,
                        device=device)

        # Compute policy target from visit counts
        visits = root.get_visit_counts()
        legal_moves = get_legal_moves(state)
        policy_tgt = compute_policy_target(visits, len(legal_moves))

        # Select move with exponential temperature decay
        temp = temp_initial * (temp_decay_rate ** state.move_count)
        if temp < temp_min:
            temp = 0.0

        if forced_opening is not None and ply < len(forced_opening):
            move = forced_opening[ply]
            if move not in legal_moves:
                # Safety fallback -- forced move illegal in this position
                # (shouldn't happen for a well-formed opening pool, but don't
                # let a bad config value crash a training run).
                move = select_move(root, temperature=temp)
        else:
            move = select_move(root, temperature=temp)

        # Record this move
        record = MoveRecord(
            state_tensor=encode_state(state),
            policy_target=policy_tgt,
            opp_policy_target=torch.zeros(81),  # placeholder, set below
            opp_legal_mask=torch.zeros(81),     # placeholder, set below
            legal_mask=get_legal_move_mask(state),
            current_player=state.current_player,
        )
        move_records.append(record)

        # Apply move
        from importlib import import_module
        rules = import_module('01_game.rules')
        state = rules.apply_move(state, move)
        ply += 1

    # Set opponent policy targets: move i's opp target = move i+1's policy target
    for i in range(len(move_records) - 1):
        move_records[i].opp_policy_target = move_records[i + 1].policy_target
        move_records[i].opp_legal_mask = move_records[i + 1].legal_mask
    # Last move has no opponent turn in this game, so ignore aux target via zero mask.
    if move_records:
        move_records[-1].opp_policy_target = torch.zeros(81)
        move_records[-1].opp_legal_mask = torch.zeros(81)

    # Compute value/score/ownership targets after game ends
    winner = state.winner if state.winner is not None else 0
    final_results = state.sub_board_results.copy()

    _finalize_move_targets(move_records, winner, final_results)

    return GameRecord(
        moves=move_records,
        winner=winner,
        game_length=len(move_records),
        final_sub_board_results=final_results,
    )


def _network_batch_evaluator(network, device: str):
    """Adapt a network to ``run_mcts_batched`` without tracking gradients."""
    network.eval()

    def evaluate(batch: torch.Tensor):
        with torch.inference_mode():
            return network(batch.to(device))

    return evaluate


def _record_mcts_position(state, root) -> MoveRecord:
    board_mod = import_module('01_game.board')
    rules_mod = import_module('01_game.rules')
    policy_target_mod = import_module('03_mcts.policy_target')
    legal_moves = rules_mod.get_legal_moves(state)
    return MoveRecord(
        state_tensor=board_mod.encode_state(state),
        policy_target=policy_target_mod.compute_policy_target(
            root.get_visit_counts(), len(legal_moves)
        ),
        opp_policy_target=torch.zeros(81),
        opp_legal_mask=torch.zeros(81),
        legal_mask=rules_mod.get_legal_move_mask(state),
        current_player=state.current_player,
    )


def _completed_game_record(state, move_records: list[MoveRecord]) -> GameRecord:
    """Finalize auxiliary and outcome targets for one completed continuation."""
    for index in range(len(move_records) - 1):
        move_records[index].opp_policy_target = move_records[index + 1].policy_target
        move_records[index].opp_legal_mask = move_records[index + 1].legal_mask
    if move_records:
        move_records[-1].opp_policy_target = torch.zeros(81)
        move_records[-1].opp_legal_mask = torch.zeros(81)

    winner = state.winner if state.winner is not None else 0
    final_results = state.sub_board_results.copy()
    _finalize_move_targets(move_records, winner, final_results)
    return GameRecord(
        moves=move_records,
        winner=winner,
        game_length=len(move_records),
        final_sub_board_results=final_results,
    )


def play_self_play_games_batched(
    network: 'UltimateTTTNetwork',
    num_games: int,
    num_simulations: int = 800,
    temp_initial: float = 2.0,
    temp_decay_rate: float = 0.94,
    temp_min: float = 0.15,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.35,
    device: str = 'cpu',
    dirichlet_epsilon_boost: float = 0.55,
    dirichlet_boost_plies: int = 5,
    forced_openings: list[list[int] | None] | None = None,
) -> list[GameRecord]:
    """Play independent self-play games with one leaf-evaluation batch.

    Each game keeps its own tree, temperature, Dirichlet sample, and optional
    forced opening.  Only neural leaf evaluation is shared, so every selected
    leaf is still backed up exactly once per simulation.
    """
    if num_games <= 0:
        return []
    if forced_openings is None:
        forced_openings = [None] * num_games
    if len(forced_openings) != num_games:
        raise ValueError("forced_openings must have one entry per game")

    board_mod = import_module('01_game.board')
    rules_mod = import_module('01_game.rules')
    search_mod = import_module('03_mcts.search')
    states = [board_mod.create_initial_state() for _ in range(num_games)]
    moves_by_game: list[list[MoveRecord]] = [[] for _ in range(num_games)]
    evaluator = _network_batch_evaluator(network, device)

    while True:
        active_indices = [
            index for index, state in enumerate(states) if not state.is_terminal
        ]
        if not active_indices:
            break
        active_states = [states[index] for index in active_indices]
        epsilons = [
            search_mod.epsilon_for_ply(
                state.move_count,
                base_epsilon=dirichlet_epsilon,
                boosted_epsilon=dirichlet_epsilon_boost,
                boost_plies=dirichlet_boost_plies,
            )
            for state in active_states
        ]
        roots = search_mod.run_mcts_batched(
            active_states,
            evaluator,
            num_simulations=num_simulations,
            c_puct=1.0,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=epsilons,
        )

        for active_offset, game_index in enumerate(active_indices):
            state = states[game_index]
            root = roots[active_offset]
            record = _record_mcts_position(state, root)
            moves_by_game[game_index].append(record)

            temperature = temp_initial * (temp_decay_rate ** state.move_count)
            if temperature < temp_min:
                temperature = 0.0
            legal_moves = rules_mod.get_legal_moves(state)
            forced = forced_openings[game_index]
            ply = len(moves_by_game[game_index]) - 1
            if forced is not None and ply < len(forced) and forced[ply] in legal_moves:
                move = forced[ply]
            else:
                move = search_mod.select_move(root, temperature=temperature)
            states[game_index] = rules_mod.apply_move(state, move)

    return [
        _completed_game_record(states[index], moves_by_game[index])
        for index in range(num_games)
    ]


def play_reanalyzed_games_batched(
    network: 'UltimateTTTNetwork',
    start_states: list['GameState'],
    num_simulations: int = 800,
    device: str = 'cpu',
    temperature: float = 0.0,
) -> list[GameRecord]:
    """Adjudicate arbitrary non-terminal states with batched strong play."""
    if any(state.is_terminal for state in start_states):
        raise ValueError("cannot reanalyse a terminal state")
    if not start_states:
        return []

    rules_mod = import_module('01_game.rules')
    search_mod = import_module('03_mcts.search')
    states = [state.copy() for state in start_states]
    moves_by_game: list[list[MoveRecord]] = [[] for _ in states]
    evaluator = _network_batch_evaluator(network, device)

    while True:
        active_indices = [
            index for index, state in enumerate(states) if not state.is_terminal
        ]
        if not active_indices:
            break
        active_states = [states[index] for index in active_indices]
        roots = search_mod.run_mcts_batched(
            active_states,
            evaluator,
            num_simulations=num_simulations,
            c_puct=1.0,
            dirichlet_epsilon=0.0,
        )
        for active_offset, game_index in enumerate(active_indices):
            state = states[game_index]
            root = roots[active_offset]
            moves_by_game[game_index].append(_record_mcts_position(state, root))
            move = search_mod.select_move(root, temperature=temperature)
            states[game_index] = rules_mod.apply_move(state, move)

    return [
        _completed_game_record(states[index], moves_by_game[index])
        for index in range(len(states))
    ]


def play_random_vs_random_game() -> GameRecord:
    """Play a legacy representation-bootstrap game with two random players.

    No network, no MCTS. Both players pick moves uniformly at random.
    Every position from both players is recorded.

    The outcomes estimate random-policy value, not strong-play value.  This
    function is intentionally excluded from ``generate_mixed_batch`` and its
    records must never be mixed into the calibrated W/D/L replay buffer.

    Policy targets are set to the uniform distribution over legal moves.
    These are informational only — the pretrain script sets lambda_policy=0.0
    so they do not contribute to the pretrain loss.

    opp_policy_target and opp_legal_mask are left as zeros (same convention
    as play_vs_random_game — the loss ignores them via zero mask).

    Returns:
        Complete GameRecord with value/score/ownership targets.
    """
    import random
    from importlib import import_module
    board_mod = import_module('01_game.board')
    rules_mod = import_module('01_game.rules')

    create_initial_state = board_mod.create_initial_state
    encode_state = board_mod.encode_state
    get_legal_moves = rules_mod.get_legal_moves
    get_legal_move_mask = rules_mod.get_legal_move_mask

    state = create_initial_state()
    move_records: list[MoveRecord] = []

    while not state.is_terminal:
        legal_moves = get_legal_moves(state)
        move = random.choice(legal_moves)

        # Uniform policy target over legal moves
        legal_mask = get_legal_move_mask(state)
        n_legal = float(legal_mask.sum().item())
        policy_tgt = legal_mask.float() / n_legal  # uniform over legal moves

        record = MoveRecord(
            state_tensor=encode_state(state),
            policy_target=policy_tgt,
            opp_policy_target=torch.zeros(81),
            opp_legal_mask=torch.zeros(81),
            legal_mask=legal_mask,
            current_player=state.current_player,
        )
        move_records.append(record)

        state = rules_mod.apply_move(state, move)

    # Compute value/score/ownership targets
    winner = state.winner if state.winner is not None else 0
    final_results = state.sub_board_results.copy()

    _finalize_move_targets(move_records, winner, final_results)

    return GameRecord(
        moves=move_records,
        winner=winner,
        game_length=len(move_records),
        final_sub_board_results=final_results,
    )


def play_vs_random_game(
    network: 'UltimateTTTNetwork',
    num_simulations: int = 200,
    device: str = 'cpu',
    network_player: int = 1,
) -> GameRecord:
    """Play one legacy evaluation game: network versus a random player.

    Random player samples uniformly from legal moves (no MCTS).
    Its outcomes are opponent-policy conditional and are therefore not valid
    strong-play value targets.  ``generate_mixed_batch`` no longer calls this
    function; it remains for standalone evaluation compatibility.

    IMPORTANT: Only records the NETWORK's moves as training data.
    Recording random player moves would teach the policy head that uniform
    random play is correct, which is the opposite of what we want.

    Args:
        network: Current network.
        num_simulations: MCTS simulations per move.
        device: Compute device.
        network_player: 1 to go first, -1 to go second.

    Returns:
        Complete GameRecord with value/score/ownership targets.
    """
    import random
    from importlib import import_module
    board_mod = import_module('01_game.board')
    rules_mod = import_module('01_game.rules')
    search_mod = import_module('03_mcts.search')
    policy_target_mod = import_module('03_mcts.policy_target')

    create_initial_state = board_mod.create_initial_state
    encode_state = board_mod.encode_state
    get_legal_moves = rules_mod.get_legal_moves
    get_legal_move_mask = rules_mod.get_legal_move_mask
    run_mcts = search_mod.run_mcts
    select_move = search_mod.select_move
    compute_policy_target = policy_target_mod.compute_policy_target

    state = create_initial_state()
    move_records: list[MoveRecord] = []

    while not state.is_terminal:
        if state.current_player == network_player:
            # Network's turn — use MCTS with greedy play to maximize wins
            root = run_mcts(state, network, num_simulations=num_simulations,
                            device=device)
            visits = root.get_visit_counts()
            legal_moves = get_legal_moves(state)
            policy_tgt = compute_policy_target(visits, len(legal_moves))

            # Greedy: temp=0.0 to maximize wins, producing clear value signal
            move = select_move(root, temperature=0.0)

            # Only record the NETWORK's moves — not the random player's
            record = MoveRecord(
                state_tensor=encode_state(state),
                policy_target=policy_tgt,
                opp_policy_target=torch.zeros(81),  # set below
                opp_legal_mask=torch.zeros(81),      # set below
                legal_mask=get_legal_move_mask(state),
                current_player=state.current_player,
            )
            move_records.append(record)
        else:
            # Random turns are deliberately not training examples.  Their final
            # outcomes estimate a random continuation policy, not the strong-play
            # value that MCTS expects for the player to move.
            legal_moves = get_legal_moves(state)
            move = random.choice(legal_moves)

        # Apply move
        rules = import_module('01_game.rules')
        state = rules.apply_move(state, move)

    # Opponent policy targets: for consecutive network moves, we can't set
    # meaningful opp targets since the random player's moves are not MCTS.
    # Leave them as zeros with zero mask (the loss will ignore them).
    # This is correct — we don't want to teach the opponent head random play.
    if move_records:
        for rec in move_records:
            rec.opp_policy_target = torch.zeros(81)
            rec.opp_legal_mask = torch.zeros(81)

    # Compute value/score/ownership targets
    winner = state.winner if state.winner is not None else 0
    final_results = state.sub_board_results.copy()

    _finalize_move_targets(move_records, winner, final_results)

    return GameRecord(
        moves=move_records,
        winner=winner,
        game_length=len(move_records),
        final_sub_board_results=final_results,
    )


def play_vs_best_game(
    network: 'UltimateTTTNetwork',
    best_network: 'UltimateTTTNetwork',
    num_simulations: int = 200,
    device: str = 'cpu',
    network_player: int = 1,
) -> GameRecord:
    """Play one game: current network vs frozen best_model.

    The current network plays as player1 (trying to win).
    The best_model plays as player2 (strong baseline).
    This generates data where the network experiences LOSING,
    teaching the value head to recognize bad positions.

    Args:
        network: Current network (plays as player1).
        best_network: Frozen best_model.pt (plays as player2).
        num_simulations: MCTS simulations per move.
        device: Compute device.

    Returns:
        Complete GameRecord with value/score/ownership targets.
    """
    from importlib import import_module
    board_mod = import_module('01_game.board')
    rules_mod = import_module('01_game.rules')
    search_mod = import_module('03_mcts.search')
    policy_target_mod = import_module('03_mcts.policy_target')

    create_initial_state = board_mod.create_initial_state
    encode_state = board_mod.encode_state
    get_legal_moves = rules_mod.get_legal_moves
    get_legal_move_mask = rules_mod.get_legal_move_mask
    run_mcts = search_mod.run_mcts
    select_move = search_mod.select_move
    compute_policy_target = policy_target_mod.compute_policy_target

    state = create_initial_state()
    move_records: list[MoveRecord] = []

    while not state.is_terminal:
        # Choose network based on current player
        net = network if state.current_player == network_player else best_network

        # Run MCTS with some exploration (temperature 0.5) for diversity
        root = run_mcts(state, net, num_simulations=num_simulations,
                        dirichlet_epsilon=0.0, device=device)
        visits = root.get_visit_counts()
        legal_moves = get_legal_moves(state)
        policy_tgt = compute_policy_target(visits, len(legal_moves))

        # Use slightly exploratory temperature to get diverse data
        move = select_move(root, temperature=0.2)

        # Record this move
        record = MoveRecord(
            state_tensor=encode_state(state),
            policy_target=policy_tgt,
            opp_policy_target=torch.zeros(81),
            opp_legal_mask=torch.zeros(81),
            legal_mask=get_legal_move_mask(state),
            current_player=state.current_player,
        )
        move_records.append(record)

        # Apply move
        rules = import_module('01_game.rules')
        state = rules.apply_move(state, move)

    # Set opponent policy targets
    for i in range(len(move_records) - 1):
        move_records[i].opp_policy_target = move_records[i + 1].policy_target
        move_records[i].opp_legal_mask = move_records[i + 1].legal_mask
    if move_records:
        move_records[-1].opp_policy_target = torch.zeros(81)
        move_records[-1].opp_legal_mask = torch.zeros(81)

    # Compute value/score/ownership targets
    winner = state.winner if state.winner is not None else 0
    final_results = state.sub_board_results.copy()

    _finalize_move_targets(move_records, winner, final_results)

    return GameRecord(
        moves=move_records,
        winner=winner,
        game_length=len(move_records),
        final_sub_board_results=final_results,
    )


def sample_random_prefix_state(
    min_prefix_plies: int = 4,
    max_prefix_plies: int = 20,
) -> 'GameState':
    """Create a diverse non-terminal state without assigning random-play value.

    Random moves are used only to reach a part of the state space that current
    self-play may not visit.  Targets are generated later by a strong MCTS
    continuation from this state, so random policy weakness never becomes the
    value label.
    """
    import random
    board_mod = import_module('01_game.board')
    rules_mod = import_module('01_game.rules')

    if min_prefix_plies < 0 or max_prefix_plies < min_prefix_plies:
        raise ValueError("invalid random-prefix ply range")

    for _ in range(100):
        state = board_mod.create_initial_state()
        target_plies = random.randint(min_prefix_plies, max_prefix_plies)
        for _ in range(target_plies):
            if state.is_terminal:
                break
            state = rules_mod.apply_move(
                state, random.choice(rules_mod.get_legal_moves(state))
            )
        if not state.is_terminal:
            return state

    raise RuntimeError("could not sample a non-terminal random-prefix state")


def play_reanalyzed_game(
    network: 'UltimateTTTNetwork',
    start_state: 'GameState',
    num_simulations: int = 800,
    device: str = 'cpu',
    temperature: float = 0.0,
) -> GameRecord:
    """Adjudicate an arbitrary state with strong MCTS on both sides.

    This is the entry point for random-prefix and future human-position
    reanalysis.  It records only the strong continuation and uses the eventual
    W/D/L result from each recorded player-to-move perspective.
    """
    board_mod = import_module('01_game.board')
    rules_mod = import_module('01_game.rules')
    search_mod = import_module('03_mcts.search')
    policy_target_mod = import_module('03_mcts.policy_target')

    state = start_state.copy()
    if state.is_terminal:
        raise ValueError("cannot reanalyse a terminal state")

    move_records: list[MoveRecord] = []
    while not state.is_terminal:
        root = search_mod.run_mcts(
            state,
            network,
            num_simulations=num_simulations,
            dirichlet_epsilon=0.0,
            device=device,
        )
        legal_moves = rules_mod.get_legal_moves(state)
        policy_tgt = policy_target_mod.compute_policy_target(
            root.get_visit_counts(), len(legal_moves)
        )
        move_records.append(
            MoveRecord(
                state_tensor=board_mod.encode_state(state),
                policy_target=policy_tgt,
                opp_policy_target=torch.zeros(81),
                opp_legal_mask=torch.zeros(81),
                legal_mask=rules_mod.get_legal_move_mask(state),
                current_player=state.current_player,
            )
        )
        move = search_mod.select_move(root, temperature=temperature)
        state = rules_mod.apply_move(state, move)

    for i in range(len(move_records) - 1):
        move_records[i].opp_policy_target = move_records[i + 1].policy_target
        move_records[i].opp_legal_mask = move_records[i + 1].legal_mask

    winner = state.winner if state.winner is not None else 0
    final_results = state.sub_board_results.copy()
    _finalize_move_targets(move_records, winner, final_results)
    return GameRecord(
        moves=move_records,
        winner=winner,
        game_length=len(move_records),
        final_sub_board_results=final_results,
    )


def play_random_prefix_reanalyzed_game(
    network: 'UltimateTTTNetwork',
    num_simulations: int = 800,
    device: str = 'cpu',
    min_prefix_plies: int = 4,
    max_prefix_plies: int = 20,
) -> GameRecord:
    """Sample a diverse state, then adjudicate it with strong play."""
    start_state = sample_random_prefix_state(min_prefix_plies, max_prefix_plies)
    return play_reanalyzed_game(
        network,
        start_state,
        num_simulations=num_simulations,
        device=device,
    )


def generate_mixed_batch(
    network: 'UltimateTTTNetwork',
    num_self_play: int,
    num_vs_random: int,
    num_vs_best: int,
    best_network: 'UltimateTTTNetwork',
    num_simulations: int = 800,
    temp_initial: float = 2.0,
    temp_decay_rate: float = 0.94,
    temp_min: float = 0.15,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.35,
    dirichlet_epsilon_boost: float = 0.55,
    dirichlet_boost_plies: int = 5,
    forced_opening_fraction: float = 0.0,
    forced_opening_pool: 'list[list[int]] | None' = None,
    num_reanalyzed: int = 0,
    reanalysis_min_prefix_plies: int = 4,
    reanalysis_max_prefix_plies: int = 20,
    self_play_batch_size: int = 1,
    device: str = 'cpu',
) -> list[GameRecord]:
    """Generate a mixed batch of self-play data.

    Mix of pure self-play, diverse-state strong reanalysis, and vs best.

    ``num_vs_random`` is retained as a deprecated configuration alias.  Those
    games are now converted to random-prefix strong reanalysis so old configs
    cannot silently reintroduce random-policy value targets.

    Forced opening diversity: instead of relying only on Dirichlet noise and
    temperature to naturally produce diverse openings, a configurable fraction
    of the pure self-play games are made to start from a pre-selected opening
    move, cycling round-robin through forced_opening_pool. This guarantees the
    network accumulates training data on those lines regardless of how skewed
    its own search priors have become for them -- a line no longer needs to
    "win" the search to get played and recorded. Only applied to pure
    self-play games (not vs-random/vs-best), so the mix still contains plenty
    of freely-chosen openings.

    Args:
        network: Current network.
        num_self_play: Number of pure self-play games.
        num_vs_random: Deprecated alias added to ``num_reanalyzed``.
        num_reanalyzed: Number of random-prefix states adjudicated by strong
            MCTS on both sides.
        num_vs_best: Number of games vs best opponent.
        best_network: The best network checkpoint (if available).
        num_simulations: MCTS simulations per move.
        dirichlet_epsilon_boost: Root epsilon during the early-ply boost window
            (see epsilon_for_ply() in 03_mcts/search.py).
        dirichlet_boost_plies: Number of opening plies the boost applies to.
            Set to 0 to disable and use flat dirichlet_epsilon throughout.
        forced_opening_fraction: Fraction (0.0-1.0) of num_self_play games per
            call that are forced to start from an opening in forced_opening_pool
            instead of letting MCTS choose freely. 0.0 = disabled (default).
        forced_opening_pool: List of forced-opening move-index lists to cycle
            through round-robin. Defaults to the 9 cells of the center
            sub-board (sub-board index 4) -- i.e. encode_move(4, 0..8) -- which
            covers the center-center opening plus its 8 neighbors, the exact
            line found to be almost never chosen by MCTS's own search.
        self_play_batch_size: Number of independent games whose leaf positions
            are evaluated together. A value greater than one activates batched
            MCTS for pure self-play and strong reanalysis games.
        device: Compute device.

    Returns:
        List of GameRecord objects (shuffled).
    """
    if self_play_batch_size <= 0:
        raise ValueError("self_play_batch_size must be positive")
    if forced_opening_pool is None:
        # Default pool: all 9 first moves into the center sub-board (index 4).
        # encode_move(sub_board, cell) = sub_board * 9 + cell.
        forced_opening_pool = [[4 * 9 + cell] for cell in range(9)]

    records: list[GameRecord] = []

    # 1. Pure self-play
    num_forced = int(round(num_self_play * forced_opening_fraction))
    forced_openings: list[list[int] | None] = []
    for i in range(num_self_play):
        forced_openings.append(
            forced_opening_pool[i % len(forced_opening_pool)]
            if i < num_forced and forced_opening_pool
            else None
        )
    if self_play_batch_size == 1:
        for forced in forced_openings:
            records.append(play_self_play_game(
                network, num_simulations, temp_initial, temp_decay_rate, temp_min,
                dirichlet_alpha, dirichlet_epsilon, device,
                dirichlet_epsilon_boost=dirichlet_epsilon_boost,
                dirichlet_boost_plies=dirichlet_boost_plies,
                forced_opening=forced,
            ))
    else:
        for start in range(0, num_self_play, self_play_batch_size):
            chunk_forced = forced_openings[start:start + self_play_batch_size]
            records.extend(play_self_play_games_batched(
                network,
                num_games=len(chunk_forced),
                num_simulations=num_simulations,
                temp_initial=temp_initial,
                temp_decay_rate=temp_decay_rate,
                temp_min=temp_min,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_epsilon=dirichlet_epsilon,
                device=device,
                dirichlet_epsilon_boost=dirichlet_epsilon_boost,
                dirichlet_boost_plies=dirichlet_boost_plies,
                forced_openings=chunk_forced,
            ))

    # 2. Diverse states, with strong play used for every recorded target.
    num_diverse = num_reanalyzed + num_vs_random
    if self_play_batch_size == 1:
        for _ in range(num_diverse):
            records.append(play_random_prefix_reanalyzed_game(
                network,
                num_simulations=num_simulations,
                device=device,
                min_prefix_plies=reanalysis_min_prefix_plies,
                max_prefix_plies=reanalysis_max_prefix_plies,
            ))
    else:
        starts = [
            sample_random_prefix_state(
                reanalysis_min_prefix_plies, reanalysis_max_prefix_plies
            )
            for _ in range(num_diverse)
        ]
        for start in range(0, num_diverse, self_play_batch_size):
            records.extend(play_reanalyzed_games_batched(
                network,
                starts[start:start + self_play_batch_size],
                num_simulations=num_simulations,
                device=device,
            ))
        
    # 3. Network vs best checkpoint
    if num_vs_best > 0 and best_network is not None:
        for i in range(num_vs_best):
            net_p = 1 if i % 2 == 0 else -1
            record = play_vs_best_game(network, best_network, num_simulations, device=device, network_player=net_p)
            records.append(record)

    # Shuffle records so the network sees mixed data in each batch
    import random
    random.shuffle(records)

    return records


def generate_self_play_batch(
    network: 'UltimateTTTNetwork',
    num_games: int,
    num_simulations: int = 800,
    device: str = 'cpu',
) -> list[GameRecord]:
    """Original self-play generator — kept for backward compatibility.

    Args:
        network: Current network.
        num_games: Number of games to play.
        num_simulations: MCTS simulations per move.
        device: Compute device.

    Returns:
        List of GameRecord objects.
    """
    records: list[GameRecord] = []
    for i in range(num_games):
        record = play_self_play_game(network, num_simulations, device=device)
        records.append(record)
    return records


if __name__ == "__main__":
    from importlib import import_module
    network_mod = import_module('02_network.network')

    print("=== Episode 11: Self-Play ===\n")

    # Use tiny network and few simulations for testing
    net = network_mod.UltimateTTTNetwork(channels=32, num_blocks=2)
    record = play_self_play_game(net, num_simulations=10, device='cpu')

    print(f"Game length: {record.game_length}")
    print(f"Winner: {record.winner}")
    print(f"Move records: {len(record.moves)}")

    # Verify targets
    for i, rec in enumerate(record.moves[:3]):
        print(f"  Move {i}: player={rec.current_player}, "
              f"value_target={rec.value_target:.1f}, "
              f"score_target={rec.score_target:.3f}")

    assert len(record.moves) == record.game_length
    assert record.moves[0].state_tensor.shape == (7, 9, 9)
    assert record.moves[0].policy_target.shape == (81,)
    assert abs(record.moves[0].policy_target.sum().item() - 1.0) < 1e-4

    print("\n=== Episode 11 PASSED ===")
