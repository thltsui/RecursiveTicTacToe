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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


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
        value_target: +1 if winner, -1 if loser, 0 if draw. Set after game ends.
        score_target: Final score margin from current player's perspective. Set after game.
        ownership_target: Sub-board ownership from current player's perspective. Shape (9,).
    """
    state_tensor: torch.Tensor          # (7, 9, 9)
    policy_target: torch.Tensor         # (81,)
    opp_policy_target: torch.Tensor     # (81,)
    opp_legal_mask: torch.Tensor        # (81,)
    legal_mask: torch.Tensor            # (81,)
    current_player: int
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

    for rec in move_records:
        cp = rec.current_player
        # Value target: +1 if current player won, -1 if lost, -0.5 if draw
        # Anti-draw shaping: punishes both players for a draw to force risk-taking
        rec.value_target = float(winner * cp) if winner != 0 else -0.5

        # Score target: (own_sub_boards - opp_sub_boards) / 9
        own_wins = np.sum(final_results == cp)
        opp_wins = np.sum(final_results == -cp)
        rec.score_target = float(own_wins - opp_wins) / 9.0

        # Ownership target: binary — did current player win each sub-board?
        rec.ownership_target = torch.tensor(
            [(1.0 if final_results[i] == cp else 0.0) for i in range(9)],
            dtype=torch.float32
        )

    return GameRecord(
        moves=move_records,
        winner=winner,
        game_length=len(move_records),
        final_sub_board_results=final_results,
    )


def play_random_vs_random_game() -> GameRecord:
    """Play one complete game with both sides sampling uniformly from legal moves.

    No network, no MCTS. Both players pick moves uniformly at random.
    Every position from both players is recorded.

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

    for rec in move_records:
        cp = rec.current_player
        rec.value_target = float(winner * cp) if winner != 0 else -0.5
        own_wins = np.sum(final_results == cp)
        opp_wins = np.sum(final_results == -cp)
        rec.score_target = float(own_wins - opp_wins) / 9.0
        rec.ownership_target = torch.tensor(
            [(1.0 if final_results[i] == cp else 0.0) for i in range(9)],
            dtype=torch.float32
        )

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
    """Play one game: network (player1) vs random player.

    Random player samples uniformly from legal moves (no MCTS).
    This generates data where the network can LEARN to win,
    breaking the self-play Nash equilibrium of always drawing.

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
            # Random player's turn — play randomly and record for the Value Head,
            # but zero out the policy target and legal mask so the Policy Head ignores it.
            legal_moves = get_legal_moves(state)
            move = random.choice(legal_moves)
            
            legal_mask = get_legal_move_mask(state)
            record = MoveRecord(
                state_tensor=encode_state(state),
                policy_target=torch.zeros(81),      # Ignore policy
                opp_policy_target=torch.zeros(81),  
                opp_legal_mask=torch.zeros(81),     
                legal_mask=torch.zeros(81),         # Zero mask to ignore in policy loss
                current_player=state.current_player,
            )
            move_records.append(record)

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

    for rec in move_records:
        cp = rec.current_player
        # From the network's perspective, we still want +1 if it won
        # wait, value_target is from current_player's perspective.
        rec.value_target = float(winner * cp) if winner != 0 else 0.0
        own_wins = np.sum(final_results == cp)
        opp_wins = np.sum(final_results == -cp)
        rec.score_target = float(own_wins - opp_wins) / 9.0
        rec.ownership_target = torch.tensor(
            [(1.0 if final_results[i] == cp else 0.0) for i in range(9)],
            dtype=torch.float32
        )

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

    for rec in move_records:
        cp = rec.current_player
        rec.value_target = float(winner * cp) if winner != 0 else 0.0
        own_wins = np.sum(final_results == cp)
        opp_wins = np.sum(final_results == -cp)
        rec.score_target = float(own_wins - opp_wins) / 9.0
        rec.ownership_target = torch.tensor(
            [(1.0 if final_results[i] == cp else 0.0) for i in range(9)],
            dtype=torch.float32
        )

    return GameRecord(
        moves=move_records,
        winner=winner,
        game_length=len(move_records),
        final_sub_board_results=final_results,
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
    device: str = 'cpu',
) -> list[GameRecord]:
    """Generate a mixed batch of self-play data using a Vectorized Batched MCTS engine."""
    import random
    from importlib import import_module
    from tqdm import tqdm
    
    board_mod = import_module('01_game.board')
    rules_mod = import_module('01_game.rules')
    search_mod = import_module('03_mcts.search')
    policy_target_mod = import_module('03_mcts.policy_target')
    
    create_initial_state = board_mod.create_initial_state
    encode_state = board_mod.encode_state
    get_legal_moves = rules_mod.get_legal_moves
    get_legal_move_mask = rules_mod.get_legal_move_mask
    run_mcts_batched = search_mod.run_mcts_batched
    select_move = search_mod.select_move
    epsilon_for_ply = search_mod.epsilon_for_ply
    compute_policy_target = policy_target_mod.compute_policy_target

    if forced_opening_pool is None:
        forced_opening_pool = [[4 * 9 + cell] for cell in range(9)]

    def eval_current(batch_tensor):
        with torch.no_grad():
            out = network(batch_tensor.to(device))
            return out.policy_logits, out.win_value, out.ownership

    def eval_best(batch_tensor):
        if best_network is None:
            raise ValueError("best_network must not be None when playing vs_best")
        with torch.no_grad():
            out = best_network(batch_tensor.to(device))
            return out.policy_logits, out.win_value, out.ownership

    class ActiveGame:
        def __init__(self, game_type: str, forced: list[int] = None, network_player: int = 1):
            self.game_type = game_type
            self.state = create_initial_state()
            self.move_records = []
            self.forced = forced
            self.network_player = network_player

        def finish(self) -> GameRecord:
            for i in range(len(self.move_records) - 1):
                self.move_records[i].opp_policy_target = self.move_records[i + 1].policy_target
                self.move_records[i].opp_legal_mask = self.move_records[i + 1].legal_mask
            
            if self.move_records:
                self.move_records[-1].opp_policy_target = torch.zeros(81)
                self.move_records[-1].opp_legal_mask = torch.zeros(81)
                
                if self.game_type != 'self_play':
                    for rec in self.move_records:
                        rec.opp_policy_target = torch.zeros(81)
                        rec.opp_legal_mask = torch.zeros(81)

            winner = self.state.winner if self.state.winner is not None else 0
            final_results = self.state.sub_board_results.copy()

            for rec in self.move_records:
                cp = rec.current_player
                if self.game_type == 'self_play':
                    rec.value_target = float(winner * cp) if winner != 0 else -0.5
                else:
                    rec.value_target = float(winner * cp) if winner != 0 else 0.0
                    
                own_wins = np.sum(final_results == cp)
                opp_wins = np.sum(final_results == -cp)
                rec.score_target = float(own_wins - opp_wins) / 9.0
                rec.ownership_target = torch.tensor(
                    [(1.0 if final_results[i] == cp else 0.0) for i in range(9)],
                    dtype=torch.float32
                )

            return GameRecord(
                moves=self.move_records,
                winner=winner,
                game_length=len(self.move_records),
                final_sub_board_results=final_results,
            )

    total_games = num_self_play + num_vs_random + num_vs_best
    records = []
    pending_games = []
    
    num_forced = int(round(num_self_play * forced_opening_fraction))
    for i in range(num_self_play):
        forced = forced_opening_pool[i % len(forced_opening_pool)] if (i < num_forced and forced_opening_pool) else None
        pending_games.append(('self_play', forced, 1))
        
    for i in range(num_vs_random):
        pending_games.append(('vs_random', None, 1 if i % 2 == 0 else -1))
        
    for i in range(num_vs_best):
        pending_games.append(('vs_best', None, 1 if i % 2 == 0 else -1))

    active_games = []
    batch_size = 16
    
    def fill_active_games():
        while len(active_games) < batch_size and pending_games:
            g_type, forced, net_p = pending_games.pop(0)
            active_games.append(ActiveGame(g_type, forced, net_p))

    fill_active_games()
    
    # Smooth progress bar tracking moves evaluated instead of full games
    pbar = tqdm(desc=f"Generating Games (0/{total_games}) - Moves Evaluated", unit=" moves")

    while active_games:
        current_net_games = []
        best_net_games = []
        
        for g in active_games:
            cp = g.state.current_player
            if g.game_type == 'self_play':
                current_net_games.append(g)
            elif g.game_type == 'vs_random':
                if cp == g.network_player:
                    current_net_games.append(g)
            elif g.game_type == 'vs_best':
                if cp == g.network_player:
                    current_net_games.append(g)
                else:
                    best_net_games.append(g)

        if current_net_games:
            epsilons = []
            for g in current_net_games:
                if g.game_type == 'self_play':
                    epsilons.append(epsilon_for_ply(
                        g.state.move_count,
                        base_epsilon=dirichlet_epsilon,
                        boosted_epsilon=dirichlet_epsilon_boost,
                        boost_plies=dirichlet_boost_plies,
                    ))
                else:
                    epsilons.append(0.0)
                    
            roots = run_mcts_batched(
                [g.state for g in current_net_games],
                eval_current,
                num_simulations=num_simulations,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_epsilon=epsilons
            )
            
            for g, root in zip(current_net_games, roots):
                legal_moves = get_legal_moves(g.state)
                visits = root.get_visit_counts()
                policy_tgt = compute_policy_target(visits, len(legal_moves))
                
                if g.game_type == 'self_play':
                    temp = temp_initial * (temp_decay_rate ** g.state.move_count)
                    if temp < temp_min:
                        temp = 0.0
                        
                    ply = g.state.move_count
                    if g.forced is not None and ply < len(g.forced):
                        move = g.forced[ply]
                        if move not in legal_moves:
                            move = select_move(root, temperature=temp)
                    else:
                        move = select_move(root, temperature=temp)
                        
                elif g.game_type == 'vs_random':
                    move = select_move(root, temperature=0.0)
                elif g.game_type == 'vs_best':
                    move = select_move(root, temperature=0.2)
                
                record = MoveRecord(
                    state_tensor=encode_state(g.state),
                    policy_target=policy_tgt,
                    opp_policy_target=torch.zeros(81),
                    opp_legal_mask=torch.zeros(81),
                    legal_mask=get_legal_move_mask(g.state),
                    current_player=g.state.current_player,
                )
                g.move_records.append(record)
                g.state = rules_mod.apply_move(g.state, move)
                
        if best_net_games:
            roots = run_mcts_batched(
                [g.state for g in best_net_games],
                eval_best,
                num_simulations=num_simulations,
                dirichlet_epsilon=0.0
            )
            for g, root in zip(best_net_games, roots):
                legal_moves = get_legal_moves(g.state)
                visits = root.get_visit_counts()
                policy_tgt = compute_policy_target(visits, len(legal_moves))
                
                move = select_move(root, temperature=0.2)
                
                record = MoveRecord(
                    state_tensor=encode_state(g.state),
                    policy_target=torch.zeros(81),
                    opp_policy_target=torch.zeros(81),
                    opp_legal_mask=torch.zeros(81),
                    legal_mask=torch.zeros(81),
                    current_player=g.state.current_player,
                )
                g.move_records.append(record)
                g.state = rules_mod.apply_move(g.state, move)

        random_games = [g for g in active_games if g.game_type == 'vs_random' and g.state.current_player != g.network_player]
        for g in random_games:
            legal_moves = get_legal_moves(g.state)
            move = random.choice(legal_moves)
            
            record = MoveRecord(
                state_tensor=encode_state(g.state),
                policy_target=torch.zeros(81),
                opp_policy_target=torch.zeros(81),
                opp_legal_mask=torch.zeros(81),
                legal_mask=torch.zeros(81),
                current_player=g.state.current_player,
            )
            g.move_records.append(record)
            g.state = rules_mod.apply_move(g.state, move)
            
        # Update progress bar continuously based on number of active games processed this tick
        pbar.update(len(active_games))

        surviving_games = []
        for g in active_games:
            if g.state.is_terminal:
                records.append(g.finish())
                pbar.set_description(f"Generating Games ({len(records)}/{total_games}) - Moves Evaluated")
            else:
                surviving_games.append(g)
                
        active_games = surviving_games
        fill_active_games()
        
    pbar.close()
    
    # Shuffle records so the network sees mixed data in each batch
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
