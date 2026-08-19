"""Episode 9 — Monte Carlo Tree Search — The Four Phases

Concept taught: The complete MCTS loop: Select -> Expand -> Evaluate -> Backup.
How the neural network guides the search. Dirichlet noise for exploration at root.
The temperature parameter for move selection.

The Four Phases:
    1. SELECT:   Start at root. Follow highest PUCT scores until reaching a leaf node.
    2. EXPAND:   Call network to get (policy, value). Create children for all legal moves.
    3. EVALUATE: Use network's value estimate (no rollouts — this is AlphaZero style).
    4. BACKUP:   Propagate value up through ancestors, negating at each level.
"""

from __future__ import annotations

import sys
import os
import math

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from .node import MCTSNode


def epsilon_for_ply(
    move_count: int,
    base_epsilon: float = 0.35,
    boosted_epsilon: float = 0.55,
    boost_plies: int = 5,
) -> float:
    """Root Dirichlet-noise schedule: boost exploration noise for the first few
    plies of the game, then decay to the normal epsilon for the rest.

    Debugging context: PUCT allocates simulation budget largely by prior.
    If the policy prior settles on a strong dislike of a particular opening
    before the value head is well-calibrated on it, that branch keeps
    receiving too few visits at the root to ever be seriously re-examined,
    and the early belief compounds instead of self-correcting (confirmed via
    replay-buffer analysis: a specific opening received nonzero search visits
    in only ~4% of late-training games, down from ~14% early in training).
    A larger root epsilon for the first few plies forces enough visits onto
    otherwise-starved branches that they get genuinely stress-tested by
    search while the network is still forming its early-game opinions.

    Args:
        move_count: Current ply (0-indexed) in the game.
        base_epsilon: Standard root Dirichlet epsilon used after the boost window.
        boosted_epsilon: Epsilon used for moves 0..boost_plies-1.
        boost_plies: Number of early plies over which the boost applies.
            Set to 0 to disable boosting entirely (always returns base_epsilon).

    Returns:
        The dirichlet_epsilon value to use for this move's MCTS root.
    """
    if boost_plies > 0 and move_count < boost_plies:
        return boosted_epsilon
    return base_epsilon


def run_mcts(
    root_state: 'GameState',
    network: 'UltimateTTTNetwork',
    num_simulations: int = 800,
    c_puct: float = 1.0,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.35,
    device: str = 'cpu',
) -> MCTSNode:
    """Run MCTS from the given state and return the root node with statistics.

    Dirichlet noise is added to the ROOT node's prior probabilities only.
    This encourages exploration during self-play. The formula is:
        P_noisy(a) = (1 - epsilon) * P(a) + epsilon * Dir(alpha)
        alpha = 0.3 is appropriate for Ultimate TTT (similar to chess)
        epsilon = 0.25 (KataGo default)

    Args:
        root_state: Starting game state.
        network: Neural network for evaluation.
        num_simulations: Number of MCTS simulations (default 800).
        c_puct: Exploration constant.
        dirichlet_alpha: Dirichlet concentration parameter for root noise.
        dirichlet_epsilon: Weight of Dirichlet noise (0 = no noise).
        device: Device for network inference.

    Returns:
        Root MCTSNode with visit statistics populated after all simulations.
    """
    from importlib import import_module
    rules_mod = import_module('01_game.rules')
    policy_head_mod = import_module('02_network.policy_head')

    get_legal_moves = rules_mod.get_legal_moves
    apply_legal_mask = policy_head_mod.apply_legal_mask
    get_legal_move_mask = rules_mod.get_legal_move_mask

    root = MCTSNode(state=root_state)

    # Expand root
    if not root.is_terminal:
        net_output = network.predict(root_state, device=device)
        legal_moves = get_legal_moves(root_state)
        legal_mask = get_legal_move_mask(root_state)
        probs = apply_legal_mask(net_output.policy_logits, legal_mask)

        # Add Dirichlet noise at root for exploration
        if dirichlet_epsilon > 0 and len(legal_moves) > 0:
            noise = np.random.dirichlet([dirichlet_alpha] * len(legal_moves))
            noisy_probs = probs.clone()
            for i, move in enumerate(legal_moves):
                noisy_probs[move] = (1 - dirichlet_epsilon) * probs[move].item() + \
                                     dirichlet_epsilon * noise[i]
            # Renormalize
            total = sum(noisy_probs[m].item() for m in legal_moves)
            if total > 0:
                for m in legal_moves:
                    noisy_probs[m] = noisy_probs[m] / total
            probs = noisy_probs

        root.expand(probs, legal_moves)

    # Run simulations
    for _ in range(num_simulations):
        # Phase 1: Select
        leaf = _select(root, c_puct)
        # Phase 2+3: Expand and Evaluate
        value, wdl = _expand_and_evaluate(leaf, network, device)

        # Phase 4: Backup
        _backup(leaf, value, wdl)

    return root


def select_move(root: MCTSNode, temperature: float = 1.0) -> int:
    """Select a move from MCTS results using temperature-controlled sampling.

    Temperature controls exploration vs exploitation in move selection:
        temperature = 1.0: Sample proportional to visit counts (exploration).
        temperature -> 0.0: Select move with highest visit count (exploitation).

    Formula for temperature > 0:
        pi(a) = N(a)^(1/temperature) / sum N(a')^(1/temperature)

    Args:
        root: Root node after MCTS has been run.
        temperature: Controls randomness. 0.0 = greedy, 1.0 = proportional.

    Returns:
        Selected move_idx (int, 0-80).
    """
    visits = root.get_visit_counts()
    if not visits:
        raise ValueError("No visits recorded — MCTS may not have been run")

    moves = list(visits.keys())
    counts = np.array([visits[m] for m in moves], dtype=np.float64)

    if temperature == 0.0:
        # Greedy: pick the most visited move
        best_idx = np.argmax(counts)
        return moves[best_idx]

    # Temperature-adjusted sampling
    adjusted = counts ** (1.0 / temperature)
    total = adjusted.sum()
    if total == 0:
        # Fallback to uniform
        return int(np.random.choice(moves))
    probs = adjusted / total
    return int(np.random.choice(moves, p=probs))


def _select(node: MCTSNode, c_puct: float) -> MCTSNode:
    """Phase 1: Traverse tree following best PUCT scores until leaf.

    Args:
        node: Starting node (usually root).
        c_puct: Exploration constant.

    Returns:
        Leaf node (not yet expanded) or terminal node.
    """
    while not node.is_leaf() and not node.is_terminal:
        _, node = node.select_child(c_puct)
    return node


def _expand_and_evaluate(
    node: MCTSNode,
    network: 'UltimateTTTNetwork',
    device: str,
) -> tuple[float, np.ndarray]:
    """Phase 2+3: Expand leaf node and return network value estimate.

    If node is terminal, return actual game outcome (+1, -1, or 0).
    Otherwise call network and expand with prior probabilities.

    Args:
        node: Leaf node to expand.
        network: Neural network.
        device: Compute device.

    Returns:
        Tuple of (value, wdl) from current player's perspective.
    """
    if node.is_terminal:
        # Return actual game result from the perspective of the current player
        # at this terminal node. Since the game ended on the previous player's move,
        # the winner is from the previous player's perspective.
        if node.state.winner is None or node.state.winner == 0:
            value = 0.0
            wdl = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            value = float(node.state.winner * node.state.current_player)
            if value > 0:
                wdl = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            else:
                wdl = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        return value, wdl

    from importlib import import_module
    rules_mod = import_module('01_game.rules')
    policy_head_mod = import_module('02_network.policy_head')

    get_legal_moves = rules_mod.get_legal_moves
    get_legal_move_mask = rules_mod.get_legal_move_mask
    apply_legal_mask = policy_head_mod.apply_legal_mask

    net_output = network.predict(node.state, device=device)
    legal_moves = get_legal_moves(node.state)
    legal_mask = get_legal_move_mask(node.state)
    probs = apply_legal_mask(net_output.policy_logits, legal_mask)

    node.expand(probs, legal_moves)

    # Return zero-sum scalar and W/D/L from the current player's perspective.
    return (
        net_output.win_value.item(),
        net_output.wdl_probs.detach().cpu().numpy().flatten(),
    )


def _backup(
    node: MCTSNode,
    value: float,
    wdl: np.ndarray | None = None,
) -> None:
    """Phase 4: Backpropagate value and W/D/L up the tree.

    Args:
        node: Leaf node where evaluation was performed.
        value: Value from the leaf node's current player's perspective.
        wdl: (3,) W/D/L probabilities from the leaf perspective.
    """
    node.backup(value, wdl)


if __name__ == "__main__":
    from importlib import import_module

    board_mod = import_module('01_game.board')
    network_mod = import_module('02_network.network')

    print("=== Episode 9: MCTS Search ===\n")

    state = board_mod.create_initial_state()
    net = network_mod.UltimateTTTNetwork(channels=32, num_blocks=2)  # Small for testing

    # Run MCTS with few simulations for speed
    root = run_mcts(state, net, num_simulations=50, device='cpu')

    print(f"Root visit count: {root.visit_count}")
    visits = root.get_visit_counts()
    print(f"Moves visited: {len(visits)}")
    print(f"Top 5 moves by visits: {sorted(visits.items(), key=lambda x: -x[1])[:5]}")

    # Select move with temperature
    move_hot = select_move(root, temperature=1.0)
    move_cold = select_move(root, temperature=0.0)
    print(f"Move (temp=1.0): {move_hot}")
    print(f"Move (temp=0.0): {move_cold}")

    # The greedy move should be the most visited
    most_visited = max(visits, key=visits.get)
    assert move_cold == most_visited, "Greedy should pick most visited"
    print("Greedy selection: PASSED")

    print("\n=== Episode 9 PASSED ===")
def _unpack_batched_eval(output):
    """Normalize the supported batched evaluator protocols.

    Preferred: return ``NetworkOutput``. A four-tuple ordered
    ``(policy_logits, win_values, ownership, wdl_probs)`` is also accepted.
    Scalar-only three-tuples are rejected because they cannot preserve draw
    probability.
    """
    if hasattr(output, 'policy_logits'):
        return (
            output.policy_logits,
            output.win_value,
            output.ownership,
            output.wdl_probs,
        )
    if isinstance(output, (tuple, list)) and len(output) == 4:
        return output
    raise ValueError(
        "eval_func must return NetworkOutput or "
        "(policy_logits, win_values, ownership, wdl_probs)"
    )


def run_mcts_batched(
    root_states,
    eval_func,
    num_simulations=1200,
    c_puct=1.5,
    dirichlet_alpha=0.3,
    dirichlet_epsilon=0.0,
):
    """Run one synchronized MCTS simulation per root per evaluator batch.

    This batches network leaf evaluation across independent games. Backup is
    still performed exactly once per selected leaf and uses the same zero-sum
    W/D/L perspective transform as :func:`run_mcts`.
    """
    from importlib import import_module

    rules_mod = import_module('01_game.rules')
    board_mod = import_module('01_game.board')
    policy_head_mod = import_module('02_network.policy_head')
    get_legal_moves = rules_mod.get_legal_moves
    get_legal_move_mask = rules_mod.get_legal_move_mask
    apply_legal_mask = policy_head_mod.apply_legal_mask

    roots = [MCTSNode(state=state) for state in root_states]
    if isinstance(dirichlet_epsilon, (float, int)):
        epsilons = [float(dirichlet_epsilon)] * len(roots)
    else:
        epsilons = list(dirichlet_epsilon)
        if len(epsilons) != len(roots):
            raise ValueError("dirichlet_epsilon must have one value per root")

    indexed_roots = [(index, root) for index, root in enumerate(roots) if not root.is_terminal]
    if indexed_roots:
        batch = torch.stack([board_mod.encode_state(root.state) for _, root in indexed_roots])
        policy_logits, _, _, _ = _unpack_batched_eval(eval_func(batch))
        for batch_index, (root_index, root) in enumerate(indexed_roots):
            legal_moves = get_legal_moves(root.state)
            legal_mask = get_legal_move_mask(root.state)
            probs = apply_legal_mask(policy_logits[batch_index], legal_mask).detach().cpu()
            epsilon = epsilons[root_index]
            if epsilon > 0 and legal_moves:
                noise = np.random.dirichlet([dirichlet_alpha] * len(legal_moves))
                for noise_index, move in enumerate(legal_moves):
                    probs[move] = (
                        (1.0 - epsilon) * probs[move].item()
                        + epsilon * noise[noise_index]
                    )
                total = sum(probs[move].item() for move in legal_moves)
                if total > 0:
                    probs /= total
            root.expand(probs, legal_moves)

    for _ in range(num_simulations):
        leaves = [_select(root, c_puct) for root in roots]
        evaluations: dict[int, tuple[float, np.ndarray]] = {}
        nonterminal = [
            (index, leaf) for index, leaf in enumerate(leaves) if not leaf.is_terminal
        ]

        if nonterminal:
            batch = torch.stack([board_mod.encode_state(leaf.state) for _, leaf in nonterminal])
            policy_logits, values, _, wdl_probs = _unpack_batched_eval(
                eval_func(batch)
            )
            for batch_index, (leaf_index, leaf) in enumerate(nonterminal):
                legal_moves = get_legal_moves(leaf.state)
                legal_mask = get_legal_move_mask(leaf.state)
                probs = apply_legal_mask(
                    policy_logits[batch_index], legal_mask
                ).detach().cpu()
                leaf.expand(probs, legal_moves)
                evaluations[leaf_index] = (
                    float(values[batch_index].item()),
                    wdl_probs[batch_index].detach().cpu().numpy(),
                )

        for leaf_index, leaf in enumerate(leaves):
            if leaf.is_terminal:
                winner = leaf.state.winner if leaf.state.winner is not None else 0
                value = float(winner * leaf.state.current_player)
                if value > 0:
                    wdl = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                elif value < 0:
                    wdl = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                else:
                    wdl = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                evaluations[leaf_index] = (
                    value,
                    wdl,
                )

            value, wdl = evaluations[leaf_index]
            leaf.backup(value, wdl)

    return roots
