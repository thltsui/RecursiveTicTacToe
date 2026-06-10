# Monte Carlo Tree Search: How to Think When You Can't Calculate

In the last essay, we established that Ultimate Tic-Tac-Toe resists the kind of heuristic evaluation that makes classical chess engines work. The sending mechanic creates long-range dependencies that no hand-crafted scoring function has managed to capture reliably. The solution is to learn the evaluation function from self-play — but we haven't yet addressed a more fundamental question: how does the AI decide which move to make in the first place?

This essay is about Monte Carlo Tree Search (MCTS). It's the algorithm that connects the neural network to the actual game. Understanding it is the prerequisite for understanding why the whole system works.

---

## The Problem: You Can't Search the Whole Tree

Consider what a naive approach looks like. You're at a game state. You have, say, 9 legal moves (you've been sent to a specific sub-board). You could try all 9, then for each of those try all the resulting moves, and so on, until you reach a terminal state. The move leading to the best outcome wins.

This is minimax search, and it's exactly how chess engines work — except they can't search to the terminal state either, so they search to some fixed depth and evaluate the leaves with a heuristic.

Two problems:

**Problem 1: The tree is too large.** With a branching factor of ~9 and games lasting 30–60 moves, searching to depth 10 means evaluating 9^10 ≈ 3.5 billion positions. That's not happening in any reasonable time budget.

**Problem 2: You don't have a good evaluation function.** Even if you could search to depth 10, you'd need something to score the leaf positions. As established in the last essay, building that scoring function for Ultimate TTT is genuinely hard. You don't have it at the start of training, and you can't build it by hand.

So what do you do when the tree is too large to search exhaustively, and you don't have a good leaf evaluator?

You do what a thoughtful human does: you spend your time thinking about the *promising* moves, not all moves equally. And you make a judgment call about positions rather than simulating to the end.

MCTS is the formalization of this idea.

---

## The Four Phases

MCTS builds a partial game tree incrementally, one simulation at a time. Each simulation consists of four phases: **Select**, **Expand**, **Evaluate**, and **Backup**. Let's walk through each one with a concrete example.

Imagine we're 15 moves into a game. You've been sent to sub-board 3 (the middle-left). There are 7 legal moves in that sub-board. The MCTS tree so far has this root node plus some partially-explored children from previous simulations.

---

### Phase 1: Select

Starting from the root, we traverse the existing tree by always picking the child node that looks most promising. "Promising" is defined by a formula we'll examine shortly — for now, think of it as "the move we think is good, adjusted for how underexplored it is."

We keep descending until we reach a **leaf node** — a node that hasn't been expanded yet. In concrete terms: we're traversing the already-built portion of the tree, following the best path we've found so far.

In our example: the root has 7 children (one per legal move in sub-board 3). Previous simulations have visited some of them more than others. Selection might follow the path: root → move at cell 4 → move at cell 7 → [leaf node]. We stop here because cell 7's subtree hasn't been explored yet.

```python
def _select(node: MCTSNode, c_puct: float) -> MCTSNode:
    while not node.is_leaf() and not node.is_terminal:
        _, node = node.select_child(c_puct)
    return node
```

Simple. Four lines. The interesting logic is all in `select_child`, which we'll get to.

---

### Phase 2: Expand

At the leaf node, we call the neural network. It returns two things: a **policy** (a probability distribution over the 81 possible moves, most of which are illegal right now) and a **value** (its estimate of who's winning from this position).

We use the policy to create child nodes for all legal moves at this state, storing the network's prior probability for each. This "expansion" is what grows the tree.

```python
def _expand_and_evaluate(node, network, device) -> float:
    if node.is_terminal:
        # No need to expand — return actual game result
        if node.state.winner is None or node.state.winner == 0:
            return 0.0
        return float(node.state.winner * node.state.current_player)

    net_output = network.predict(node.state, device=device)
    legal_moves = get_legal_moves(node.state)
    legal_mask = get_legal_move_mask(node.state)
    probs = apply_legal_mask(net_output.policy_logits, legal_mask)

    node.expand(probs, legal_moves)
    return net_output.win_value.item()
```

Note what's not happening here: **no random rollout**. The original MCTS (as used in pre-AlphaGo computer Go) would simulate a random game from this leaf node to the end and use the outcome as the value estimate. AlphaZero replaced this with a direct neural network evaluation, which is faster and far more accurate once the network is trained. This is sometimes called "AlphaZero-style MCTS."

---

### Phase 3: Evaluate

This happens inside `_expand_and_evaluate` above — the return value from the network's `win_value` output is the evaluation. No separate phase needed in the code. The network is simultaneously computing the prior probabilities (for expansion) and the position value (for backup).

From the network's perspective: it sees the (7, 9, 9) encoded board tensor and returns a number in [−1, 1] representing how favorable this position looks for the current player. We take that number at face value and propagate it back up.

---

### Phase 4: Backup

The value from the leaf node gets propagated back up through all the ancestors to the root, updating the statistics stored at each edge.

```python
def backup(self, value: float) -> None:
    self.visit_count += 1

    if self.parent is not None and self.move_from_parent is not None:
        move = self.move_from_parent
        self.parent.N[move] = self.parent.N.get(move, 0) + 1
        # The value is from THIS node's perspective.
        # To the parent, the value of this move is -value.
        self.parent.W[move] = self.parent.W.get(move, 0.0) - value
        n = self.parent.N[move]
        self.parent.Q[move] = self.parent.W[move] / n

        # Negate value when going up — parent's perspective is opposite
        self.parent.backup(-value)
```

The negation is crucial. If the leaf node says "I'm winning" (+0.8), that's good news for the player who moved to get there, but bad news for their opponent — the parent node's player. So we negate the value at each level as we go up. This is what makes MCTS work correctly for two-player zero-sum games: both players' interests are automatically handled by the sign flip.

After backup, every node on the path from root to leaf has updated visit counts (N), cumulative values (W), and mean values (Q = W/N) for the edges that were traversed.

---

## The PUCT Formula: How Selection Works

Now we can tackle the formula that governs Selection. During the selection phase, at each node, we need to pick which child to visit. The score for each action is:

$$\text{PUCT}(s, a) = Q(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}$$

where:
- **Q(s, a)** is the mean value observed so far for action *a* from state *s* — the exploitation term
- **P(s, a)** is the network's prior probability for action *a* — the network's initial guess about how good this move is
- **N(s)** is the total number of visits to the parent node *s*
- **N(s, a)** is the number of times we've taken action *a* specifically
- **c_puct** is an exploration constant (we use 1.0)

In code:

```python
def puct_score(self, move_idx: int, c_puct: float = 1.0) -> float:
    q_value = self.Q.get(move_idx, 0.0)
    prior = self.P.get(move_idx, 0.0)
    n_action = self.N.get(move_idx, 0)
    n_parent = self.visit_count

    exploration = c_puct * prior * math.sqrt(n_parent) / (1 + n_action)
    return q_value + exploration
```

Let's understand each term intuitively.

**Q(s, a)** is exploitation: it tracks the empirical quality of action *a*. If we've taken action *a* 50 times and it led to winning positions 80% of the time, Q will be high. We want to keep exploiting good moves.

**The exploration term** has three factors:
- `P(s, a)`: the network's prior. Moves the network thinks are good get a boost even if unvisited. This is what focuses the search — without it, we'd explore every legal move equally regardless of how obviously bad most are.
- `sqrt(N(s))`: grows with the parent's visit count. As we visit the parent more, the bonus for exploring children increases proportionally.
- `1 / (1 + N(s, a))`: shrinks as we visit action *a* more. An unvisited move (N=0) gets score `P * sqrt(N) / 1`. After one visit, it halves. The denominator's "+1" prevents division by zero.

The combined effect: initially, the exploration term dominates and we visit moves roughly in order of the prior probability. As we accumulate visits, the Q values start to matter more, and the search converges on the moves that are actually good.

This is the core insight of PUCT: it automates the exploration-exploitation tradeoff that you'd otherwise have to hand-tune. The prior keeps you from wasting simulations on obviously bad moves, and the visit count decays ensure you eventually revisit promising moves rather than getting stuck in a local optimum.

---

## Why MCTS Needs the Neural Network

Here's a thought experiment. What happens if we run MCTS with a randomly initialized network — one where the weights are set to small random values before any training?

At iteration 0, the policy head outputs roughly uniform probabilities over all 81 moves (minus the illegal ones). The value head outputs something close to 0 for every position. Let's trace through what this does to MCTS:

**Selection**: Since all priors are ~uniform (say, 1/9 for 9 legal moves), the exploration term becomes `c_puct * (1/9) * sqrt(N) / (1 + N(a))`. With no signal from the prior, selection is essentially random. We're picking moves roughly proportional to how undervisited they are.

**Evaluation**: The value head returns ~0 everywhere. Backup propagates 0 back up. After many simulations, all Q values converge toward 0, and selection remains random.

**Result**: With 800 simulations over 9 moves, you get roughly 89 visits per move. The visit distribution is almost uniform, which means the policy target — the training signal — is almost uniform. The network has learned nothing, so MCTS provides no useful signal, so the network can't learn. A perfect failure mode.

After training, the picture is completely different. The policy head has learned that in the current game state, two specific moves are worth seriously considering, and the other seven are likely mistakes. It might assign 40% probability to cell 4, 35% to cell 7, and spread the remaining 25% across the other moves. Now MCTS has real guidance:

- Cell 4 and cell 7 get lots of visits quickly (high prior, fast initial exploration)
- MCTS discovers through simulation that cell 4 leads to positions the value head rates at +0.6, while cell 7 leads to +0.3
- The 800 simulations converge: maybe 450 visits to cell 4, 280 to cell 7, and the rest scattered
- The visit distribution (450/800, 280/800, ...) becomes a sharper, more informative policy target

The neural network and MCTS are not independent components. **MCTS is only useful when the prior P(s,a) has signal.** Without a trained prior, you're running 800 random simulations. With a strong prior, you're running 800 focused simulations that concentrate on the moves that actually matter.

This is why AlphaZero needs to iterate: early in training, MCTS is near-random, produces near-uniform policy targets, and the network barely learns. But as the network improves even slightly, MCTS becomes slightly less random, produces slightly better targets, and the network improves a bit more. The feedback loop is slow at first and accelerates as the prior gets sharper.

---

## Temperature: How You Actually Pick a Move

After running 800 simulations, you have visit counts for each move. How do you turn those into an actual move selection?

The naïve answer is "pick the most visited move." That's correct during evaluation, but wrong during training. Here's why.

During self-play training, you want diversity — you want to explore different parts of the game tree so you don't overfit to one particular style of play. If every self-play game plays out identically (always greedy), you collect a narrow distribution of positions and the network doesn't learn to handle the variety of situations it might encounter.

The solution is temperature-controlled sampling. Given visit counts N(a) for each move a, we sample from:

$$\pi(a) = \frac{N(a)^{1/T}}{\sum_{a'} N(a')^{1/T}}$$

where T is the temperature parameter.

```python
def select_move(root: MCTSNode, temperature: float = 1.0) -> int:
    visits = root.get_visit_counts()
    moves = list(visits.keys())
    counts = np.array([visits[m] for m in moves], dtype=np.float64)

    if temperature == 0.0:
        best_idx = np.argmax(counts)
        return moves[best_idx]

    adjusted = counts ** (1.0 / temperature)
    total = adjusted.sum()
    probs = adjusted / total
    return int(np.random.choice(moves, p=probs))
```

At **T = 0** (or effectively 0): `argmax` of visit counts. Completely deterministic. You always pick the most visited move. Used during arena evaluation where you want to see the network's actual best play.

At **T = 1.0**: sample proportional to visit counts. If MCTS gave 450 visits to move A and 350 to move B, you'll pick A about 56% of the time and B about 44%. Significant randomness remains.

At **T = 1.25**: the exponent `1/1.25 = 0.8` compresses the visit count differences. A move with 450 visits and one with 350 visits get closer to equal probability than at T=1. More exploration, more diverse games.

The schedule during self-play: use T = 1.25 for the first ~30 moves of the game (the opening and early midgame, where position diversity matters most), then switch to T = 0 (greedy) for the late game. The opening explores; the endgame plays to win.

Why 30 moves as the threshold? It's roughly the point at which the game has enough structure that the "correct" continuation is less ambiguous. In the opening, many lines are roughly equivalent and variety is valuable. By move 30, you've usually committed to a strategy and you want the network to learn to execute it cleanly.

---

## Dirichlet Noise: Preventing the Fixation Problem

There's a subtle failure mode in MCTS that pure temperature doesn't solve.

Suppose the network strongly believes one move is best — say, 90% prior probability on cell 4. MCTS will allocate most of its 800 simulations to cell 4. The other 8 moves get ~11 simulations each, spread uniformly. The visit distribution after 800 simulations might be 720 on cell 4, 10 on each other move.

The problem: this might be wrong. The network's prior could be miscalibrated, especially early in training. By allocating almost all simulations to one path, we've essentially committed to the network's initial judgment without genuinely interrogating the alternatives. In the worst case, the network develops a blind spot: a certain class of positions always gets one move heavily visited, that move is reinforced in training, the prior gets stronger, the blind spot deepens.

The fix is **Dirichlet noise** — a random perturbation added to the prior at the root node:

$$P_{\text{noisy}}(a) = (1 - \varepsilon) \cdot P(a) + \varepsilon \cdot \text{Dir}(\alpha)$$

where Dir(α) is a sample from the Dirichlet distribution with concentration parameter α.

```python
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
```

We use α = 0.3 and ε = 0.35. The ε = 0.35 means 35% of the prior comes from the noise distribution and 65% from the network. This is a substantial perturbation — it ensures that even when the network is confident, MCTS still allocates meaningful simulations to the alternatives.

**Critical detail**: the noise is only added at the root. Deeper nodes in the tree are not perturbed. This is intentional. We want to inject diversity at the *decision point* — the move we're actually about to play — without corrupting the downstream evaluation of what follows from that move. If we added noise at every node, we'd be simulating random play everywhere, which defeats the purpose of having a prior at all.

The Dirichlet distribution is the right choice here because it produces samples that sum to 1 (like a probability distribution over moves) and has a natural "concentration" interpretation. Small α = 0.3 produces spiky samples where most of the noise mass is on a few moves, which is what you want: the noise doesn't uniformly boost all moves, it unpredictably boosts a subset, forcing the search to seriously consider that subset this particular game.

---

## How MCTS Generates Training Data

Now we can see the feedback loop that makes the whole system work.

During self-play, every move in every game goes like this:
1. Run MCTS from the current position (800 simulations, Dirichlet noise at root)
2. Record the visit distribution π(a) = N(a) / Σ N(a') as the **policy target** for this position
3. Select the actual move using temperature sampling
4. After the game ends, retroactively label every position in the game with **z** = +1 (current player won) or −1 (lost) or 0 (draw)

These records — (state, π, z) for each position — go into a replay buffer. The network is then trained to minimize:

$$\mathcal{L} = \underbrace{-\sum_a \pi(a) \log p(a)}_{\text{policy: match MCTS visits}} + \underbrace{(z - v)^2}_{\text{value: match game outcome}}$$

The policy loss pushes the network's prior toward the MCTS visit distribution. The value loss pushes the network's position estimate toward the actual game outcome.

Why does this work? MCTS is a **better policy than the network's raw prior**. It takes the network's prior as input, runs 800 simulations, and produces a refined estimate of which moves are actually best. The visit distribution encodes what MCTS learned about this particular position. By training the network to match MCTS, we're distilling MCTS's search process into the network's weights.

After training, the network's prior is closer to what MCTS would have recommended. So the next iteration of MCTS, using this improved prior, does a better job. Which produces better training targets. Which produces a better network. Which produces better MCTS. This is the self-play loop.

A few things to note about the policy target specifically. We prune very low-probability moves before creating the target:

```python
def compute_policy_target(
    visit_counts: dict[int, int],
    num_legal_moves: int,
    pruning_threshold: float = 0.0,
) -> torch.Tensor:
    target = torch.zeros(81, dtype=torch.float32)
    total_visits = sum(visit_counts.values())

    for move, count in visit_counts.items():
        prob = count / total_visits
        if prob >= pruning_threshold:
            target[move] = prob

    # Renormalize after pruning
    total = target.sum()
    if total > 0:
        target = target / total

    return target
```

Moves that received only 1 or 2 visits out of 800 simulations weren't seriously considered — they were visited mainly due to the Dirichlet noise forcing minimal exploration. Training the network to assign nonzero probability to those moves would be training it to imitate noise. Pruning removes them from the target and renormalizes the remaining distribution.

---

## What MCTS Looks Like in Practice

To put numbers on this: with 800 simulations and c_puct = 1.0, a well-trained network will concentrate its visits dramatically on 2–3 moves, allocate moderate attention to a few more, and barely touch the rest.

Here's what you might see from a strong network at a critical position:

```
Top 5 moves by visits (800 total):
  cell 4 → 391 visits (49%)   [PUCT drove here early, Q confirmed it]
  cell 7 → 184 visits (23%)   [second-best, decent Q]
  cell 1 → 102 visits (13%)   [explored seriously, turned out ok]
  cell 6 →  82 visits (10%)   [explored, less impressive Q]
  cell 2 →  28 visits  (4%)   [mostly Dirichlet-forced exploration]
```

The visit distribution is the policy target. The network will be trained to assign ~49% probability to cell 4, ~23% to cell 7, and so on. By the time the network has seen thousands of positions like this one, it will have internalized that "in this board configuration, cell 4 is strongly preferred."

At iteration 0, with random weights, these numbers look like 95/800 per move — almost uniform. By iteration 50, the concentrations are sharper. By iteration 200, a position like this might have 600+ visits on the top move, because the prior is strong enough that MCTS barely considers the alternatives.

The sharpening of the visit distribution over training is a concrete measure of the network learning. An entropy of H = log(9) ≈ 2.2 nats is random; a fully confident selection has H = 0. Watching policy entropy drop from ~2.2 toward ~0.5–1.0 over training is watching the system learn to play.

---

## Putting It Together

MCTS is the algorithm that bridges the gap between "I have a neural network that knows something about positions" and "I can make good moves." Without MCTS, the network's raw policy output is your move — whatever the network learned during training. With MCTS, you're running 800 mini-games in your head before each move, using the network as a guide, and making a decision based on the accumulated evidence.

The four phases — Select, Expand, Evaluate, Backup — are the engine. The PUCT formula is the steering wheel. Dirichlet noise is the insurance policy against fixation. Temperature is the dial between exploration and exploitation. And the visit distribution that emerges is simultaneously the move selection signal and the training target that will make the next generation of the network slightly better.

In the next essay, we'll look at the neural network itself: the (7, 9, 9) input encoding, the residual trunk with global pooling, and the five-component loss function that trains it. The architecture is where the game-specific design decisions live.
