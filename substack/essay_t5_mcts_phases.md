# T5: Inside the MCTS Loop — How AlphaZero Thinks

*In T3, we saw that the explore/exploit trade-off governs which branches of a game tree are worth searching. In T4, we derived the PUCT formula that operationalises this trade-off. This essay shows the full loop: how MCTS actually builds a search tree, one simulation at a time, and why the result is a move selection strategy that is strictly better than the neural network's raw output.*

---

## The Problem

You're 15 moves into a game of Ultimate Tic-Tac-Toe. You have 7 legal moves. In principle, you could enumerate every possible continuation — but with a branching factor of ~9 and games lasting 30–60 moves, searching to depth 10 means evaluating 9¹⁰ ≈ 3.5 billion positions. That isn't happening in any real-time budget.

Even if it were, you'd still need something to score the leaf positions. And as we established earlier, building a reliable heuristic evaluator for Ultimate TTT is genuinely hard. You can't hand-craft one from scratch.

So what do you do? You do what a thoughtful human does: you spend your thinking time on the *promising* moves, not all moves equally. And you make a judgment call about positions rather than simulating every game to the end.

MCTS is the formalisation of this idea.

---

## Four Phases, One Simulation

MCTS builds a partial game tree incrementally, one simulation at a time. Each simulation follows four phases: **Select**, **Expand**, **Evaluate**, and **Backup**.

### Select

Starting from the root (the current game position), we traverse the existing tree by picking the most promising child at each node. "Most promising" is defined by the PUCT formula — more on that in a moment. We keep descending until we reach a **leaf**: a node that hasn't been expanded yet.

In concrete terms: we're moving through the already-built portion of the tree, following the best path found by previous simulations.

### Expand

At the leaf, we call the neural network. It returns two things simultaneously:

A **policy** — a probability distribution over all legal moves from this position, reflecting the network's prior beliefs about which moves are worth exploring.

A **value** — a number in [−1, +1] estimating how favourable this position is for the current player.

We add the leaf node to the tree, storing the network's prior probabilities as the starting weights for its children.

Note what is *not* happening: no random rollout. Classic MCTS (the kind used before AlphaGo) would simulate a random game from the leaf to the end and use the outcome as the value estimate. AlphaZero replaced this with a direct neural network evaluation — faster, and far more accurate once the network is trained. The neural network is the oracle.

### Evaluate

The value returned by the network during Expand is already our leaf evaluation. No additional step needed. The network simultaneously handles both expansion (via policy) and evaluation (via value) in a single forward pass.

### Backup

The leaf's value propagates back up through all ancestor nodes to the root. At each edge on the path, we update:

**N(s, a)** — the visit count: how many simulations have passed through this edge.

**W(s, a)** — the total accumulated value: the sum of leaf values from all simulations through this edge.

**Q(s, a) = W(s, a) / N(s, a)** — the mean value: the average quality of this action across all simulations that explored it.

One crucial detail: the value is **negated** at each level as it propagates up. If the leaf says "I'm winning (+0.8)," that's good news for the player who chose to move there — but bad news for their opponent, who is the decision-maker one level higher. The sign flip at every backup step automatically handles two-player zero-sum structure. AlphaZero never needs separate tables for White and Black.

After backup, the path from root to leaf has richer statistics. The next simulation will use these updated counts and values when deciding where to go.

---

## PUCT: The Steering Wheel

The formula that governs selection at every node:

$$\text{PUCT}(s, a) = Q(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s,a)}$$

**Q(s, a)** is exploitation: the empirical mean value of action a. High Q means this branch has consistently produced good simulations.

**The exploration term** has three factors:
- **P(s, a)**: the network's prior probability. Moves the network likes get a boost, even before being visited — this is what focuses search away from obviously bad moves.
- **√N(s)**: grows with total visits to the parent. The more we've explored from this node, the larger the exploration bonus.
- **1 / (1 + N(s, a))**: shrinks as action a gets visited more. An unvisited move gets full bonus; after many visits, the bonus collapses and Q dominates.

The combined effect: early in search, the prior drives exploration — we visit moves roughly in order of the network's confidence. As simulations accumulate, Q values sharpen and the search converges on the moves that are genuinely best, not just what the network initially thought.

---

## Why MCTS Needs the Neural Network (and Vice Versa)

Here's a thought experiment. Run MCTS with a randomly initialised network — weights set to small random values before any training.

The policy head outputs nearly uniform probabilities across all legal moves. The value head returns something close to 0 everywhere. With a uniform prior, PUCT degenerates to "visit whichever move has been explored least." After 800 simulations over 9 legal moves, you get roughly 89 visits per move — almost uniform. The training target derived from these visits is also almost uniform. The network learns nothing, so the next round of MCTS provides no better signal. A perfect deadlock.

After training, the picture is completely different. The network might assign 40% probability to one move and 35% to another, with the remaining 25% spread across the rest. MCTS immediately focuses there: early simulations concentrate on those two moves. If they both turn out to be good (the value head confirms it), visits pile up, and the final visit distribution is sharp: maybe 450 and 280 out of 800, versus 10 each for the rest.

That sharp distribution is a better training target than anything MCTS can produce with random weights. The network learns from it, becomes slightly better at identifying good moves, and the next MCTS round is slightly more focused. The feedback loop is slow at first and accelerates as the prior sharpens.

This mutual dependence — MCTS needs a good prior, the prior is trained on MCTS outputs — is the fundamental design insight of AlphaZero. Neither component works alone.

---

*Next: T6 — Temperature, Dirichlet Noise, and the Training Loop. We look at how move selection is actually randomised during training, how Dirichlet noise prevents the search from fixating on the network's initial beliefs, and how (state, π, z) triples are used to train the network to imitate — and improve on — its own search.*
