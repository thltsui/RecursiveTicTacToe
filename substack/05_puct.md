# From Zero to AlphaZero: PUCT — How AlphaZero Weighs Curiosity, Evidence, and Intuition

*Last time we derived the UCB score from first principles: start with the observation that sample means are uncertain, apply Hoeffding's inequality to quantify that uncertainty, and you arrive at an exploration bonus that shrinks exactly as fast as your estimate tightens. This essay takes that formula and puts it inside a game tree.*

---

## The Same Problem, Bigger Space

The slot machine casino had K levers. A game tree has something more daunting: at each position, there are between 1 and 81 legal moves, and each of those moves leads to a new position with its own set of moves, branching millions of times before the game ends.

But if you squint at it, each node in the tree is just a bandit problem. You are sitting at some game position *s*, and you must choose which move *a* to explore next. The "arms" are the legal moves. The "payout" of pulling arm *a* is whatever value the game continuation turns out to have. You want to explore enough to find the best move, but not waste simulations on moves that are clearly bad.

UCB solved this for slot machines. AlphaZero adapts the same idea for trees, under the name **PUCT** (Predictor + Upper Confidence bound for Trees):

$$\text{PUCT}(s, a) = Q(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}$$

Every term in this formula has a direct counterpart in UCB. The differences are instructive.

---

## Q(s, a): The Sample Mean, Built from Simulations

In the bandit, the sample mean $\hat{\mu}_a$ was simple: pull arm *a* a few times, average the rewards, done. In the game tree, there are no direct rewards to observe — the game hasn't been played to completion.

Instead, each time a simulation passes through the branch (s, a), it eventually reaches an unexplored leaf node where the neural network evaluates the position and returns a value estimate between −1 (loss) and +1 (win). Q(s, a) is the running average of all those estimates:

$$Q(s, a) = \frac{1}{N(s, a)} \sum_{\text{simulations through } (s,a)} V_\theta(\text{leaf})$$

After N(s, a) simulations, Q(s, a) is the sample mean of N(s, a) independent value estimates, each drawn from a different game continuation through that branch. This is the same object as $\hat{\mu}_a$ in UCB — built the same way, by the same logic.

---

## N(s, a): Visit Count as a Reliability Certificate

The visit count N(s, a) counts how many simulations have explored branch (s, a). In UCB this was n(a) — the number of times arm *a* was pulled — and it appeared in the denominator of the confidence interval because more pulls mean a tighter estimate.

Exactly the same logic applies here. With N(s, a) = 1, Q(s, a) rests on a single network evaluation from a single leaf. The network is good but not perfect; one data point could be misleading. With N(s, a) = 100, Q(s, a) is the average of a hundred independent evaluations, each taken along a different path through that branch. By the Law of Large Numbers, that average is much closer to the branch's true value.

The 1/(1 + N(s, a)) factor in the exploration term encodes this directly. When N(s, a) = 0 — a move that has never been simulated — the denominator is 1 and the exploration bonus is at its ceiling. The search treats this branch as maximally uncertain and gives it a strong push to be tried. As N(s, a) grows, the denominator grows with it, the bonus shrinks, and Q(s, a) increasingly takes over. Certainty is earned by visiting; uncertainty is the price of not visiting.

This is the same mechanism as UCB's √(log t / n(a)): both formulas give high scores to undersampled options and progressively hand control to the empirical mean as evidence accumulates.

---

## N(s): The Shared Clock

In UCB, the numerator carried log t — the total number of pulls so far — because the confidence bound needs to hold across all rounds simultaneously, not just at any one moment. As t grows, the bound widens slightly to stay valid.

In PUCT, the numerator carries √N(s), where N(s) = Σ_b N(s, b) is the total number of simulations run from state *s* across all moves. It plays the same role: as more simulations run, a branch that has been visited 10 times looks more neglected relative to the full budget than it did early on, so the bar for "sufficiently explored" rises.

The switch from log t to √N(s) is a simplification — the full UCB derivation would give log N(s) — but it works well in practice, and the key property is preserved: the exploration bonus rewards undersampled branches relative to total effort.

---

## P(s, a): The Prior That UCB Didn't Have

Here is the one term with no UCB counterpart: P(s, a), the neural network's prior probability for move *a* from position *s*.

In the bandit, all arms start as equally unknown. You have no reason to prefer one slot machine over another before you've tried them. The confidence bonus therefore starts equal for all arms and differentiates only as pulls accumulate.

In a game of Ultimate Tic-Tac-Toe, some moves are obviously better than others before any search — blocking a winning threat, completing a sub-board, forcing the opponent into a bad position. The neural network, trained on thousands of self-play games, has learned to recognise these patterns. P(s, a) encodes that pre-search judgment as a probability distribution over moves.

Multiplying the exploration bonus by P(s, a) tilts early exploration toward moves the network considers promising. A move with P(s, a) = 0.4 gets twice the initial exploration push of a move with P(s, a) = 0.2, even before either has been visited. This lets the search concentrate its first simulations where the payoff is likely highest, rather than spreading them uniformly across all 81 cells.

The prior matters most when N(s, a) is small — at the very start of search, or deep in the tree where branches are newly created. As N(s, a) grows and Q(s, a) accumulates real evidence, the influence of P(s, a) fades. The search progressively stops trusting the prior and starts trusting what it has actually seen.

---

## How the Formula Balances the Three Forces

Put it together: at any moment during search, the PUCT score for move *a* from position *s* is:

- **High** if Q(s, a) is high — the branch has looked good across many simulations
- **High** if N(s, a) is low — the branch is underexplored and uncertain
- **High** if P(s, a) is high — the network considers this move promising

A move scores high if it is *good*, or *underexplored*, or *recommended by the network*. The search will always prefer a well-supported good move, but it will also revisit uncertain moves and network-favoured moves rather than ignoring them. Nothing is permanently abandoned.

As simulations run, the first and third forces gradually dominate the second. Q values stabilise as they are backed by many samples; visit counts grow and the exploration bonus shrinks; the network prior, fixed throughout the search, recedes. What started as prior-guided exploration ends as evidence-guided exploitation.
