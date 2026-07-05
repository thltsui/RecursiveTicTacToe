# T6: Temperature, Dirichlet Noise, and the Training Loop

*After 800 MCTS simulations, you have visit counts for each legal move. How do you turn those into a move? And how do those moves become training data for the network? This essay covers the two remaining components — temperature and Dirichlet noise — and shows how they combine with the four phases from T5 into the self-play training loop that makes AlphaZero work.*

---

## Temperature: Exploration vs Exploitation at Decision Time

After running 800 simulations, the naive answer for move selection is "pick the most visited move." That's right during evaluation — when you want to see the network's actual best play — but it's wrong during training. Always playing the top move produces identical self-play games, an extremely narrow distribution of board positions, and a network that never learns to handle positions that deviate from its own favourite lines.

The solution is **temperature-controlled sampling**. Given visit counts N(a) for each legal move, we sample from:

$$\pi(a) = \frac{N(a)^{1/T}}{\sum_{a'} N(a')^{1/T}}$$

where T is the temperature parameter.

At **T = 0** (greedy): always pick the most visited move. Completely deterministic. Used during arena evaluation.

At **T = 1.0**: sample proportional to visit counts. If MCTS gave 450 visits to move A and 350 to move B, you pick A about 56% of the time and B about 44%.

At **T = 1.25**: the exponent 1/1.25 = 0.8 compresses the differences between visit counts. Moves with 450 visits and 350 visits get closer to equal probability. More randomness, more diverse games.

The schedule during self-play: use T = 1.25 for the first ~30 moves (opening and early midgame, where many lines are roughly equivalent and position diversity matters most), then switch to T = 0 (greedy) for the late game. The opening explores; the endgame plays to win.

Temperature is the coarsest tool for diversity. It can soften the gap between the best and second-best move, but it can't force MCTS to seriously explore a move the network has nearly ignored. For that, you need Dirichlet noise.

---

## Dirichlet Noise: The Insurance Policy Against Fixation

Suppose the network assigns 90% prior probability to one move. MCTS will allocate most of its 800 simulations to that move. The other moves get roughly 11 simulations each — not enough to genuinely evaluate them.

This might be fine if the network is right. But the network could be miscalibrated, especially early in training. By allocating almost all simulations to one path, we've committed to the network's initial judgment without genuinely interrogating the alternatives. In the worst case: a blind spot in the network's prior leads to a blind spot in the search, which reinforces the prior in training, which deepens the blind spot. A fixation loop.

The fix is **Dirichlet noise** — a random perturbation added to the prior at the root node:

$$P_{\text{noisy}}(a) = (1 - \varepsilon) \cdot P(a) + \varepsilon \cdot \text{Dir}(\alpha)$$

where Dir(α) is a draw from the Dirichlet distribution with concentration parameter α.

We use α = 0.3 and ε = 0.35. The ε = 0.35 means 35% of the prior comes from noise and 65% from the network. This is a substantial perturbation — even when the network is confident, MCTS still allocates meaningful simulations to the alternatives.

Two design choices matter here:

**Noise only at the root.** Deeper nodes in the tree are not perturbed. We want diversity at the *decision point* — the move we're actually playing — without corrupting the downstream evaluation of what follows. Adding noise everywhere would simulate near-random play throughout the tree, defeating the purpose of having a prior.

**α = 0.3 produces spiky samples.** The Dirichlet distribution with small α concentrates most of its mass on a few moves, unpredictably. The noise doesn't uniformly boost all alternatives — it boosts a random subset. This means the search is forced to seriously consider some alternatives, but it's not paralysed into equal treatment of obviously bad moves.

---

## From Simulations to Training Data

During self-play, every move in every game produces a training example. The process:

**Step 1.** Run 800 MCTS simulations from the current position (with Dirichlet noise at root).

**Step 2.** Record the visit distribution as the **policy target** π:

$$\pi(a) = \frac{N(a)}{\sum_{a'} N(a')}$$

This tells the network what fraction of MCTS's attention went to each move — a refined version of what the raw prior said.

**Step 3.** Select the actual move via temperature sampling and play it.

**Step 4.** Continue until the game ends. Retroactively label every position in the game with **z** = +1 (current player won from that position), −1 (lost), or 0 (draw).

The training record for each position is the triple **(state, π, z)**: the board position, what MCTS recommended, and what eventually happened.

---

## The Loss Function and the Feedback Loop

The network is trained to minimise:

$$\mathcal{L} = \underbrace{-\sum_a \pi(a) \log p(a)}_{\text{policy: match MCTS visits}} + \underbrace{(z - v)^2}_{\text{value: match game outcome}}$$

The **policy loss** (cross-entropy) pushes the network's prior toward the MCTS visit distribution. If MCTS concentrated 49% of its simulations on move A, the network learns to assign ~49% probability to that move.

The **value loss** (mean squared error) pushes the network's position estimate toward the actual game outcome. If the network said +0.3 but the game was lost, the loss pulls the estimate downward.

Why does this work? MCTS is a **better policy than the raw prior**. It takes the network's prior as input, runs 800 focused simulations, and produces a refined estimate that accounts for actual game continuations — not just the network's initial guess. By training on MCTS visit distributions instead of raw game outcomes, we're distilling the search process into the weights.

After training, the prior is closer to what MCTS would have recommended. The next MCTS round, using this improved prior, does better search. Which produces better training targets. Which produces a better prior. This self-reinforcing loop — search improves policy, policy improves search — is how AlphaZero begins at random play and converges on expert-level decisions.

The entropy of the visit distribution is a concrete measure of learning progress. At initialisation, visits are nearly uniform across 9 moves: entropy ≈ log(9) ≈ 2.2 nats. As training advances, the network concentrates its attention: by iteration 200, a single move might absorb 600 of 800 simulations, corresponding to entropy ≈ 0.5 nats. Watching policy entropy fall from ≈ 2.2 toward ≈ 0.5–1.0 over training iterations is watching the system learn to play.

---

*Next: the neural network itself — the (7, 9, 9) input encoding, the residual trunk, global pooling, and the five-component loss. The architecture is where the game-specific design decisions live.*
