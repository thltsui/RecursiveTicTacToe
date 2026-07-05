# T2b: TD Learning, Q-Learning, and Deep Q-Networks

*In the last essay, we set up the credit-assignment problem and introduced value functions — the mathematical target that reinforcement learning algorithms are trying to approximate. We saw that Monte Carlo learning works in principle but is slow and noisy because it waits for the game to end before updating anything. This essay introduces a faster alternative: Temporal Difference learning, which updates estimates on every step using the current game state rather than waiting for a final outcome.*

---

## Temporal Difference Learning: Don't Wait

Temporal Difference (TD) learning makes a different bet than Monte Carlo. Instead of waiting for the full return, it **bootstraps** — it uses its current estimate of future value as a target, updating after every single transition.

After observing a step sₜ → aₜ → rₜ → sₜ₊₁, the TD update is:

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

This quantity δₜ is the **TD error** — the gap between what we expected (V(sₜ)) and what we observed plus what we now expect from the next state (rₜ + γ·V(sₜ₊₁)). We update:

$$V(s_t) \leftarrow V(s_t) + \alpha \delta_t$$

TD learning can update after every single step, not just at the end of a game. Its update target is *biased* — it uses V(sₜ₊₁), which is itself an estimate — but it is much lower variance than Monte Carlo, because the target only looks one step into the future.

### The Propagation Problem

There is a subtlety worth pausing on. At the start of training, V(s) = 0 for every state. Suppose rewards are zero except +1 for winning.

On the first game, every transition has rₜ = 0 and V(sₜ₊₁) = 0, so δₜ = 0. Nothing is learned from any non-terminal state. Only the transition *into* the terminal winning state gets a nonzero update: δₜ₋₁ = 1 + 0 − 0 = 1.

On the second game, if the agent visits the same pre-winning state again, *that* state now has positive value, so the state before it gets a nonzero update. The signal is propagating backward — but only one step per game.

In theory, over many games, this "correction wave" propagates from terminal states all the way back through the game tree. In practice, for Ultimate Tic-Tac-Toe, the wave never arrives.

The reason is the state space: roughly 3⁸¹ ≈ 4.4 × 10³⁸ possible board configurations. Almost every game traces a completely different path. The probability that two games share a non-terminal state is vanishingly small. Corrections from one game have nowhere to propagate.

This is not a problem that more games will fix. Even after a billion games, you've visited a negligible fraction of the space. Tabular TD learning — storing one number per state — is simply not viable at this scale.

---

## Q-Learning

Rather than estimating V(s), it is often more useful to estimate Q(s, a) directly.

Q-learning updates Q after each transition:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]$$

The target rₜ + γ · maxₐ' Q(sₜ₊₁, a') is the "greedy" TD target: it uses the *best* Q-value available at the next state, regardless of what action the agent actually took. This makes Q-learning an **off-policy** algorithm — it learns the optimal policy even while following a different (e.g. exploratory) one.

Q-learning has strong convergence guarantees in the tabular case. But it inherits the same scaling problem: storing one number per (state, action) pair is impossible when the state space is 3⁸¹.

---

## Deep Q-Networks (DQN)

The fix is to replace the Q-table with a neural network Q(s, a; θ). Given a state s encoded as a tensor, the network outputs Q-values for every legal action simultaneously. One forward pass, all Q-values at once.

Training minimises:

$$L(\theta) = \mathbb{E}\left[ (r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta))^2 \right]$$

Two engineering details make this work in practice:

**Target network (θ⁻).** A periodically frozen copy of the main network. Without it, the target changes every gradient step — you're chasing a moving target, which causes oscillation and divergence. Freezing the target for a few thousand steps stabilises training dramatically.

**Experience replay.** Past transitions (s, a, r, s') are stored in a buffer and sampled randomly for training. Sequential game data is temporally correlated — consecutive positions look nearly identical — which biases gradient estimates. Random sampling from a large buffer breaks these correlations and ensures each gradient step gets a diverse batch.

This combination — neural function approximation + target network + experience replay — is what made deep reinforcement learning practical. DeepMind's DQN (2015) achieved superhuman performance on dozens of Atari games using only raw pixel input. The same network architecture, with no game-specific engineering, learned to play Pong, Breakout, Space Invaders, and dozens of others.

### The Limitation of DQN

For Ultimate Tic-Tac-Toe, DQN is a natural starting point — but it has a structural limitation. It is purely **value-based**: it estimates how good each action is, and acts greedily on that estimate. It says nothing about what to do when multiple actions have similar values, and it cannot reason about the long-range dependencies that make Ultimate TTT hard.

More fundamentally, DQN is reactive. It evaluates states it has encountered. It does not *search forward* — it doesn't simulate "if I play here, and then they play there, and then I play here..." That kind of deliberate look-ahead is precisely what distinguishes a strong game-playing agent from a weak one.

---

## Where This Leaves Us

We now have two complementary tools: Monte Carlo (unbiased but high-variance, waits for full games) and TD learning (biased but low-variance, updates every step). DQN scales both to large state spaces using neural function approximation.

But all of these methods share a fundamental limitation: they learn to react to states they have seen. They don't search.

The next essay turns to a different approach: instead of estimating value, we directly estimate the *policy* — the probability distribution over actions. This unlocks a different class of algorithms, from REINFORCE to PPO, and ultimately leads to the actor-critic architecture at the heart of AlphaZero.

*Next: T2c — From Policy Gradients to PPO.*
