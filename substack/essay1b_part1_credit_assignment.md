# T2a: The Credit-Assignment Problem

*This is the second essay in the Zero to AlphaZero series. In the first, we introduced Ultimate Tic-Tac-Toe — the game at the heart of this series — and explained why it's an ideal testbed for reinforcement learning. Now we turn to the foundational ideas that make learning from games possible at all. We start with a deceptively simple question: how do you figure out which of your decisions actually mattered?*

---

## The Credit-Assignment Problem

Imagine you just won a game of Ultimate Tic-Tac-Toe after forty moves. You know the outcome — you won — but which of those forty moves deserves the credit? The final move closed out the game, but it was only possible because move thirty-seven created the right global configuration, which itself depended on move twenty-two establishing a threat on the middle local board.

This is the **credit-assignment problem**, and it is the central challenge in reinforcement learning. An agent takes a sequence of actions, receives a reward at some point (possibly much later), and must figure out which actions caused which rewards. It is the machine-learning equivalent of asking: what, exactly, did you do right?

Everything in this essay — every algorithm, every design choice — is a different answer to that question.

---

## What to Estimate: V vs Q

Before we can assign credit, we need a way to evaluate positions. This requires choosing what, exactly, we are trying to estimate.

First, a concept of a **policy** (denoted π). A policy is a strategy: it dictates how the agent approaches the game and what moves it is likely to play in any given situation.

With a policy in mind, we can define the **value of a state** s. Written Vπ(s), this is the expected total reward the agent will collect if it starts from state s and plays out the rest of the game following policy π:

$$V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k r_{t+k} \mid s_t = s \right]$$

The parameter γ ∈ (0, 1) is the **discount factor**. Each future reward is multiplied by γ raised to the number of steps away it occurs — so rewards sooner are worth more than rewards later.

Why does γ appear? Three interlocking reasons.

**First, mathematical convergence.** For games that could theoretically continue indefinitely, an infinite sum of undiscounted rewards would diverge. γ < 1 ensures the math converges.

**Second, uncertainty.** Even in a finite game, a reward ten steps away is contingent on ten more transitions going as expected. Discounting by γ encodes the idea that a win *now* is worth more than a win *eventually* — because everything in between could go wrong.

**Third, and most importantly, structure.** γ < 1 gives the value function a recursive form that unlocks everything that follows. If you're following the best possible strategy, then the moves you make from *any* point must also be the optimal way to finish from that point. This is Bellman's principle of optimality, and it implies:

$$V^*(s) = \max_a \left[ r(s,a) + \gamma \cdot \mathbb{E}[V^*(s')] \right]$$

This is the **Bellman optimality equation**. The value of the best action from state s equals the immediate reward plus the discounted value of wherever that action takes you. Here, s' is the next state — a random variable because the opponent or environment may be unpredictable.

The Bellman equation is the compass. If you somehow knew V* perfectly, you would never need to search or think ahead: at any state, you'd simulate every legal move, look up the value of the resulting state, and pick the action with the highest expected score. The entire goal of reinforcement learning is to *approximate* V*.

### From States to Actions: The Q-Function

V(s) has a practical limitation. To find the best move using V alone, you must simulate every possible action to see what state it leads to, then look up V of that state. This requires knowing the rules of the game — a *model* of the environment.

To avoid this, we define Q(s, a): the expected total reward if we start in state s, take action a, and then follow our policy for the rest of the game:

$$Q^\pi(s, a) = r(s,a) + \gamma \cdot \mathbb{E}[V^\pi(s')]$$

Because Q already has the action baked in, choosing the best move is trivial: look at Q(s, a) for all legal moves a, and pick the highest. No model required.

This is why Q-functions dominate in practice: they decouple decision-making from environment knowledge.

---

## Monte Carlo Learning: Wait for the Outcome

Now that we have something to estimate, how do we estimate it from experience?

The simplest approach is to play games all the way to the end, then assign credit based on the final outcome.

Define the **return** from time t as the discounted sum of all future rewards up to the end of the game:

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots + \gamma^{T-t} r_T$$

After the game ends, we know Gₜ exactly for every step. We then update our value estimate at each visited state:

$$V(s_t) \leftarrow V(s_t) + \alpha \left[ G_t - V(s_t) \right]$$

This is an exponentially weighted running average: each update moves V(sₜ) a fraction α toward the observed return.

**The advantage:** Monte Carlo is *unbiased*. Because Gₜ is the actual return from the actual game — not an estimate of an estimate — we're moving toward the true value on average.

**The cost:** *variance*. Gₜ depends on everything that happens after time t. In a forty-move game, the return from the first move reflects thirty-nine more decisions, each of which could have gone differently. Individual estimates are noisy, and convergence is slow.

The deeper issue is credit assignment. Every state in the game gets updated, but the update for the first move is driven by a cumulative return that includes the quality of thirty-nine more decisions. The signal is blunt. It worked well enough for the game to end in a win — but which of those decisions was the *key* one?

Monte Carlo can't answer that question directly. The update for move 1 and move 39 both look the same. Everything is weighted only by how far it is from the terminal reward.

---

## Where This Leaves Us

We have a precise mathematical target — the Bellman-optimal value function V* — and a straightforward way to estimate it from complete games. But Monte Carlo has a fundamental limitation: it waits for the game to end before updating anything. In games with long horizons and sparse rewards, this produces slow, noisy learning.

The next essay introduces Temporal Difference learning — an approach that updates estimates *as the game is being played*, using current estimates of future value rather than waiting for the actual outcome. It's a key step toward the credit-assignment solution that makes AlphaZero work.

*Next: T2b — TD Learning, Q-Learning, and Deep Q-Networks.*
