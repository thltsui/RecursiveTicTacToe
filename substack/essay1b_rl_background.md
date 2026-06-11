# Essay 1b: The Reinforcement Learning Landscape

*How do you teach a program to play a game it has never seen before — without telling it what good play looks like? This essay builds up the reinforcement learning framework that makes this possible, from the Bellman equation to policy gradients, tracing a single thread: the credit-assignment problem.*

---

## 1. The Credit-Assignment Problem

Imagine you just won a game of Ultimate Tic-Tac-Toe after forty moves. You know the outcome — you won — but which of those forty moves deserves the credit? The final move closed out the game, but it was only possible because move thirty-seven created the right global configuration, which itself depended on move twenty-two establishing a threat on the middle local board.

This is the credit-assignment problem, and it is the central challenge in reinforcement learning. An agent takes a sequence of actions, receives a reward at some point (possibly much later), and must figure out which actions caused which rewards. It is the machine-learning equivalent of asking: what, exactly, did you do right?

Everything in this essay — every algorithm, every design choice — is a different answer to that question.

## 2. The Bellman Equation

Before choosing actions, we need a way to evaluate positions. Define the *value* of a state s as the expected total future reward an agent will collect from that state onwards, under some policy π:

V^π(s) = E_π [ r_t + γ·r_{t+1} + γ²·r_{t+2} + … | s_t = s ]

The parameter γ ∈ (0, 1) is the discount factor. Each reward is multiplied by γ raised to the number of steps in the future it occurs. This produces a discounted sum that we can write compactly as:

V^π(s) = E_π [ Σ_{k=0}^{∞} γ^k · r_{t+k} | s_t = s ]

Why does γ appear? There are three interlocking reasons, each more compelling than the last.

The first is purely mathematical: for the infinite sum to converge at all, we need the terms to shrink. If γ = 1 and rewards can be nonzero, the sum diverges. γ < 1 ensures convergence.

The second is philosophical: rewards far in the future should be worth less than immediate rewards. This captures the economic intuition of time preference, but it also encodes something deeper — uncertainty. A reward ten steps away is contingent on ten more transitions going as expected. Discounting by γ at each step is equivalent to assuming the game might end at any moment with probability 1 − γ.

The third reason is the one that gives the Bellman equation its power. Consider the optimisation problem we actually want to solve: find the policy π* that maximises E[Σ γ^t r_t]. This optimisation has a remarkable recursive structure. Any suffix of an optimal trajectory must itself be optimal — otherwise we could improve the overall trajectory by substituting a better suffix. This is Bellman's principle of optimality, and it implies that V* satisfies:

V*(s) = max_a [ r(s,a) + γ · E[V*(s')] ]

This is the Bellman optimality equation. It says: the value of the best action from s equals the immediate reward plus the discounted value of wherever that action takes you.

What's the mathematical justification for this recursive form? It falls out directly from the law of iterated expectations — what probabilists call the tower property. If G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + … denotes the return from time t, then:

G_t = r_t + γ · G_{t+1}

Taking expectations:

V(s_t) = E[r_t + γ · G_{t+1} | s_t] = E[r_t | s_t] + γ · E[E[G_{t+1} | s_{t+1}] | s_t] = E[r_t | s_t] + γ · E[V(s_{t+1}) | s_t]

The inner expectation E[G_{t+1} | s_{t+1}] is just V(s_{t+1}) by definition; the tower property lets us collapse it. The Bellman equation is, in this sense, nothing more than the law of iterated expectations applied to discounted sums.

Now — why does γ < 1 guarantee a *unique* solution to this equation? This is where the mathematics becomes beautiful, and where Brian Christian and Tom Griffiths, in *Algorithms to Live By*, make an observation that is easy to miss. The Bellman equation is not just a recursive formula — it is a *fixed-point equation* for the value function. We are looking for a V* such that applying the Bellman backup operator T* leaves it unchanged: T*V* = V*.

The discount factor γ is what makes T* a contraction mapping. For any two value functions V and U, the maximum difference after one application of T* is:

‖T*V − T*U‖_∞ ≤ γ · ‖V − U‖_∞

Because γ < 1, each application of T* brings V and U strictly closer together. By the Banach fixed-point theorem, any contraction on a complete metric space has exactly one fixed point — and repeated application of T* converges to it from any starting point. Without γ < 1, T* is no longer a contraction, and there is no guarantee that a unique solution exists or that iterative methods will converge to it.

Christian and Griffiths frame this through the lens of what they call the explore/exploit trade-off: future rewards are genuinely worth less not just because of impatience, but because the further ahead we peer, the more uncertain the path. The discount factor is thus both a mathematical necessity for convergence and a principled encoding of irreducible uncertainty about the future. Remove it, and you have not simplified the problem — you have broken it.

There is a third, deeper reason that we will meet in Essay 1c. The Gittins index theorem — the exact optimal solution to the multi-armed bandit problem under geometric discounting — requires γ < 1 for the same reason the Bellman equation does. Each arm's optimal value is computed by solving its own Bellman fixed-point equation, which is a contraction only when γ < 1. More remarkably, the proof that the K-arm optimisation decomposes into K independent single-arm problems depends on the geometric structure of discounting: γ < 1 makes each arm's future contributions separable from the others'. The discount factor is not just a convergence device — it is the key that unlocks tractability.

## 3. Two Design Choices

Given the value function, we need to decide how to store and compute it. There are two dimensions to this decision.

**What to estimate: V or Q?**

V(s) is the value of a state: a single number saying how good it is to be in state s. If we store one number per state, V is a *vector* of length |S| — one entry per state.

Q(s, a) is the value of taking action a from state s. It is a *matrix* of dimensions |S| × |A| — one entry per (state, action) pair. Q is more expensive to store, but it is more directly useful: to choose the best action, you read off the row for your current state and take the column with the highest value, without needing a model of the environment.

**How much of the game to observe before updating?**

This is the deeper design choice, and it determines the entire character of the algorithm. You are in state s_t. You have taken some actions. At some point you must use your observations to update your value estimates. The question is: how much of the future do you observe before making that update?

One extreme: you play the entire game to completion, collect the final outcome, and then work backward attributing credit. This is the Monte Carlo approach.

The other extreme: you take a single step, observe the immediate reward and the new state, and update immediately. This is the Temporal Difference approach.

Both are legitimate strategies. They are not approximations to each other — they are genuinely different algorithms with different statistical properties. To understand why both exist and when each is preferred, we need to look at them in detail.

## 4. Monte Carlo: Learn from Complete Games

The simplest strategy for the credit-assignment problem is to stop worrying about it during the game, play all the way to the end, and only then assign credit.

Define the return from time t as the discounted sum of all future rewards:

G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + … + γ^{T-t}·r_T

where T is the final timestep. After the game ends, we know G_t exactly for every timestep t. We then update our value estimate at each state visited:

V(s_t) ← V(s_t) + α · [G_t − V(s_t)]

This is an exponentially weighted running average: each update moves V(s_t) a fraction α toward the observed return.

Monte Carlo estimation is *unbiased*: because G_t is the actual return from the actual game, not an estimate of an estimate, we are moving toward the true value on average. The cost is *variance*: the return G_t depends on every subsequent action and every subsequent environment transition. In a forty-move game, G_t is a function of thirty-nine more steps, each of which could have gone differently. This makes individual estimates noisy, and convergence slow.

The deeper problem is credit assignment. Every state in the game gets updated, but only the terminal state receives a nonzero reward directly. The update for the first move is driven entirely by the cumulative return G_0 — a noisy signal that reflects the quality of thirty-nine more decisions. The first move is not well-served by this.

## 5. TD Learning: Learn from One Step

Temporal Difference learning makes a different bet. Instead of waiting for the game to end, it bootstraps: it uses its *current estimate* of future value to construct a training target.

After observing a transition s_t → a_t → r_t → s_{t+1}, the TD update is:

δ_t = r_t + γ · V(s_{t+1}) − V(s_t)

This quantity δ_t is the *TD error* — the gap between what we expected (V(s_t)) and what we observed plus what we now expect from the next state (r_t + γ·V(s_{t+1})). We update:

V(s_t) ← V(s_t) + α · δ_t

TD learning can update after every single step, not just at the end of a game. Its update target is *biased* — it uses V(s_{t+1}), which is itself an estimate, not the true value — but it is much lower variance than Monte Carlo, because the target only looks one step into the future.

There is a subtlety here worth pausing on. At the start of training, we initialise V(s) = 0 for every state. Consider a game where all rewards are zero except at the terminal state, where the winner receives +1. Now follow the TD update rule:

On the first game, every transition has r_t = 0 and (before the terminal state) V(s_{t+1}) = 0. So δ_t = 0 + γ·0 − 0 = 0. The update is zero. Nothing is learned from any non-terminal state.

At the terminal state, r_T = 1 and V(s_{T+1}) = 0 (game over), so δ_T = 1 + 0 − 0 = 1. The terminal state gets updated: V(s_T) moves from 0 to something positive.

On the second game, if the agent happens to visit the same s_T again, it will find V(s_T) > 0, and the state just before it will now have a nonzero target. The signal is propagating backward — but only one step per game.

In theory, over many games, this "correction wave" propagates from terminal states backward through the game tree, eventually giving every state an accurate value estimate. In practice, for Ultimate Tic-Tac-Toe, the wave never arrives.

The problem is the 3^81 state space. There are roughly 4.4 × 10^38 possible board configurations. Every game traces a path through this enormous space, and — crucially — almost every game traces a *different* path. The probability that any two games share a non-terminal state is vanishingly small. The correction learned from one game has nowhere to propagate to.

This is not a problem that more games will fix. You could play a billion games and still have visited a negligible fraction of the 3^81 possible states. The tabular approach — storing one number per state — is simply not viable for a game of this scale.

What is needed is a way to *generalise*: to update the value of one state and have that update automatically propagate to similar states. This is what neural networks provide. A neural network parameterises V(s) as a function of the state — shared weights mean that an update for one state influences the estimated values of all similar states. The correction wave becomes, in effect, a gradient that flows through the entire function class at once. We will return to this in later essays. For now, the lesson is: tabular TD learning is theoretically clean but practically unusable at scale.

<!-- Figure: figures/fig4_tensor_channels.png — "The 7-channel tensor encoding of a UTTT position after 5 moves (O to move). Each channel is a 9×9 binary plane. The encoding is always from the current player's perspective: Ch 0 shows O's pieces, Ch 1 shows X's pieces, Ch 2 highlights the sub-board O must play in, Ch 4 shows the sub-board X has already won, and Ch 6 is all-ones because it is O's turn." -->

## 6. Q-Learning

Rather than estimating V(s), it is often more useful to estimate Q(s, a) — the value of taking action a from state s, then playing optimally thereafter.

Q-learning updates Q directly:

Q(s_t, a_t) ← Q(s_t, a_t) + α · [r_t + γ · max_{a'} Q(s_{t+1}, a') − Q(s_t, a_t)]

The target r_t + γ · max_{a'} Q(s_{t+1}, a') is the "greedy" TD target: it uses the best Q-value available at the next state, regardless of what action the agent actually took. This makes Q-learning an *off-policy* algorithm — it learns the optimal policy even while following a different (e.g. exploratory) one.

Q-learning enjoys strong convergence guarantees in the tabular case. But it inherits exactly the same scaling problem as tabular V-learning: storing one number per (state, action) pair is impossible when the state space is 3^81.

## 7. Deep Q-Networks (DQN)

The fix is to replace the Q-table with a neural network Q(s, a; θ), parameterised by weights θ. Given a state s (encoded as a vector or tensor), the network outputs Q-values for every action simultaneously.

Training proceeds by minimising the loss:

L(θ) = E[ (r + γ · max_{a'} Q(s', a'; θ⁻) − Q(s, a; θ))² ]

where θ⁻ are the weights of a *target network* — a periodically frozen copy of the main network. The target network stabilises training: without it, the target itself changes every gradient step, leading to oscillation or divergence.

DQN also uses an *experience replay buffer*: past transitions (s, a, r, s') are stored and sampled randomly for training. Random sampling breaks the temporal correlations in sequential game data, which would otherwise bias the gradient estimates.

This combination — neural function approximation + target network + experience replay — is what made deep reinforcement learning practical. DeepMind's DQN paper (2015) demonstrated superhuman performance on dozens of Atari games using only raw pixels as input.

For UTTT, DQN is a natural starting point. But it has a limitation: it is purely value-based. It estimates how good each action is, and acts greedily. It says nothing about *how* to choose actions when many are similarly valued, or how to reason about the structure of the action space.

## 8. Policy Gradients and REINFORCE

An alternative approach is to directly parameterise the policy — the function that maps states to actions. Instead of learning V or Q and deriving a policy from them, we learn π_θ(a|s) directly: a probability distribution over actions given the current state.

The objective is simple to state: maximise expected return.

J(θ) = E_{π_θ} [G_0] = E_{π_θ} [ Σ_{t=0}^{T} γ^t · r_t ]

J(θ) is the expected discounted total reward when the agent follows policy π_θ from the start. We want to find θ* = argmax_θ J(θ).

To maximise J(θ), we need its gradient with respect to θ. The policy gradient theorem gives:

∇_θ J(θ) = E_{π_θ} [ Σ_t ∇_θ log π_θ(a_t | s_t) · G_t ]

The key object is ∇_θ log π_θ(a_t | s_t) — the gradient of the log-probability of the action taken, with respect to the policy parameters. This is called the *score function*. The intuition: if G_t is large (the game went well), we want to increase the log-probability of the actions we took. If G_t is small, we want to decrease them. The score function tells us which direction to push θ.

Why log π rather than π? Because ∇_θ log π_θ = ∇_θ π_θ / π_θ. When we take an expectation over actions (which are sampled from π_θ), the π_θ in the denominator cancels the π_θ in the sampling measure, giving a clean gradient estimate that we can compute from samples.

REINFORCE is the algorithm that implements this directly: play a complete game, compute G_t for every step, and update θ by gradient ascent. It is Monte Carlo policy gradient — it requires the full return G_t, not an estimate.

REINFORCE is unbiased: since G_t is the actual return from the actual game, the gradient estimate is correct on average. The cost, again, is variance. G_t is influenced by everything that happens after time t, including many things unrelated to the quality of action a_t. The signal is noisy, convergence is slow, and large numbers of games are required.

## 9. Actor-Critic, GAE, and PPO

The actor-critic architecture is to policy gradients what TD is to Monte Carlo: it replaces the full return G_t with a bootstrapped estimate, reducing variance at the cost of some bias.

The *actor* is the policy π_θ(a|s). The *critic* is a value function V_φ(s), parameterised separately, whose job is to estimate how good each state is.

Instead of weighting the score function by G_t, actor-critic weights it by the *advantage*:

Â_t = G_t − V_φ(s_t)

The advantage says: was this action better or worse than what we expected from this state? Subtracting V_φ(s_t) does not change the expected gradient (V_φ(s_t) is independent of the actions taken from s_t), but it dramatically reduces variance by removing the component of G_t that is due to the quality of the state rather than the quality of the action.

In the simplest one-step actor-critic, the advantage is estimated by the TD error:

Â_t ≈ δ_t = r_t + γ · V_φ(s_{t+1}) − V_φ(s_t)

This mirrors exactly the MC/TD trade-off we saw for value functions: REINFORCE is MC policy gradient, actor-critic is TD policy gradient.

**Generalised Advantage Estimation (GAE)**

Between the one-step TD estimate (low variance, high bias) and the full Monte Carlo return (high variance, low bias) is a continuum parameterised by λ ∈ [0, 1]:

Â_t^{GAE(γ,λ)} = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}

where δ_{t+l} = r_{t+l} + γ·V(s_{t+l+1}) − V(s_{t+l}) is the TD error at step t+l.

When λ = 0, this collapses to the one-step TD error δ_t. When λ = 1, it becomes the full Monte Carlo return minus the baseline. Any λ ∈ (0,1) interpolates smoothly between the two. GAE is now standard in high-performance RL systems.

**Proximal Policy Optimisation (PPO)**

A practical problem with policy gradient methods is that gradient steps can be too large, accidentally destroying a good policy. PPO addresses this by constraining how much the policy can change in a single update.

Define the probability ratio:

r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)

If r_t > 1, the new policy is more likely to take action a_t than the old one was. PPO's clipped objective is:

L^{CLIP}(θ) = E_t [ min( r_t(θ) · Â_t, clip(r_t(θ), 1−ε, 1+ε) · Â_t ) ]

The min of the unclipped and clipped versions ensures that the policy is only pushed in a beneficial direction, and never too far. If an action was advantageous (Â_t > 0), the objective rewards increasing its probability — but only up to a factor of 1+ε. If an action was disadvantageous, the objective penalises increasing its probability — but cannot benefit from decreasing it past 1−ε.

PPO is the most widely used policy gradient algorithm in practice. It is the backbone of RLHF.

**RLHF: Reinforcement Learning from Human Feedback**

A brief note on the connection to large language models. Reinforcement Learning from Human Feedback (RLHF) is the technique that took GPT-3 to ChatGPT and is responsible for much of the "helpful, harmless, honest" behaviour of modern AI assistants.

The pipeline works in three stages. First, a language model is pretrained on text in the usual way. Second, human raters compare pairs of model outputs and express preferences ("response A is better than response B"). These preferences are used to train a *reward model* — a separate neural network that predicts which responses humans will prefer. Third, the language model is fine-tuned with PPO, using the reward model as the reward signal. The policy (the language model) is optimised to generate outputs that score highly according to the reward model — which, by construction, reflects human preferences.

The actor is the language model. The critic is the reward model. The advantage is estimated from the critic's scores. The clipping in PPO prevents the fine-tuned model from drifting too far from the pretrained model, preserving its broad language abilities while nudging it toward human-preferred behaviour.

## 10. The Remaining Problem

We now have a rich toolkit: TD learning, Q-learning, DQN, REINFORCE, actor-critic, GAE, PPO. All of them are principled solutions to the credit-assignment problem. All of them have been demonstrated to work, in some domain, at scale.

None of them, applied naively to Ultimate Tic-Tac-Toe, produces a strong player.

The issue is that all of these methods are fundamentally reactive: they learn to evaluate or act upon states they have encountered. They do not, by themselves, engage in *deliberate search*. A DQN agent chooses the action with the highest Q-value; it does not look ahead three moves and ask whether that action will create an unavoidable threat. A policy gradient agent samples an action from π_θ(a|s); it does not simulate the opponent's best response.

In games with deep tactical structure — games where the right move depends on reading opponent responses several turns ahead — this is a serious limitation. What we need is a way to combine learned value estimates with explicit forward search.

That combination is Monte Carlo Tree Search — and it is the subject of the next essay.

---

*Next: Essay 1c introduces the multi-armed bandit problem and shows how the explore/exploit trade-off that governs slot machines also governs tree search. The UCB formula that solves the bandit problem turns out to be the key ingredient that makes MCTS both principled and practical.*

---

*Code: [Notebook 1 — The UTTT Game Engine](https://colab.research.google.com/github/thltsui/UlltimateTicTacToe/blob/Substack/substack/notebook_1_uttt_engine.ipynb) shows the full 7-channel `encode_state()` function in action, including a live demonstration of each channel's content for a real game position.*
