# T2c: From Policy Gradients to PPO

*So far in this series, we've built up value-based reinforcement learning: estimating how good each state or action is, then acting greedily on those estimates. TD learning and DQN are powerful tools — but they share a fundamental limitation: they learn to react to positions they've seen. They don't search forward. This essay completes the RL toolkit with a different approach: instead of estimating value, we directly learn the policy itself. And we'll see where this naturally leads — to the actor-critic architecture at the heart of AlphaZero.*

---

## Policy Gradients: Learn the Strategy Directly

A policy gradient method skips the Q-function and directly learns πθ(a|s): a probability distribution over actions given the current state. The parameter θ represents the weights of a neural network.

The objective is simple: maximise expected return.

$$J(\theta) = \mathbb{E}_{\pi_\theta}\left[ \sum_{t=0}^{T} \gamma^t r_t \right]$$

To maximise J(θ), we need its gradient. The **policy gradient theorem** gives:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t \right]$$

The key object is ∇θ log πθ(aₜ | sₜ) — the *score function*. The intuition is direct: if Gₜ is large (the game went well), increase the log-probability of the actions you took. If Gₜ is small, decrease them.

Why log π rather than π? Because ∇θ log πθ = ∇θ πθ / πθ. When we take an expectation over actions sampled from πθ, the πθ in the denominator cancels the πθ in the sampling measure, producing a clean gradient estimate computable from samples.

### REINFORCE

REINFORCE implements this directly: play a complete game, compute Gₜ for every step, update θ by gradient ascent.

It's unbiased — Gₜ is the actual return, so the gradient estimate is correct on average. The cost is variance: Gₜ reflects everything that happens after time t, including decisions and events unrelated to the quality of action aₜ. Convergence is slow. In practice, large numbers of games are needed before the signal becomes reliable.

---

## Actor-Critic: The Best of Both Worlds

The actor-critic architecture addresses REINFORCE's variance problem by replacing the full return Gₜ with a bootstrapped estimate — exactly as TD learning did for value functions.

The **actor** is the policy πθ(a|s). The **critic** is a value function Vφ(s) trained to estimate how good each state is.

Instead of weighting the score function by Gₜ, actor-critic weights it by the **advantage**:

$$\hat{A}_t = G_t - V_\phi(s_t)$$

The advantage asks: was this action *better or worse than expected from this state*? Subtracting Vφ(sₜ) doesn't change the expected gradient (it's independent of the actions taken), but it dramatically reduces variance by removing the component of Gₜ that reflects how good the state was — regardless of the action taken.

In the simplest actor-critic, the advantage is approximated by the TD error:

$$\hat{A}_t \approx r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

This mirrors the MC/TD trade-off we saw earlier: REINFORCE is Monte Carlo policy gradient; actor-critic is TD policy gradient.

### Generalised Advantage Estimation (GAE)

Between the one-step TD estimate (low variance, high bias) and the full Monte Carlo return (high variance, low bias) is a continuum parameterised by λ ∈ [0, 1]:

$$\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

where δₜ₊ₗ is the TD error at step t+l. When λ = 0, this collapses to the one-step TD error. When λ = 1, it becomes the full Monte Carlo return minus baseline. GAE is now standard in high-performance RL systems.

---

## PPO: Keeping Updates Stable

A practical problem with policy gradient methods is that gradient steps can be too large, accidentally destroying a good policy. If you shift πθ too much in one update, you might move from "pretty good" to "broken" with no easy way back.

PPO (Proximal Policy Optimisation) constrains how much the policy can change in a single update. Define the probability ratio:

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}$$

PPO's clipped objective:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t,\ \text{clip}(r_t(\theta), 1{-}\epsilon, 1{+}\epsilon) \hat{A}_t \right) \right]$$

If an action was advantageous (Â > 0), the objective rewards increasing its probability — but only up to a factor of 1+ε. If disadvantageous, the objective penalises increasing its probability — but cannot benefit from decreasing it below 1−ε.

The result: meaningful learning per update, without destabilising the policy.

### A Note on RLHF

PPO is the backbone of Reinforcement Learning from Human Feedback — the technique that took GPT-3 to ChatGPT. The pipeline: pretrain a language model, train a reward model from human preference comparisons, then fine-tune the language model with PPO against the reward model's scores. The language model is the actor; the reward model plays the role of the environment.

Actor-critic in the language domain, at scale.

---

## The Remaining Problem

We now have a complete toolkit: Monte Carlo, TD learning, Q-learning, DQN, REINFORCE, actor-critic, GAE, PPO. All are principled solutions to the credit-assignment problem. All have been demonstrated to work at scale in various domains.

None of them, applied directly to Ultimate Tic-Tac-Toe, produces a strong player.

The issue is that all of these methods are **reactive**: they learn to evaluate or act on states they have encountered. They do not engage in deliberate search. A DQN agent picks the action with the highest Q-value; it doesn't look ahead three moves and ask whether that action creates an unavoidable threat. A policy gradient agent samples from πθ(a|s); it doesn't simulate the opponent's best response.

In games with deep tactical structure — where the right move depends on reading opponent responses several turns ahead — this is a serious limitation.

What we need is a way to combine learned value estimates with explicit forward search. That combination requires solving one more foundational problem first: how do you decide which moves are even worth thinking about?

*Next: T3 — From Bandits to Trees. We start at a slot machine, arrive at the UCB formula, and show how the explore/exploit trade-off that governs random reward machines also governs game tree search.*
