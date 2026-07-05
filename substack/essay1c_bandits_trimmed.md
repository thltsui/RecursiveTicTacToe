# T3: From Bandits to Trees

*The algorithms we've covered so far — TD learning, Q-learning, policy gradients — all share a common limitation: they react to states they have seen, but they don't search forward through states they haven't. Before we can fix this, we need to solve a more fundamental problem: when you have many options and limited time, how do you decide what to explore? We start at a slot machine and end up at the formula that drives AlphaZero's search.*

---

## The Multi-Armed Bandit

Imagine you are standing in front of a row of slot machines. Each machine has a different payout rate, but you don't know which is which. Pull a machine, collect some coins, and repeat. Your goal is to maximise your total winnings over a fixed number of pulls.

This is the **multi-armed bandit problem**. The name is old and slightly tongue-in-cheek — each machine is a different "arm" of the same bandit — but the problem it describes is ubiquitous: it appears in clinical trials (which drug to test?), online advertising (which ad to show?), and — as we will see — game tree search.

The core tension is between *exploration* and *exploitation*.

**Exploitation** means pulling the machine that has paid off best so far. It's rational: why waste a pull on an unknown machine when you know this one is good?

**Exploration** means trying machines you haven't pulled much, because one of them might be even better. The problem with pure exploitation is obvious: if you only ever pull the machine that seemed best in the first two tries, you might miss the best machine entirely.

The optimal balance is front-loaded exploration that tapers into pure exploitation as the horizon approaches. Early on, information about unknown machines is enormously valuable — you have many future pulls to benefit from it. Late on, exploration is wasteful: there isn't time to recoup a bad pull.

This insight has a precise mathematical form. The *regret* of a strategy is the gap between what you earned and what you would have earned pulling the best machine every time. The remarkable result from bandit theory: no strategy, however clever, can achieve regret better than O(log N). Logarithmic regret is the best possible — not because algorithms are limited, but because to confidently identify a suboptimal machine, you have to try it a number of times that grows logarithmically with your total pulls.

---

## UCB: Optimism in the Face of Uncertainty

UCB (Upper Confidence Bound) achieves this logarithmic regret bound with an elegant rule. For each arm a, maintain two quantities: the empirical mean reward μ̂(a) and a confidence bonus based on how many times you've pulled it.

The UCB score for arm a at time t:

$$\text{UCB}(a) = \hat{\mu}(a) + c \cdot \sqrt{\frac{\log t}{n(a)}}$$

where n(a) is the number of times arm a has been pulled, t is the total number of pulls so far, and c is a constant balancing exploration against exploitation.

The first term is exploitation: favour arms that have paid off well. The second term is exploration: favour arms that have been pulled less often (small n(a)) or where total experience is limited relative to how much you've tried this particular arm.

The key insight — "optimism in the face of uncertainty" — is that UCB always picks the arm with the highest *upper confidence bound*: the best plausible payout, not just the best observed payout. An arm you've only pulled twice might be terrible, or it might be the best arm you haven't yet found. By acting optimistically about uncertain arms, UCB is forced to explore them. Once an arm has been tried enough times, its uncertainty collapses and the empirical mean dominates.

The result: UCB explores exactly as often as mathematically necessary to identify the best arm — no more, no less. Suboptimal arms are visited logarithmically often, which is the minimum the information-theoretic argument demands.

---

## From Bandits to Trees: PUCT

Now take the bandit problem and put it inside a game tree.

At every node in the tree, we face a choice of which branch to explore next. This is exactly the bandit problem: the "arms" are the child nodes (the available moves), and the "payout" of pulling an arm is the value of the game continuation that arm leads to.

AlphaZero uses a version of UCB called **PUCT** (Predictor + Upper Confidence bound applied to Trees):

$$\text{PUCT}(s, a) = Q(s, a) + c_\text{puct} \cdot P(s, a) \cdot \frac{\sqrt{\sum_b N(s,b)}}{1 + N(s,a)}$$

Breaking this down:

**Q(s, a)** is the exploitation term: the average value of taking action a from state s, across all simulations that have explored this branch.

**N(s, a)** is the visit count for (s, a): how many simulations have gone down this branch. **Σ_b N(s,b)** is the total visits at state s — a proxy for total time, just like t in UCB.

**P(s, a)** is the *prior probability* of action a from state s — the neural network's prediction about how promising this move looks, before any search. This is the "Predictor" part that makes PUCT an improvement on vanilla UCB.

The structure is the same as UCB. Exploration is driven by the ratio √(total visits) / (1 + branch visits): branches that have been explored rarely get a large bonus, pulling the search toward underexplored parts of the tree. Exploitation is driven by Q(s, a): branches that have consistently produced high-value simulations get a higher score.

The prior P(s, a) is what distinguishes PUCT from vanilla UCB. Instead of treating all arms as equally unknown at the start, we initialise with the neural network's best guess about which actions are worth exploring. A strong prior concentrates search on promising moves immediately; a weak prior (early in training, when the network is random) produces near-uniform exploration, which is slower but still correct.

The prior is not fixed — it is *learned*. Early in training, P(s, a) is nearly uniform. As training progresses, the network learns which moves lead to wins and concentrates its prior there. This tightens the search, making each simulation more efficient. This feedback loop — search improves policy, policy improves search — is AlphaZero's core mechanism.

---

*Next: T4 — The PUCT Formula in Depth. We work through what Q(s, a) actually tracks, why visit counts beat Q-values for picking moves, and how the prior and search interact across thousands of simulations.*
