import re

with open('substack/essay1b_rl_background.md', 'r') as f:
    text = f.read()

old_intro = """## 2. The Bellman Equation

Before choosing actions, we need a way to evaluate positions. Define the *value* of a state s as the expected total future reward an agent will collect from that state onwards, under some policy π:

V^π(s) = E_π [ r_t + γ·r_{t+1} + γ²·r_{t+2} + … | s_t = s ]

The parameter γ ∈ (0, 1) is the discount factor. Each reward is multiplied by γ raised to the number of steps in the future it occurs. This produces a discounted sum that we can write compactly as:

V^π(s) = E_π [ Σ_{k=0}^{∞} γ^k · r_{t+k} | s_t = s ]

Why does γ appear? There are three interlocking reasons, each more compelling than the last.

The first is purely mathematical: for the infinite sum to converge at all, we need the terms to shrink. If γ = 1 and rewards can be nonzero, the sum diverges. γ < 1 ensures convergence.

The second is philosophical: rewards far in the future should be worth less than immediate rewards. This captures the economic intuition of time preference, but it also encodes something deeper — uncertainty. A reward ten steps away is contingent on ten more transitions going as expected. Discounting by γ at each step is equivalent to assuming the game might end at any moment with probability 1 − γ."""

new_intro = r"""## 2. The Bellman Equation

Before choosing actions, we need a way to evaluate positions. To do this, we first need a concept of a **policy** (denoted by $\pi$). You can think of a policy as a mental model or a strategy: it dictates how the agent approaches the game and what moves it is likely to play in any given situation.

With a strategy in mind, we can define the *value* of a state $s$. The value, denoted $V^\pi(s)$, is the expected total reward the agent will collect if it starts from state $s$ and plays out the rest of the game strictly following its mental model $\pi$. Mathematically, it is an expectation ($\mathbb{E}$) over the sum of future rewards:

$$
V^\pi(s) = \mathbb{E}_\pi \left[ r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots \mid s_t = s \right]
$$

The parameter $\gamma \in (0, 1)$ is the discount factor. Each reward is multiplied by $\gamma$ raised to the number of steps in the future it occurs. This produces a discounted sum that we can write compactly as:

$$
V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k r_{t+k} \mid s_t = s \right]
$$

Why does $\gamma$ appear? There are three interlocking reasons, each more compelling than the last.

The first is mathematical convergence. For games that could theoretically go on forever, an infinite sum of rewards would diverge if $\gamma = 1$. By setting $\gamma < 1$, we ensure the math behaves. However, in games like Ultimate Tic-Tac-Toe, the sum actually doesn't go to infinity because the game eventually ends. So why do we still use it?

This brings us to the second, more philosophical reason: uncertainty. Even though the game is finite, a reward ten steps away is contingent on ten more transitions going exactly as expected. Discounting by $\gamma$ ensures that we focus on securing a reward (like winning the game) earlier rather than later. It is equivalent to assuming the game might suddenly end or slip out of our control at any moment with probability $1 - \gamma$."""

text = text.replace(old_intro, new_intro)

replacements = {
    "V*(s) = max_a [ r(s,a) + γ · E[V*(s')] ]": r"$$ V^*(s) = \max_a \left[ r(s,a) + \gamma \cdot \mathbb{E}[V^*(s')] \right] $$",
    "G_t = r_t + γ · G_{t+1}": r"$$ G_t = r_t + \gamma \cdot G_{t+1} $$",
    "V(s_t) = E[r_t + γ · G_{t+1} | s_t] = E[r_t | s_t] + γ · E[E[G_{t+1} | s_{t+1}] | s_t] = E[r_t | s_t] + γ · E[V(s_{t+1}) | s_t]": r"$$ V(s_t) = \mathbb{E}[r_t + \gamma G_{t+1} \mid s_t] = \mathbb{E}[r_t \mid s_t] + \gamma \mathbb{E}[\mathbb{E}[G_{t+1} \mid s_{t+1}] \mid s_t] = \mathbb{E}[r_t \mid s_t] + \gamma \mathbb{E}[V(s_{t+1}) \mid s_t] $$",
    "‖T*V − T*U‖_∞ ≤ γ · ‖V − U‖_∞": r"$$ \|T^*V - T^*U\|_\infty \leq \gamma \|V - U\|_\infty $$",
    "G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + … + γ^{T-t}·r_T": r"$$ G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots + \gamma^{T-t} r_T $$",
    "V(s_t) ← V(s_t) + α · [G_t − V(s_t)]": r"$$ V(s_t) \leftarrow V(s_t) + \alpha \left[ G_t - V(s_t) \right] $$",
    "δ_t = r_t + γ · V(s_{t+1}) − V(s_t)": r"$$ \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) $$",
    "V(s_t) ← V(s_t) + α · δ_t": r"$$ V(s_t) \leftarrow V(s_t) + \alpha \delta_t $$",
    "Q(s_t, a_t) ← Q(s_t, a_t) + α · [r_t + γ · max_{a'} Q(s_{t+1}, a') − Q(s_t, a_t)]": r"$$ Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right] $$",
    "L(θ) = E[ (r + γ · max_{a'} Q(s', a'; θ⁻) − Q(s, a; θ))² ]": r"$$ L(\theta) = \mathbb{E}\left[ (r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta))^2 \right] $$",
    "J(θ) = E_{π_θ} [G_0] = E_{π_θ} [ Σ_{t=0}^{T} γ^t · r_t ]": r"$$ J(\theta) = \mathbb{E}_{\pi_\theta}[G_0] = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t r_t \right] $$",
    "∇_θ J(θ) = E_{π_θ} [ Σ_t ∇_θ log π_θ(a_t | s_t) · G_t ]": r"$$ \nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) G_t \right] $$",
    "Â_t = G_t − V_φ(s_t)": r"$$ \hat{A}_t = G_t - V_\phi(s_t) $$",
    "Â_t ≈ δ_t = r_t + γ · V_φ(s_{t+1}) − V_φ(s_t)": r"$$ \hat{A}_t \approx \delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t) $$",
    "Â_t^{GAE(γ,λ)} = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}": r"$$ \hat{A}_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l} $$",
    "r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)": r"$$ r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)} $$",
    "L^{CLIP}(θ) = E_t [ min( r_t(θ) · Â_t, clip(r_t(θ), 1−ε, 1+ε) · Â_t ) ]": r"$$ L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right] $$"
}

for old, new in replacements.items():
    text = text.replace(old, new)

# Inline
inline_replacements = {
    "π*": r"$\pi^*$",
    "V*": r"$V^*$",
    "T*": r"$T^*$",
    "θ*": r"$\theta^*$",
    "γ < 1": r"$\gamma < 1$",
    "γ = 1": r"$\gamma = 1$",
    "1 − γ": r"$1 - \gamma$",
    "α": r"$\alpha$",
    "θ": r"$\theta$",
    "δ_t": r"$\delta_t$",
    "G_t": r"$G_t$",
    "G_0": r"$G_0$",
    "V(s_t)": r"$V(s_t)$",
    "V(s_{t+1})": r"$V(s_{t+1})$",
    "Q(s, a)": r"$Q(s, a)$",
    "V(s)": r"$V(s)$",
    "π_θ(a|s)": r"$\pi_\theta(a|s)$",
    "V_φ(s)": r"$V_\phi(s)$",
    "V_φ(s_t)": r"$V_\phi(s_t)$",
    "E[Σ γ^t r_t]": r"$\mathbb{E}[\sum \gamma^t r_t]$",
    "r_t": r"$r_t$",
    "s_t": r"$s_t$",
    "s_{t+1}": r"$s_{t+1}$",
    "a_t": r"$a_t$",
    "r_T": r"$r_T$",
    "s_T": r"$s_T$",
    "s_{T-1}": r"$s_{T-1}$",
    "δ_{T-1}": r"$\delta_{T-1}$",
    "δ_{t+l}": r"$\delta_{t+l}$"
}

# we must do inline safely so we don't mess up existing blocks.
# but since the old text had space-separated variables, replacing space-padded is fine.
# We'll just replace the exact tokens.
for word in text.split():
    if word in inline_replacements:
        # replace isolated word
        pass

# A simpler way to replace inline is by regex matching word boundaries, 
# but python regex doesn't see subscripts as words always.
# So I'll just use string replacement with careful padding.
for old, new in inline_replacements.items():
    text = text.replace(" " + old + " ", " " + new + " ")
    text = text.replace(" " + old + ",", " " + new + ",")
    text = text.replace(" " + old + ".", " " + new + ".")
    text = text.replace("(" + old + ")", "(" + new + ")")

# Format equations properly for block math
text = text.replace("$$ $$", "$$")
text = re.sub(r'\$\$\s*\$\$', '$$', text)

with open('substack/essay1b_rl_background.md', 'w') as f:
    f.write(text)

