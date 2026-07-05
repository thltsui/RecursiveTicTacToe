import re

with open('substack/essay1b_rl_background.md', 'r') as f:
    text = f.read()

replacements = {
    "Vπ(s) = Eπ [ rₜ + γ·rₜ₊₁ + γ²·rₜ₊₂ + … | sₜ = s ]": r"$$ V^\pi(s) = \mathbb{E}_\pi \left[ r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots \mid s_t = s \right] $$",
    "Vπ(s) = Eπ [ Σ γᵏ · rₜ₊ₖ | sₜ = s ]": r"$$ V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k r_{t+k} \mid s_t = s \right] $$",
    "V*(s) = maxₐ [ r(s,a) + γ · E[ V*(s') ] ]": r"$$ V^*(s) = \max_a \left[ r(s,a) + \gamma \cdot \mathbb{E}[V^*(s')] \right] $$",
    "Gₜ = rₜ + γ · Gₜ₊₁": r"$$ G_t = r_t + \gamma \cdot G_{t+1} $$",
    "V(sₜ) = E[ rₜ + γ · Gₜ₊₁ | sₜ ] = E[ rₜ | sₜ ] + γ · E[ E[ Gₜ₊₁ | sₜ₊₁ ] | sₜ ] = E[ rₜ | sₜ ] + γ · E[ V(sₜ₊₁) | sₜ ]": r"$$ V(s_t) = \mathbb{E}[r_t + \gamma G_{t+1} \mid s_t] = \mathbb{E}[r_t \mid s_t] + \gamma \mathbb{E}[\mathbb{E}[G_{t+1} \mid s_{t+1}] \mid s_t] = \mathbb{E}[r_t \mid s_t] + \gamma \mathbb{E}[V(s_{t+1}) \mid s_t] $$",
    "‖T*V − T*U‖_∞ ≤ γ · ‖V − U‖_∞": r"$$ \|T^*V - T^*U\|_\infty \leq \gamma \|V - U\|_\infty $$",
    "Gₜ = rₜ + γ·rₜ₊₁ + γ²·rₜ₊₂ + … + γᵀ⁻ᵗ·r_{T}": r"$$ G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots + \gamma^{T-t} r_T $$",
    "V(sₜ) ← V(sₜ) + α · [Gₜ − V(sₜ)]": r"$$ V(s_t) \leftarrow V(s_t) + \alpha \left[ G_t - V(s_t) \right] $$",
    "δₜ = rₜ + γ · V(sₜ₊₁) − V(sₜ)": r"$$ \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) $$",
    "V(sₜ) ← V(sₜ) + α · δₜ": r"$$ V(s_t) \leftarrow V(s_t) + \alpha \delta_t $$",
    "Q(sₜ, aₜ) ← Q(sₜ, aₜ) + α · [ rₜ + γ · maxₐ' Q(sₜ₊₁, a') − Q(sₜ, aₜ) ]": r"$$ Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right] $$",
    "L(θ) = E[ (r + γ · maxₐ' Q(s', a'; θ⁻) − Q(s, a; θ))² ]": r"$$ L(\theta) = \mathbb{E}\left[ (r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta))^2 \right] $$",
    "J(θ) = Eπθ [ G₀ ] = Eπθ [ Σ γᵗ · rₜ ]": r"$$ J(\theta) = \mathbb{E}_{\pi_\theta}[G_0] = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t r_t \right] $$",
    "∇θ J(θ) = Eπθ [ Σ ∇θ log πθ(aₜ | sₜ) · Gₜ ]": r"$$ \nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) G_t \right] $$",
    "Âₜ = Gₜ − Vφ(sₜ)": r"$$ \hat{A}_t = G_t - V_\phi(s_t) $$",
    "Âₜ ≈ δₜ = rₜ + γ · Vφ(sₜ₊₁) − Vφ(sₜ)": r"$$ \hat{A}_t \approx \delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t) $$",
    "Âₜ(GAE) = Σ (γλ)ˡ · δₜ₊ₗ": r"$$ \hat{A}_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l} $$",
    "rₜ(θ) = πθ(aₜ | sₜ) / πθ_old(aₜ | sₜ)": r"$$ r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)} $$",
    "L(CLIP)(θ) = E [ min( rₜ(θ) · Âₜ, clip(rₜ(θ), 1−ε, 1+ε) · Âₜ ) ]": r"$$ L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right] $$"
}

for old, new in replacements.items():
    if old not in text:
        print(f"WARNING: Could not find '{old}'")
    text = text.replace(old, "\n" + new + "\n")

# In 5c3b37e, some markdown formatting might be left if I didn't strip it all. Let's force strip all markdown asterisks `**` and `*` used for formatting.
# Actually I already did it in 5c3b37e but let's just make absolutely sure no * exists EXCEPT for math operators `V*`, `π*`, `T*`, `θ*`.
# Since they specifically said "Please take out all *", I will just literally remove ALL `*` from the entire text except inside `$$` blocks!
# To do this safely:
parts = text.split("$$")
for i in range(len(parts)):
    if i % 2 == 0:
        # Outside math blocks, literally remove ALL asterisks as requested
        parts[i] = parts[i].replace('*', '')

text = "$$".join(parts)

with open('substack/essay1b_rl_background.md', 'w') as f:
    f.write(text)

