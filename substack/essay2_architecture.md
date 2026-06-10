# Teaching a Neural Network to Play Ultimate Tic-Tac-Toe: The Architecture

In the previous essay, we established that Ultimate Tic-Tac-Toe resists the kind of heuristic evaluation that makes classical chess engines work. The sending mechanic creates long-range dependencies — the value of a local move depends on global board structure in ways that are hard to express as a hand-crafted function. The solution is to learn the evaluation function from self-play data.

This essay is about how. Specifically: how do you turn a board position into a number between −1 and +1 that meaningfully represents "who's winning"? The answer involves some deliberate choices about how to encode the game state, a residual network with a specific trick for handling non-local dependencies, and a training signal composed of five distinct loss components. Each choice has a reason.

---

## 1. The Input: Encoding the Board as a (7, 9, 9) Tensor

Neural networks consume numbers. Before we can do anything interesting, we need to convert a game state into a tensor.

The board has 81 cells arranged as 9 sub-boards of 9 cells each. We represent this as a 9×9 spatial grid — the natural layout where each 3×3 block corresponds to one sub-board. The spatial structure matters: we want the network to be able to reason about adjacency, so we preserve the physical geometry of the board.

Each board position is encoded as 7 channels over this 9×9 grid — a tensor of shape (7, 9, 9):

```python
def encode_state(state: GameState) -> torch.Tensor:
    tensor = torch.zeros(7, 9, 9, dtype=torch.float32)
    cp = state.current_player  # 1 or -1

    # ... (see full implementation)

    # Channel 0: cells occupied by the current player
    # Channel 1: cells occupied by the opponent
    # Channel 2: valid cells for the next move (the "active sub-board" mask)
    # Channel 3: sub-boards won by the current player (all 9 cells marked)
    # Channel 4: sub-boards won by the opponent
    # Channel 5: sub-boards that are drawn / dead
    # Channel 6: turn indicator (all 0.0 if current player is P1, all 1.0 if P2)
```

A few things are worth unpacking here.

**Always from the current player's perspective.** Channel 0 is always "my pieces" and channel 1 is always "opponent's pieces" — regardless of whether the current player is player 1 or player 2. This is the *current-player convention* from AlphaZero, and it's important for a subtle reason: the network only needs to learn one policy (how to win from any position), not two separate policies (how to win as black, how to win as white). It sees itself as the positive player every time it's called, which dramatically simplifies what it needs to learn.

**Why channels 3–5 separately?** A won sub-board is strategically very different from an ongoing one. Won sub-boards are locked — you can't play in them — and if you're sent there, you get free choice. Flagging won, opponent-won, and drawn sub-boards as separate channels gives the network immediate access to this structural information without having to infer it from channels 0–1.

**Channel 2: the constraint.** The active sub-board mask is perhaps the most game-specific channel. It marks which cells are actually legal to play in on this turn. By giving the network direct access to where it *can* move, we spare it from having to infer this from the game history.

The result is a clean 7-channel representation that contains everything the network needs to know about the current position.

---

## 2. The Shared Trunk: Residual Blocks

The first thing the trunk does is lift the 7 input channels to 128 feature channels via a 3×3 convolution with padding:

```python
self.input_conv = nn.Conv2d(7, 128, kernel_size=3, padding=1, bias=False)
self.input_bn = nn.BatchNorm2d(128)
# then: h = F.relu(self.input_bn(self.input_conv(x)))  # (B, 128, 9, 9)
```

128 channels is a deliberate choice: AlphaZero uses 256, but Ultimate TTT is simpler than Go or chess, and 128 gives a good balance between representational capacity and training speed.

Then 8 residual blocks refine this representation. Each block keeps the shape fixed at (B, 128, 9, 9) — same channels, same spatial dimensions. After 8 blocks, the trunk outputs a (B, 128, 9, 9) tensor that is then split between the two heads.

**The residual connection.** Each block computes:

```
output = ReLU(F_local(x) + F_global(x) + x)
```

The `+ x` term — the *skip connection* — is what makes this a residual block. In a plain deep network, gradients must pass backwards through every layer from the loss to the early layers. As you add more layers, gradients tend to either explode or vanish before they reach the early weights, making deep networks hard to train. The skip connection provides a gradient highway: error signals can flow directly from the output back to any earlier layer without being filtered through intermediate transformations. This is why residual networks (ResNets) can be trained at depths that would be completely intractable for plain feedforward networks.

---

## 3. The KataGo Global Pooling Trick

This is where the architecture diverges from a standard ResNet, and it's the most important design decision for this particular game.

In each residual block, the full computation is:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    identity = x  # (B, C, 9, 9)

    # Branch 1: local convolutions (receptive field ~5x5 after two 3x3 convs)
    local = self.conv1(x)         # Conv(C,C,3,pad=1) -> BN -> ReLU
    local = self.bn1(local)
    local = F.relu(local)
    local = self.conv2(local)     # Conv(C,C,3,pad=1) -> BN
    local = self.bn2(local)

    # Branch 2: global pooling injection
    pooled = x.mean(dim=[2, 3])            # (B, C) — collapse spatial dims
    global_vec = self.global_mlp(pooled)   # Linear(C, C//8) -> ReLU -> Linear(C//8, C)
    global_broadcast = global_vec.unsqueeze(-1).unsqueeze(-1).expand_as(x)  # (B, C, 9, 9)

    # Combine
    out = F.relu(local + global_broadcast + identity)
    return out
```

The local branch is two stacked 3×3 convolutions. After both, each output position has a receptive field of roughly 5×5 cells — it can see its immediate neighbourhood but nothing beyond. This is fine for learning local patterns (whether a sub-board is being threatened, whether a row is almost complete) but fundamentally insufficient for Ultimate TTT's sending mechanic, which can couple any two sub-boards on the board.

The global branch solves this. Global average pooling collapses the entire (B, C, 9, 9) spatial tensor to a single (B, C) vector — the mean activation at each channel across all 81 positions. This vector is a summary of the full board state. Two linear layers transform it and then it's broadcast back out to (B, 128, 9, 9) and added to the local features.

The effect: every spatial position's features are conditioned on the global board state. A cell in the top-left can "know" what's happening in the bottom-right. This is the minimal, computationally cheap way to implement what the sending mechanic demands — the ability to reason about non-local consequences.

It's worth noting what this is *not*: it's not full self-attention (as in a transformer). Full self-attention allows every position to attend to every other position with position-specific weights. Global pooling is a rougher approximation — all positions get the same global summary injected. But it's far cheaper, and for a 9×9 board it appears sufficient.

The complete residual update written out is:

```
output = ReLU(F_local(x) + F_global(x) + x)
```

where F_local is the local convolutional path, F_global is the global pool → MLP → broadcast path, and x is the skip connection.

---

## 4. The Two Heads

After the trunk, the (B, 128, 9, 9) representation is shared between a policy head and a value head. Both read from the same features — this is the key property of shared representations. Whatever the trunk has learned about good and bad board positions benefits both heads simultaneously.

### Policy Head

```python
class PolicyHead(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.main_conv = nn.Conv2d(in_channels, 2, kernel_size=1, bias=False)
        self.main_bn   = nn.BatchNorm2d(2)
        self.main_fc   = nn.Linear(2 * 9 * 9, 81)  # 162 -> 81

        # Auxiliary head: predict opponent's likely next move
        self.opp_conv  = nn.Conv2d(in_channels, 2, kernel_size=1, bias=False)
        self.opp_bn    = nn.BatchNorm2d(2)
        self.opp_fc    = nn.Linear(2 * 9 * 9, 81)
```

The data flow:

```
(B, 128, 9, 9) -> Conv2d(128→2, k=1) -> BN -> ReLU -> flatten -> (B, 162) -> Linear(162→81) -> (B, 81)
```

Why 2 channels in the policy head? We want to compress the 128-channel spatial representation into something focused on spatial move discrimination — which cells are promising moves. Two channels is a standard choice from KataGo and AlphaZero: enough to capture two distinct types of spatial signal (e.g., "offensive" vs. "defensive" move quality), while keeping the subsequent linear layer small.

The output is 81 raw logits — one per possible move. These are not probabilities yet; the softmax (with illegal move masking) is applied externally during both inference and loss computation.

The second head — `opp_conv`, `opp_bn`, `opp_fc` — predicts the opponent's next move from the current position. It's an auxiliary training target, not used during play. The idea, from KataGo, is that predicting the opponent's policy forces the network to represent the game from both sides' perspectives simultaneously, which regularizes the shared trunk and generally improves value accuracy.

### Value Head

```python
class ValueHead(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv  = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.bn    = nn.BatchNorm2d(1)
        self.fc    = nn.Linear(81, 64)

        self.win_head       = nn.Linear(64, 1)   # -> tanh  -> win_value    (B, 1)
        self.score_head     = nn.Linear(64, 1)   # -> tanh  -> score_margin (B, 1)
        self.ownership_head = nn.Linear(64, 9)   # -> sigmoid -> ownership  (B, 9)
```

The value head uses only 1 channel after the initial convolution — unlike the policy head's 2. The reasoning is directional: the policy head needs spatial discrimination (which cells are better?), which benefits from more channels. The value head needs to collapse spatial information into a scalar judgment (who's winning?), which is better served by immediate dimensionality reduction. One channel, then flatten to 81, then a linear layer to 64 shared features.

From those 64 shared features, three outputs branch off:

**win_value** (through `tanh`) is the main output — a number in [−1, 1] representing the estimated probability that the current player wins, rescaled to the range used during training.

**score_margin** (through `tanh`) predicts the normalized score differential: (sub-boards won − sub-boards lost) / 9, expressed in [−1, 1]. This is an auxiliary target — it's only used during training. But it provides much richer information per game than win/loss alone. Every position now contributes not just "did we win?" but "how much did we win by?"

**ownership** (through `sigmoid`) predicts, for each of the 9 sub-boards, the probability that the current player ultimately wins that sub-board. Nine binary predictions per position. Again, auxiliary — but forces the network to develop an internal model of board control that substantially helps the value and policy predictions.

---

## 5. The Loss Function

The network is trained to minimize a weighted combination of five terms:

```
L_total = L_policy + λ_value · L_value + λ_score · L_score + λ_ownership · L_ownership + λ_opp · L_opp
```

with λ_value = 1.0, λ_score = 0.5, λ_ownership = 0.5, λ_opp = 0.15.

Let's go through each component.

### L_policy — Cross-Entropy Against MCTS Visit Distribution

```python
def policy_loss(logits, targets, legal_masks):
    masked_logits = logits.masked_fill(~legal_masks.bool(), float('-inf'))
    log_probs = F.log_softmax(masked_logits, dim=-1)
    log_probs = torch.nan_to_num(log_probs, nan=-100.0, neginf=-100.0)
    loss_per_sample = -(targets * log_probs).sum(dim=-1)
    return loss_per_sample.mean()
```

The target π(a) is the MCTS visit distribution — the fraction of simulations that explored each move. If MCTS ran 800 simulations and visited move 4 forty times, π(4) = 0.05. This is a *soft* probability distribution, not a one-hot label: MCTS might consider several moves roughly equally good, and the policy head learns to match that uncertainty.

The cross-entropy loss is: L = −∑_a π(a) · log p(a), where p(a) is the softmax of the network's logits. Minimizing this pushes the network's prior (used on future MCTS calls) toward the posterior that MCTS computed on this call.

Note the masking: illegal moves get logit −∞ before the softmax, ensuring they have probability 0 and contribute nothing to the gradient.

### L_value — MSE Against Game Outcome

```
L_value = mean((z − v)²)
```

z is the actual game outcome from the current player's perspective: +1 for a win, −1 for a loss, 0 for a draw. v is the network's win_value output (tanh, so bounded in [−1, 1]).

We use MSE rather than cross-entropy because v is a continuous output in [−1, 1], not a categorical distribution. MSE is the natural loss for regressing to a continuous target.

One subtlety: draws are *penalized* by construction. A drawn game gives z = 0, and if the network predicts v = 0, the loss is 0 — but if it predicts v = +0.5 (optimistically), loss = 0.25. The effect is that the network learns to treat draws as neutral rather than as near-wins. This anti-draw shaping (sometimes written z_draw = −0.5 in other implementations) prevents the network from developing a bias toward accepting draws in winning positions.

### L_score — MSE Against Normalized Score Margin

```
L_score = mean((s_target − score_margin)²)
```

s_target is the final score margin, normalized: if the current player won 6 sub-boards and the opponent won 3, s_target = (6 − 3)/9 ≈ 0.33. This auxiliary target provides graded signal — it's not just "did you win" but "by how much." It's especially informative early in training when the network hasn't yet learned to distinguish close games from lopsided ones.

### L_ownership — Binary Cross-Entropy Over Sub-Board Outcomes

```
L_ownership = −mean(∑_i [o_i · log(ô_i) + (1 − o_i) · log(1 − ô_i)])
```

o_i is 1 if the current player ultimately won sub-board i, 0 otherwise. ô_i is the network's sigmoid prediction for that sub-board. Binary cross-entropy is appropriate here because ownership is a per-sub-board binary outcome.

The value of this target is that it forces the network to think about territorial control at an intermediate resolution — not just the global win/loss, but which parts of the board each player dominated. A network that can accurately predict sub-board ownership from mid-game positions has necessarily learned a lot about the game's strategic structure.

### L_opp — Auxiliary Opponent Policy

```
L_opp = policy_loss(opp_policy_logits, opp_policy_targets, opp_legal_masks)
```

The same cross-entropy form as L_policy, but applied to the opponent policy head's predictions against the MCTS visit distribution from the opponent's perspective. The weight λ_opp = 0.15 is deliberately small — this is a regularizer, not the primary signal. It nudges the shared trunk to maintain representations that are useful for predicting opponent behavior, which implicitly trains it to model both sides of the game.

The combined effect of all five components is that a single forward pass through a single position contributes five distinct learning signals simultaneously, all of which update the shared trunk. This is the core efficiency argument for multi-task learning: the trunk is trained harder per game played than it would be with a single scalar outcome.

---

## 6. The Training Loop

The full pipeline runs in iterations. Each iteration has two phases.

In the **self-play phase**, the current network plays against itself: MCTS runs 800 simulations per move, guided by the network's policy and value outputs. At each position, we record the state tensor, the MCTS visit distribution (policy target), and the legal mask. After the game ends, we retroactively label each position with the game outcome (value target), final score margin, and per-sub-board ownership.

These labeled positions go into a **replay buffer** — a circular buffer of up to 500,000 positions. The circular structure ensures that older positions are eventually overwritten as newer, stronger games accumulate, while still providing a mix of positions from different stages of training. Training directly on the most recent games would cause the network to overfit to its current style of play; the buffer provides a diverse, approximately i.i.d. sample.

In the **training phase**, we sample random minibatches from the buffer and compute L_total for each. Gradients flow through all five loss components simultaneously, updating every layer of the network. After a fixed number of gradient steps, we checkpoint the network and evaluate it against the previous best version in an arena of test games. If the new network wins more than 55% of arena games, it becomes the new "best network" and generates self-play data in the next iteration.

```python
# Sketch of one training iteration
for _ in range(batches_per_iteration):
    batch = replay_buffer.sample(batch_size)
    output = network(batch.state_tensors)
    loss = compute_total_loss(output, batch,
                              lambda_value=1.0,
                              lambda_score=0.5,
                              lambda_ownership=0.5,
                              lambda_opp=0.15)
    optimizer.zero_grad()
    loss.total.backward()
    nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
    optimizer.step()
```

Gradient clipping (max norm 1.0) prevents any single catastrophic update from destabilizing a network that has been trained for many iterations. It's a cheap insurance policy that costs nothing when training is healthy and prevents disaster when it isn't.

---

The full network — 8 residual blocks of 128 channels each, with global pooling in every block — has roughly 3.5 million trainable parameters. That's modest by modern deep learning standards, but the game is also modest in its complexity. The architecture is sized to match the problem: enough capacity to capture the non-local dependencies that make Ultimate TTT hard, but not so overparameterized that training requires industrial-scale compute.

The key design decisions — the current-player encoding convention, the global pooling in every residual block, the five-component loss, the auxiliary heads — are each a response to a specific property of the game. Understanding why each piece is there is, in some sense, understanding the game itself.
