# From Zero to AlphaZero: How PyTorch Builds a Computational Graph

Every post in this series so far has treated the network as a function: feed in a board state, get back a policy and a value. [PUCT — How AlphaZero Weighs Curiosity, Evidence, and Intuition](https://tthl.substack.com/p/from-zero-to-alphazero-puct-how-alphazero) leaned on that abstraction most directly, treating Q(s, a) and P(s, a) as given quantities without asking where either number actually comes from. This post opens that box: how does `loss.backward()`, one line at the end of `train_step()`, know how to compute gradients for every one of the network's parameters, across a trunk shared by five different loss terms, without anyone writing a single derivative by hand? The answer is the computational graph, and PyTorch builds one automatically, every time, as a side effect of running the forward pass. We walk through how that works using the actual network, loss function, and MCTS code from this repository rather than toy examples.

**Previous posts in this series:**

- [From Zero to AlphaZero: Ultimate Tic-Tac-Toe](https://tthl.substack.com/p/from-zero-to-alphazero-ultimate-tic)
- [From Zero to AlphaZero: Building the Ultimate Tic-Tac-Toe Engine in Python](https://tthl.substack.com/p/building-the-ultimate-tic-tac-toe)
- [From Zero to AlphaZero: The Reinforcement Learning Landscape](https://tthl.substack.com/p/from-zero-to-alphazero-the-reinforcement)
- [From Zero to AlphaZero: The Explore-Exploit Trade-off — The Bandit Algorithm Behind AlphaZero](https://tthl.substack.com/p/t3-the-slot-machine-problem-where)
- [From Zero to AlphaZero: PUCT — How AlphaZero Weighs Curiosity, Evidence, and Intuition](https://tthl.substack.com/p/from-zero-to-alphazero-puct-how-alphazero)

---

## Define-by-run: the graph is a byproduct of computation, not a separate step

A computational graph is a record of every operation applied to a tensor, with enough information attached to reverse each operation and work out how a small change to any input would change the final output. Every tensor in the graph is a node, and every operation, a convolution, an addition, a `tanh`, is an edge connecting its input tensors to its output tensor. Once we have the graph, computing gradients for the whole network just means walking it backward from the loss and applying the chain rule at each edge, which is exactly what `.backward()` does.

What makes PyTorch pleasant to work with is that this graph gets built dynamically, as the forward pass actually executes, rather than declared up front as a fixed structure - "define-by-run," in the usual terminology. Early TensorFlow, and Theano before it, worked the other way: you first constructed a static graph describing every operation symbolically, with no real data flowing through it yet, and only afterward fed data into that fixed structure to get results. That approach can run faster once the graph exists, since the whole computation is known in advance and can be optimized as a unit, but it makes anything data-dependent, a loop whose length depends on the input, a conditional branch chosen at runtime, considerably more awkward, since the graph itself cannot change shape once declared.

PyTorch skips the separate graph-construction phase entirely. Every time we call `network(x)`, PyTorch runs the actual Python code, convolutions, batch norm, ReLUs, the residual block's global pooling branch, and records each operation into a graph attached to the output tensor as it runs. Call the network again with a different-shaped input, or with a different number of MCTS simulations feeding a different batch size, and it just builds a new graph reflecting that specific execution. That flexibility matters here because the code producing our training targets, MCTS search, is genuinely dynamic: the tree it builds depends on the position and on however many simulations were run, and the network gets called many times within a single search with whatever shapes MCTS hands it. A static graph framework would need a separate mechanism to express that; for PyTorch, this is simply what running Python code does.

## Watching a real forward pass build the graph

The clearest place to watch this happen is `ResidualBlock`, in `02_network/residual_block.py`, the building block stacked eight times to form the network's trunk. Its forward method is short enough to trace by hand:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    identity = x  # (B, C, 9, 9)

    # Branch 1: local convolutions
    local = self.conv1(x)
    local = self.bn1(local)
    local = self.relu(local)
    local = self.conv2(local)
    local = self.bn2(local)

    # Branch 2: global pooling injection
    pooled = x.mean(dim=[2, 3])
    global_vec = self.global_mlp(pooled)
    global_broadcast = global_vec.unsqueeze(-1).unsqueeze(-1).expand_as(x)

    # Combine: skip connection + local + global
    out = self.relu(local + global_broadcast + identity)
    return out
```

Every line here does two things at once: it computes a tensor, and it silently extends the graph. `self.conv1(x)` returns a new tensor that also carries a reference back to `x` and to `conv1`'s weights, tagged with "this came from a 2D convolution," exactly what we need later to work out how the loss changes with respect to those weights. The same happens for `bn1`, `relu`, `conv2`, `bn2`. The global pooling branch builds its own chain in parallel: `x.mean(dim=[2, 3])` records that this came from averaging over those two dimensions, `self.global_mlp(pooled)` extends it through two linear layers and a ReLU, and the broadcast step records how a `(B, C)` vector expanded back out to `(B, C, 9, 9)`, so the reverse operation, summing the incoming gradient back down, is well defined. The final line, `local + global_broadcast + identity`, joins all three branches, so a residual block's contribution to the graph is an entire diamond: the input splits into two branches, an unbounded number of operations happen inside each, and everything reconverges at the addition.

Stack eight of these blocks, add the input convolution before and the two heads after, and the full forward pass through `UltimateTTTNetwork.forward()` (`02_network/network.py`) produces a single graph with thousands of nodes, built without any of that structure being declared anywhere. Nobody wrote down the gradient of a residual block in advance, PyTorch derived it by recording what actually happened and knowing how to invert each individual operation.

## The backward pass: one shared trunk, five loss terms pulling on it at once

Training uses a five-term loss, defined in `04_training/loss.py`:

```python
total = (lambda_policy * l_policy
         + lambda_value * l_value
         + lambda_score * l_score
         + lambda_ownership * l_ownership
         + lambda_opp * l_opp)
```

Each of the five terms comes from a different output of the network: `l_policy` from `policy_logits`, `l_value` from `win_value`, `l_score` from `score_margin`, `l_ownership` from `ownership`, `l_opp` from `opp_policy_logits`. `PolicyHead` and `ValueHead` (`02_network/policy_head.py`, `02_network/value_head.py`) are separate modules with entirely separate weights, but both take the same trunk output `h` as their input. In graph terms, `h` is a single node with five downstream paths leading out of it, one per loss term, and `total` is where all five paths get summed back together through the weighted sum above.

That is the concrete answer to the question we opened with. When `total.backward()` runs, it walks the graph backward starting from `total`, and at every node where multiple paths converge, or, running backward, diverge, it sums the contributions arriving from each path. For a parameter that belongs to `PolicyHead` alone, say `main_fc.weight`, only the `l_policy` path, scaled by `lambda_policy = 1.0`, contributes a gradient, since the loss simply does not depend on that weight through any other route. But for a parameter inside the trunk, say a convolution weight inside `trunk[3].conv1`, the gradient arriving there is the sum of five separate contributions, one from each loss term's backward path, each weighted by that term's lambda. This is what shared representations mean at the level of gradients: the trunk gets pulled by all five objectives at once, one gradient signal per term, summed automatically wherever the graph says paths converge. Nobody combines five gradients by hand; the sum in the forward pass, `total = lambda_policy * l_policy + ...`, is precisely what tells autograd to sum the corresponding backward paths.

`train_step()` in `04_training/trainer.py` shows the full cycle in five lines:

```python
optimizer.zero_grad()
breakdown.total.backward()
torch.nn.utils.clip_grad_norm_(network.parameters(), config.grad_clip_norm)
optimizer.step()
```

`zero_grad()` clears whatever gradients accumulated on each parameter from the previous step, since PyTorch accumulates gradients into `.grad` by default rather than overwriting them. That default is useful for gradient accumulation across micro-batches, but left unzeroed it would silently corrupt a normal training loop. `backward()` is the traversal we just walked through: it moves through the graph built during the forward pass and populates `.grad` on every leaf tensor with `requires_grad=True`, which for this network means every weight and bias in every layer. Gradient clipping then rescales that whole set of gradients if their combined norm exceeds `grad_clip_norm`, and `optimizer.step()` uses the now-populated `.grad` tensors to update the weights via Adam's update rule. The graph itself gets discarded immediately afterward; the next forward pass, on the next batch, builds an entirely new one from scratch, with no persistent graph object to manage or reset between iterations.

## `.detach()`: keeping a value without keeping its history

`compute_total_loss()` returns a `LossBreakdown` with six tensors: `total`, plus the five individual components. Only `total` ever gets passed to `.backward()`. The other five exist purely so the training loop can log them to `training_metrics.json` and print them to the console, and we detach them deliberately before returning:

```python
return LossBreakdown(
    total=total,
    policy=l_policy.detach(),
    value=l_value.detach(),
    score=l_score.detach(),
    ownership=l_ownership.detach(),
    opp_policy=l_opp.detach(),
)
```

`.detach()` returns a new tensor with the same value but no connection to the graph that produced it: no reference back to `policy_logits`, no record of the cross-entropy operation that computed it. This matters for two reasons. First, memory: any tensor still attached to a live graph keeps the entire graph, and everything it depends on, alive, since the graph needs all of it to run a backward pass later. `total` has to stay attached, because `backward()` has not been called yet, but the five components have already done their job for this step's gradient computation the moment `total` was formed as their weighted sum, so keeping them attached would hold onto graph memory for nothing. Second, correctness: even with unlimited memory, calling `.backward()` on one of the individual components by mistake later in the code would only update the parameters that specific loss term touches, silently skipping the other four. Detaching rules that out entirely, since a detached tensor has no graph behind it to walk in the first place.

The same pattern shows up in `03_mcts/search.py`, in the function that evaluates a leaf node during search:

```python
net_output = network.predict(node.state, device=device)
...
return net_output.win_value.item(), net_output.ownership.detach().cpu().numpy().flatten()
```

`.item()` on `win_value` already extracts a plain Python float with no graph attachment at all, and `network.predict()`, just below, already runs inside `torch.no_grad()`, so the `.detach()` on `ownership` here is technically redundant, there is no graph to detach from in the first place. We keep it anyway as a defensive habit: MCTS evaluates thousands of positions per move and never trains on any of them directly, so nothing coming out of this function should ever carry gradient history forward, and writing `.detach()` explicitly documents that intent even when the surrounding `no_grad()` already guarantees it.

## `torch.no_grad()`: turning graph-building off entirely for inference

`.detach()` strips history from a tensor after the fact; `torch.no_grad()` stops PyTorch from building any graph at all for the operations inside it, which is both faster and lighter on memory, since none of the bookkeeping a later `.backward()` call would need ever gets recorded in the first place. `UltimateTTTNetwork.predict()`, in `02_network/network.py`, wraps every single-position inference call in exactly this:

```python
def predict(self, state: 'GameState', device: str = 'cpu') -> NetworkOutput:
    ...
    self.eval()
    with torch.no_grad():
        output = self.forward(tensor)
    ...
```

This is the method MCTS calls every time it needs to evaluate a leaf node, which during a single move with 800 simulations can mean hundreds of network calls. None of these evaluations ever get backpropagated through directly; the network only learns later, in `train_step()`, from batches sampled out of the replay buffer, using the MCTS-derived policy targets and game outcomes as training data rather than gradients computed during search itself. Building a full computational graph for every one of those hundreds of forward passes, only to discard it immediately unused, would be pure waste, and `no_grad()` avoids that waste directly rather than leaving `.detach()` to clean up after the fact.

The paired `self.eval()` call is doing something different: it does not touch the graph at all, it changes the runtime behavior of specific layers, batch normalization in particular, which behaves differently depending on whether the model is in train or eval mode. During training, `BatchNorm2d` normalizes each batch using that batch's own statistics and updates a running average for later use. In eval mode it normalizes using the running average collected during training rather than the current batch's own statistics, which is what we want for a single-position inference call, where a batch of one has no meaningful statistics of its own. `trainer.py`'s main loop switches back and forth explicitly, calling `network.eval()` before self-play (line 226) and `network.train()` before the gradient-update loop (line 260): self-play needs each position evaluated consistently against the network's learned running statistics, while training needs batch normalization behaving as intended during learning.

## `torch.set_num_threads(1)`: a graph concern that is really a concurrency concern

One more line, from `web_app/app.py`:

```python
# Configure torch for thread safety and to prevent CPU thrashing
torch.set_num_threads(1)
```

This has nothing to do with the graph directly, but it follows from the same execution model. By default, PyTorch parallelizes individual operations, a single large convolution, say, across multiple CPU threads internally, which is a sensible default for a training script that wants to use every available core for one job at a time. The web app is a different situation: Flask can be handling multiple concurrent requests, each of which might trigger its own MCTS search, and each MCTS search calls `network.predict()` many times. If every one of those concurrent calls tried to spawn its own set of internal PyTorch threads, they would compete for the same CPU cores, and the resulting thread contention typically makes everything slower overall than running each call single-threaded, since the machine spends more time switching between threads than doing useful computation. Setting `torch.set_num_threads(1)` at startup tells PyTorch not to parallelize individual operations internally, leaving concurrency to Flask, one request thread at a time, rather than having two layers of the stack independently fighting over the same CPU cores.

## Why this is worth knowing, beyond satisfying curiosity

None of this changes how the network is trained or what it learns. But three things in this codebase only click once we know how the graph actually works. The five-term loss's shared trunk is a literal statement about which nodes in the graph have multiple incoming backward paths, and that is exactly why `lambda_value`, `lambda_score`, `lambda_ownership`, and `lambda_opp` in `TrainingConfig` matter: they are the scaling factors applied to each backward path before it sums into the trunk's shared gradient. The scattered `.detach()` calls mark the exact points where a value crosses from needed-for-training to needed-only-for-a-number-to-log-or-return. And `torch.no_grad()` around every MCTS evaluation is the difference between search running fast enough for 800 simulations per move and being prohibitively slow, since building and immediately discarding a graph for every one of those evaluations would cost real time and memory for nothing. `train_step()` reads as five lines that happen to work either way, but each of those lines is manipulating a real, inspectable graph object, and that is what makes the rest of this series's code legible.
