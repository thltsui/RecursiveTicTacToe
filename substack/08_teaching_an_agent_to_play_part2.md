# From Zero to AlphaZero: Teaching an Agent to Play, Part 2 — Deep Q-Learning

In Part 1, we built a Q-table for Ultimate Tic-Tac-Toe and watched it hit a wall: the table only knows what it has seen, and in a game this large almost every position is new. Seeing position A tells it nothing about position B, even when they differ by a single move, so a table that has to see every state before it can say anything useful about it was never going to scale. What we actually want is something that can compare a new position to similar ones it has already learned from, and generalise. That's exactly what a neural network gives us.

**Previous posts in this series:**

- [From Zero to AlphaZero: Ultimate Tic-Tac-Toe](https://tthl.substack.com/p/from-zero-to-alphazero-ultimate-tic)
- [From Zero to AlphaZero: Building the Ultimate Tic-Tac-Toe Engine in Python](https://tthl.substack.com/p/building-the-ultimate-tic-tac-toe)
- [From Zero to AlphaZero: The Reinforcement Learning Landscape](https://tthl.substack.com/p/from-zero-to-alphazero-the-reinforcement)
- [From Zero to AlphaZero: The Explore-Exploit Trade-off — The Bandit Algorithm Behind AlphaZero](https://tthl.substack.com/p/t3-the-slot-machine-problem-where)
- [From Zero to AlphaZero: PUCT — How AlphaZero Weighs Curiosity, Evidence, and Intuition](https://tthl.substack.com/p/from-zero-to-alphazero-puct-how-alphazero)
- *From Zero to AlphaZero: What Is a Computational Graph, Really?* (link TBD)
- [From Zero to AlphaZero: Teaching an Agent to Play, Part 1 — Tabular Q-Learning](https://tthl.substack.com/p/from-zero-to-alphazero-teaching-an)

---

## Neural Q-Learning: Function Approximation

The fix is to swap the table for a neural network that maps board states to Q-values. Networks generalise where tables cannot: positions that look similar as tensors get similar Q-value estimates, even ones the network has never seen.

![Tabular Q-learning versus a Q-network](images/fig07_qtable_vs_qnetwork.png)

We reuse the 7-channel `encode_state` tensor from [From Zero to AlphaZero: Building the Ultimate Tic-Tac-Toe Engine in Python](https://tthl.substack.com/p/building-the-ultimate-tic-tac-toe) as input, and the network outputs 81 Q-values, one per possible move.

### The Q-Network

The network is three convolutional layers over the (7, 9, 9) board tensor, flattened into a two-layer fully connected head producing 81 Q-values: [dqn/model.py, lines 12–58](https://github.com/thltsui/RecursiveTicTacToe/blob/8e19888892029b086e01376cfbe0d4f7aaace3e4/experiments/dqn/model.py#L12-L58).

Convolutions are a natural fit here because the board has real spatial structure: the 3×3 arrangement of sub-boards and the 3×3 cells within each sub-board both matter, and a convolutional filter can pick up a local pattern, a row inside a sub-board, a diagonal, no matter where on the 9×9 grid it happens to sit.

### The DQN Agent

Neural Q-learning (DQN) adds two ingredients on top of plain Q-learning: an **experience replay buffer** that stores past transitions and samples from them randomly, and a **target network** that gives training stable Q-value targets to aim at.

Training the network means giving it something to predict, the same idea as the table above, with a differentiable function standing in for the lookup. For each transition sampled from the replay buffer, the target is the reward received plus the discounted best Q-value the target network assigns to the next state: target = r + γ · max Q_target(s′, a′). The loss is the mean squared error between the network's own Q(s, a) for the action actually taken and that target, a number built entirely from tensors the network produced itself. Calling .backward() on that loss walks it back through the fully connected head and the convolutional layers, and gradient descent nudges every weight so the Q-value estimate moves a little closer to the target.

![DQN architecture](images/fig07_dqn_architecture.png)

The buffer stores past transitions and samples a random batch each step: [lines 34–74](https://github.com/thltsui/RecursiveTicTacToe/blob/8e19888892029b086e01376cfbe0d4f7aaace3e4/experiments/dqn/agent.py#L34-L74). Action selection is epsilon-greedy over the online network's own Q-values, masked to legal moves: [lines 123–135](https://github.com/thltsui/RecursiveTicTacToe/blob/8e19888892029b086e01376cfbe0d4f7aaace3e4/experiments/dqn/agent.py#L123-L135). And the training step itself, computing the TD target off the target network, the mean squared error against it, and the gradient step, is the mechanism described above: [lines 152–182](https://github.com/thltsui/RecursiveTicTacToe/blob/8e19888892029b086e01376cfbe0d4f7aaace3e4/experiments/dqn/agent.py#L152-L182).

### Training Loop

Training plays the agent against itself and a random opponent within the same loop, storing a transition and calling the training step after every move: [dqn/train.py, lines 41–82](https://github.com/thltsui/RecursiveTicTacToe/blob/8e19888892029b086e01376cfbe0d4f7aaace3e4/experiments/dqn/train.py#L41-L82). The outer loop wraps this across 20,000 episodes with the same kind of epsilon decay and periodic evaluation as the tabular case: [lines 89–160](https://github.com/thltsui/RecursiveTicTacToe/blob/8e19888892029b086e01376cfbe0d4f7aaace3e4/experiments/dqn/train.py#L89-L160).

### Results and the Ceiling

Training for 20,000 episodes (roughly 15 minutes on a CPU):

```
Ep  1000 | W 56.5%  D 6.0%  L 37.5%
Ep  5000 | W 64.0%  D 7.0%  L 29.0%
Ep 10000 | W 70.5%  D 8.5%  L 21.0%
Ep 20000 | W 73.5%  D 8.5%  L 18.0%
```

Progress is real, but slow, and the curve flattens out for several reasons that compound on each other.

**Sparse rewards.** The only strong signal is the terminal win or loss: every move in a forty-move game gets reward 0, and only the very last one gets plus or minus one, so the network has to credit forty decisions off a single scalar, exactly the credit-assignment problem [The Reinforcement Learning Landscape](https://tthl.substack.com/p/from-zero-to-alphazero-the-reinforcement) described. The small sub-board shaping reward takes the edge off, but does not come close to solving it.

**No lookahead.** The Q-network evaluates each position on its own and cannot reason about a sequence of moves, so a position that looks neutral might actually be a forced win three moves out, and there is no way for the network to see that without search.

**Single-opponent training.** Training against a random opponent caps how good the agent ever needs to get: it learns to punish random mistakes and never learns to handle an opponent that plays well.

**Bootstrapping instability.** Each update nudges an estimate toward another estimate, and in a large, sparse-reward space that chain is noisy and slow to settle.

## What Would Fix This

Each of these problems has an answer, and all four come straight out of the AlphaZero playbook.

**Sparse rewards → MCTS search targets.** Instead of training on terminal outcomes, AlphaZero trains the network on the output of a Monte Carlo tree search, which hands back a dense signal at every single position: these specific moves, right here, are worth exploring. The network learns directly from the search itself.

**No lookahead → explicit tree search.** MCTS builds a search tree at inference time; the network supplies value and policy estimates at each node, and the tree search does the actual lookahead. What a position looks like is the network's job, where the search should go next is the tree's job, and keeping those separate is the key architectural insight.

**Single-opponent training → self-play.** AlphaZero trains exclusively against itself, so as the network gets better, its opponent gets better at exactly the same rate. The agent always faces a competent opponent, and the training distribution keeps pace with the current level of play automatically.

**Bootstrapping instability → no more bootstrapping.** AlphaZero trains directly against the final game outcome z and the MCTS visit distribution π, not a TD target computed off a second, slightly-stale copy of the network. There is no target network and nothing to bootstrap from, so the instability that came from chasing a moving target disappears along with the mechanism that caused it.

The neural Q-network we built carries over directly. The `QNetwork` architecture, convolutional layers over the 7-channel tensor followed by a fully connected head, is essentially the same as AlphaZero's tower, minus the dual-head output of value and policy and the residual connections. The next step adds residual blocks and the dual head, wires it to MCTS, and trains it via self-play.

---

*Theory: [From Zero to AlphaZero: The Reinforcement Learning Landscape](https://tthl.substack.com/p/from-zero-to-alphazero-the-reinforcement) explains TD learning, Q-learning, and the credit-assignment problem that motivates what comes next. [From Zero to AlphaZero: The Explore-Exploit Trade-off — The Bandit Algorithm Behind AlphaZero](https://tthl.substack.com/p/t3-the-slot-machine-problem-where) and [From Zero to AlphaZero: PUCT — How AlphaZero Weighs Curiosity, Evidence, and Intuition](https://tthl.substack.com/p/from-zero-to-alphazero-puct-how-alphazero) cover the UCB and PUCT theory.*

---

*Full notebook: A companion notebook for this post (coming soon) will include the full training run, loss curves, visualisations of learned Q-values, and a head-to-head tournament between the tabular agent, the DQN agent, and random play.*
