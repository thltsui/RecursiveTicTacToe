# From Zero to AlphaZero: Teaching an Agent to Play, Part 1 — Tabular Q-Learning

*Prerequisites: [From Zero to AlphaZero: Building the Ultimate Tic-Tac-Toe Engine in Python](https://tthl.substack.com/p/building-the-ultimate-tic-tac-toe) (the game engine). You will need Python 3.9+, NumPy, and PyTorch. All code in this post builds on the `GameState`, `apply_move`, `get_legal_moves`, `encode_state`, and `get_legal_move_mask` functions from [From Zero to AlphaZero: Building the Ultimate Tic-Tac-Toe Engine in Python](https://tthl.substack.com/p/building-the-ultimate-tic-tac-toe).*

---

In our last essay, [What Is a Computational Graph, Really?](https://tthl.substack.com/p/from-zero-to-alphazero-what-is-a), we introduced PyTorch and explained how it builds a computational graph to fine-tune the parameters of a function through gradient descent on a specified loss function. With that setting the scene, we now want to address a more fundamental question: what is the loss function for evaluating a good Ultimate Tic-Tac-Toe move? This is a highly ambiguous question, because there are so many possible game states in Ultimate Tic-Tac-Toe that we cannot hand-craft a loss function for every one of them. We need a mechanism that assigns a loss to any game state, or at least to every game state our training ever covers. To find one, recall that our earlier essay on [the reinforcement learning landscape](https://tthl.substack.com/p/from-zero-to-alphazero-the-reinforcement) covered the mathematical framework behind Q-learning. Let's revisit that framework now and see how it works in practice.

**Previous posts in this series:**

- [From Zero to AlphaZero: Ultimate Tic-Tac-Toe](https://tthl.substack.com/p/from-zero-to-alphazero-ultimate-tic)
- [From Zero to AlphaZero: Building the Ultimate Tic-Tac-Toe Engine in Python](https://tthl.substack.com/p/building-the-ultimate-tic-tac-toe)
- [From Zero to AlphaZero: The Reinforcement Learning Landscape](https://tthl.substack.com/p/from-zero-to-alphazero-the-reinforcement)
- [From Zero to AlphaZero: The Explore-Exploit Trade-off — The Bandit Algorithm Behind AlphaZero](https://tthl.substack.com/p/t3-the-slot-machine-problem-where)
- [From Zero to AlphaZero: PUCT — How AlphaZero Weighs Curiosity, Evidence, and Intuition](https://tthl.substack.com/p/from-zero-to-alphazero-puct-how-alphazero)
- [From Zero to AlphaZero: What Is a Computational Graph, Really?](https://tthl.substack.com/p/from-zero-to-alphazero-what-is-a)

---

## Tabular Q-Learning: The State-Space Problem

Q-learning maintains a table mapping each (state, action) pair to its Q-value, the expected total future reward if that action is taken and play continues optimally afterward. Seen as a function, this table is the crudest possible parametrization: one free parameter for every (state, action) pair, with no sharing between them at all, essentially a one-layer network with one weight per input it will ever see. The update rule below is gradient descent on that absurdly high-dimensional function, one game at a time.

UTTT is a different scale entirely. The state space runs to roughly 3^81 ≈ 4.4 × 10^38 distinct board configurations, and a Q-table with one float per entry, even if we could find somewhere to put it, would need more storage than exists on Earth.

We can still run tabular Q-learning anyway, just to see how far it gets, knowing in advance it will only ever learn from positions it has actually visited and generalise to nothing else.

Here is how a Q-value actually gets learned from played games. Play one to the end, and the last move has a target we know exactly: the actual outcome, since nothing happens after it. Every earlier move does not have that luxury at the moment it needs an update, so it bootstraps instead: the target is whatever reward that move earned, usually zero since Ultimate Tic-Tac-Toe is decided only at the end, plus the table's own current best guess for the position that move led to. The code below replays each game in reverse, last move first, so the guess it bootstraps from is often freshly updated within the same game: the table already knows the finishing move's value by the time it computes the target for the move before it. Each update only nudges the table a fraction of the way toward its target, the learning rate decides how much, so an early move's value only becomes trustworthy after many games happen to revisit the same or a similar position. In a state space of 3^81 possible boards, almost no position gets visited twice.

![How a Q-table target gets built, one game](images/fig07_bellman_backup.png)

Every move's target has two possible parts: the reward earned by that specific move, and, if the game has not ended yet, a discounted estimate of what happens next. The reward part is known exactly the moment the move is played: 0 for a routine move, +0.02 if it wins a sub-board, and, if the move ends the game, +1 for a win, −1 for a loss, or 0 for a draw. Only a game-ending move stops there; every other move's target adds the table's own current best guess for the position it led to, discounted by γ, since there is no way yet to know how the game will actually turn out.

The update above already is the loss function from the opening, just applied to the smallest possible model. Treat each table entry Q(s, a) as its own free parameter, and the update is one step of gradient descent on the squared error between that entry and its target: L = (target − Q(s, a))². The gradient of that loss with respect to the one parameter it touches is −2(target − Q(s, a)), so nudging Q(s, a) toward the target by a fraction set by the learning rate is exactly what a gradient step produces. A table entry does not depend on any other parameters, so there is nothing to run the chain rule through. Last essay's machinery becomes necessary once Q(s, a) is computed by a function with layers in between, which is exactly what the network in Part 2 adds.

The implementation is a plain Python dictionary keyed by (state, action), defaulting to zero, updated with exactly the terminal-or-bootstrap rule described above, plus an epsilon-greedy rule that explores at rate ε and otherwise takes the highest-value legal move: [tabular_q/agent.py, lines 13–88](https://github.com/thltsui/RecursiveTicTacToe/blob/8e19888892029b086e01376cfbe0d4f7aaace3e4/experiments/tabular_q/agent.py#L13-L88).

Epsilon-greedy is what keeps this table honest while it learns. Most moves, the agent simply plays whatever the table currently rates highest, the greedy part. On a fraction ε of moves, though, it ignores the table and plays a uniformly random legal move instead. It's the same tension as the [From Zero to AlphaZero: The Explore-Exploit Trade-off — The Bandit Algorithm Behind AlphaZero](https://tthl.substack.com/p/t3-the-slot-machine-problem-where), now applied to a single lookup table rather than a slot machine: without some random moves mixed in, the agent only ever revisits positions its table already favours, and a move that picked up an unlucky Q-value early, before it had really been tested, would never get tried again to find out that estimate was wrong.

### Training loop

To train, we run self-play games where one agent plays as X and a copy (or a random agent) plays as O. After each move, we compute a reward and update:

Each training game runs both players through their own agent, records the (state, move) pairs each one visited, and applies the update to every visited pair in reverse once the game ends: [tabular_q/train.py, lines 40–92](https://github.com/thltsui/RecursiveTicTacToe/blob/8e19888892029b086e01376cfbe0d4f7aaace3e4/experiments/tabular_q/train.py#L40-L92).

### Evaluation

After training, we measure win rate against a random agent:

Evaluation plays a fixed number of games against a uniformly random opponent with exploration turned off, and reports the win, draw, and loss rate: [tabular_q/evaluate.py, lines 23–61](https://github.com/thltsui/RecursiveTicTacToe/blob/8e19888892029b086e01376cfbe0d4f7aaace3e4/experiments/tabular_q/evaluate.py#L23-L61).

### Results

Training the tabular agent for 5,000 games:

Training runs two such agents against each other for 5,000 games, decaying epsilon from 0.3 down to 0.05 over the run: [tabular_q/train.py, lines 99–152](https://github.com/thltsui/RecursiveTicTacToe/blob/8e19888892029b086e01376cfbe0d4f7aaace3e4/experiments/tabular_q/train.py#L99-L152).

Decaying epsilon from 0.3 to 0.05 over the run shifts that balance as training goes on. Early, when the table barely knows the game, moving randomly on 30% of moves floods it with a wide spread of positions to learn from, including ones the agent would never have picked on its own. By the end, with epsilon down to 5%, the agent mostly plays what the table has already learned to trust, only occasionally still spot-checking whether some path it dismissed early deserves another look.

Typical output:
```
Q-table size: 142,847 entries
Win/Draw/Loss vs random: 54.0% / 7.5% / 38.5%
```

Across 5,000 games the agent visits 142,847 (state, move) pairs, a tiny fraction of the full state space, and against a random opponent it manages just 54% wins, barely above the roughly 50% that random play achieves against itself. It has not generalised to unseen positions at all, which is the whole limitation of a lookup table.

The 54% win rate against random play suggests marginal improvement after training on 5,000 games, so the natural question is whether more training can get us to really good gameplay eventually. We can reliably infer that it is unlikely. First, the table keeps growing as more games are played, and memory usage grows with it. At some point, storing every possible transition becomes infeasible. Second, because of how numerous the possible game states are, every new game, with very high probability, lands on states that share nothing with the ones already visited. Given that the Q-table is completely state-dependent, seeing position A tells the table nothing about position B, even when they differ by a single move, so the win rate is most likely to grow at most linearly with the amount of training. This is akin to a student who memorises every game they have played, move by move, but never extracts any wisdom about which states are good and which moves are good. What we actually want is a student who can compare a new position against similar positions they have seen before, and extrapolate what the best move probably is. That is the motivation behind neural Q-learning.

---

*Theory: [From Zero to AlphaZero: The Reinforcement Learning Landscape](https://tthl.substack.com/p/from-zero-to-alphazero-the-reinforcement) explains TD learning, Q-learning, and the credit-assignment problem behind all of this.*

---

*Part 2, [Teaching an Agent to Play, Part 2: Deep Q-Learning](https://tthl.substack.com/p/from-zero-to-alphazero-teaching-an-037), picks up where the table breaks down and swaps it for a neural network.*
