# From Zero to AlphaZero: What Is a Computational Graph, Really?

Every post in this series so far has come back to the same core problem: building an agent that can actually play Ultimate Tic-Tac-Toe well. In the previous essays we covered how our agent will think, the algorithm it runs to compute the best next move, assuming it already has expert knowledge of what a good game state looks like ([PUCT — How AlphaZero Weighs Curiosity, Evidence, and Intuition](https://tthl.substack.com/p/from-zero-to-alphazero-puct-how-alphazero)). But how do we actually make the agent learn to recognise good game states in the first place, from nothing more than thousands of simulated games, say 1000 wins for X, 500 draws, 300 wins for O? Conceptually, this is done by assigning loss to the game plays that lost, and using that loss to inform better decision making after training. But in practice, how? That is the crux of this essay: how PyTorch, and its native computational graph structure, lets us actually teach a neural network to do better next time.

Before tracing that through the real eight-block network with five separate outputs, it's worth seeing the whole idea happen in the smallest possible example first: a single number in, a single number out.

**Previous posts in this series:**

- [From Zero to AlphaZero: Ultimate Tic-Tac-Toe](https://tthl.substack.com/p/from-zero-to-alphazero-ultimate-tic)
- [From Zero to AlphaZero: Building the Ultimate Tic-Tac-Toe Engine in Python](https://tthl.substack.com/p/building-the-ultimate-tic-tac-toe)
- [From Zero to AlphaZero: The Reinforcement Learning Landscape](https://tthl.substack.com/p/from-zero-to-alphazero-the-reinforcement)
- [From Zero to AlphaZero: The Explore-Exploit Trade-off — The Bandit Algorithm Behind AlphaZero](https://tthl.substack.com/p/t3-the-slot-machine-problem-where)
- [From Zero to AlphaZero: PUCT — How AlphaZero Weighs Curiosity, Evidence, and Intuition](https://tthl.substack.com/p/from-zero-to-alphazero-puct-how-alphazero)

---

## A one-parameter toy problem

Say we want a tiny model, $\hat y = \tanh(wx + b)$, to fit one training example: input $x = 2$, target $y = 1$, and some starting guess $w = 0.5$, $b = -1$. We measure how wrong we are with squared error, $L = (\hat y - y)^2$, and we want to know: which direction should we nudge $w$ and $b$ to make $L$ smaller?

Written as a sequence of operations, computing $\hat y$ from $x$ looks like this:

$$z = wx + b, \qquad \hat y = \tanh(z), \qquad L = (\hat y - y)^2$$

Three operations, three intermediate values. This is already a computational graph, just a very small one.

## Nodes are values, edges are "this operation produced this value from these inputs"

A node is nothing more mysterious than one of the tensors above: $x$, $w$, $b$, $z$, $\hat y$, $L$ are each a node. An edge is the fact, recorded at the moment an operation runs, that a particular node was produced by a particular operation acting on particular other nodes. When PyTorch computes $z = wx + b$, it does not just return the number; it returns a tensor that also carries a pointer back to $w$, a pointer back to $x$, a pointer back to $b$, and a tag saying "I came from a multiply, then an add." That bundle, value plus pointers plus operation tag, is the whole graph. Nothing is declared in advance: the graph only exists because each operation, as it ran, quietly wrote down what it had just done.

Plugging in numbers makes this concrete. With $w=0.5$, $x=2$, $b=-1$: $z = 0.5 \times 2 + (-1) = 0$. Then $\hat y = \tanh(0) = 0$. Then $L = (0 - 1)^2 = 1$. Three numbers, and three recorded edges: "$z$ came from multiplying $w$ and $x$, then adding $b$," "$\hat y$ came from applying $\tanh$ to $z$," "$L$ came from subtracting $y$ from $\hat y$ and squaring the result."

![The toy graph's forward pass (left) computes z, ŷ, and L in turn while recording three edges as it goes; the backward pass (right) walks the same graph in reverse, chaining local derivatives — ∂L/∂ŷ = −2, ∂ŷ/∂z = 1, ∂z/∂w = 2, ∂z/∂b = 1 — to reach ∂L/∂w = −4 and ∂L/∂b = −2.](images/fig_t5_graph_forward_backward.svg)

## Walking backward: the same graph, read in reverse

Training wants $\partial L/\partial w$ and $\partial L/\partial b$: how much would $L$ change if we nudged each parameter by a tiny amount? The graph makes this a matter of walking backward from $L$ to $w$ and $b$, multiplying local derivatives together at every edge along the way, which is exactly the chain rule.

Start at $L = (\hat y - y)^2$. Its local derivative with respect to its own input is $\partial L/\partial \hat y = 2(\hat y - y) = 2(0-1) = -2$. To compute this, the node needed to remember $\hat y$ and $y$, both of which it already had.

Next edge back: $\hat y = \tanh(z)$. Its local derivative is $\partial \hat y/\partial z = 1 - \tanh^2(z) = 1 - 0^2 = 1$. To compute this, the node needed to remember its own output, $\tanh(z)$, from the forward pass, which is exactly why PyTorch holds onto forward activations rather than discarding them the moment they're used. Chaining these two edges: $\partial L/\partial z = \partial L/\partial \hat y \times \partial \hat y/\partial z = -2 \times 1 = -2$.

Last edge: $z = wx + b$. This node has two parents, $w$ and $b$ (via the add) and, one step further back, $x$ (via the multiply). Its local derivatives are $\partial z/\partial w = x = 2$, $\partial z/\partial b = 1$, and $\partial z/\partial x = w = 0.5$. To compute $\partial z/\partial w$, the node needed to remember $x$; to compute $\partial z/\partial x$, it needed to remember $w$. Chaining once more: $\partial L/\partial w = \partial L/\partial z \times \partial z/\partial w = -2 \times 2 = -4$, and $\partial L/\partial b = \partial L/\partial z \times \partial z/\partial b = -2 \times 1 = -2$.

Gradient descent then updates $w \leftarrow w - \eta(-4)$ and $b \leftarrow b - \eta(-2)$ for some small learning rate $\eta$, moving both parameters in the direction that makes $L$ smaller. Nobody wrote $\partial L/\partial w = -4$ down as a formula ahead of time; it fell out of walking three remembered local derivatives backward through three recorded edges.

## What this buys us

Every node above needed to remember something specific from the forward pass to make its backward step possible: the squared-error node needed $\hat y$ and $y$, the $\tanh$ node needed its own output, the multiply-add node needed $w$ and $x$. That pattern, each operation quietly saving exactly what its own backward step will need, is the entire mechanism, whether the graph has three nodes or three thousand. A real network replaces $wx+b$ with a convolution and $\tanh$ with a whole residual block, but the graph itself is built the same way, one recorded edge at a time, and walked backward the same way, one local derivative at a time. That is what we will trace next, through the actual residual blocks in this project's network.
