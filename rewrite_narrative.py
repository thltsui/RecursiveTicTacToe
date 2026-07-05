import re

with open('substack/essay1b_rl_background.md', 'r') as f:
    text = f.read()

# 1. Replace "## 2. The Bellman Equation\n\nBefore choosing actions, we need a way to evaluate positions. To do this, we first need a concept of a policy"
# with the new dimension 1 intro
text = text.replace(
    "## 2. The Bellman Equation\n\nBefore choosing actions, we need a way to evaluate positions. To do this, we first need a concept of a policy",
    "## 2. Two Dimensions of Value-Based Learning\n\nBefore choosing actions, we need a way to evaluate positions. Designing a reinforcement learning algorithm requires making decisions along two dimensions: what exactly we are estimating, and when we update those estimates.\n\n### 2.1 Dimension 1: What to Estimate (V vs Q)\n\nTo evaluate a position, we first need a concept of a policy"
)

# 2. Remove "## 3. Two Design Choices" entirely, all the way to "### 3.1 Dimension 1: When to Update (Monte Carlo vs TD)"
# and replace with "### 2.2 Dimension 2: When to Update (Monte Carlo vs TD)"
text = re.sub(
    r"## 3\. Two Design Choices\n\nGiven these two ways to think about value, we need to decide how to design our learning algorithm\. The algorithms we use are essentially determined by two dimensions: when we update our estimates, and what exactly we are estimating\.\n\n### 3\.1 Dimension 1: When to Update \(Monte Carlo vs TD\)",
    "### 2.2 Dimension 2: When to Update (Monte Carlo vs TD)",
    text
)

# 3. Remove "### 3.2 Dimension 2: What to Estimate (V vs Q)" block
to_remove = """### 3.2 Dimension 2: What to Estimate (V vs Q)

The second choice is what our neural network or table is actually trying to predict. 

V(s) is the value of a state. It is cheaper to store (one number per state), but requires you to simulate the game one step forward to choose actions.

Q(s, a) is the value of taking action a from state s. It is more expensive to store (one number per action per state), but it is directly actionable: you just read off the row for your current state and take the column with the highest value, without needing a model of the environment.

"""
text = text.replace(to_remove, "")

# 4. Bump headings 4, 5, 6 down to 3, 4, 5
text = text.replace("## 4. Policy Gradients and REINFORCE", "## 3. Policy Gradients and REINFORCE")
text = text.replace("## 5. Actor-Critic, GAE, and PPO", "## 4. Actor-Critic, GAE, and PPO")
text = text.replace("## 6. The Remaining Problem", "## 5. The Remaining Problem")

with open('substack/essay1b_rl_background.md', 'w') as f:
    f.write(text)

