import re

with open('substack/essay1b_rl_background.md', 'r') as f:
    text = f.read()

# I will replace the headers with ### or #### and reorganise if needed.
# Actually, the text is already ordered: MC, TD, Q-learning, DQN.
# If I just change the headings in place, it will logically group them.

# Replace "## 4. Monte Carlo: Learn from Complete Games" with "### 3.1 When to Update: Monte Carlo vs TD\n\n#### Monte Carlo: Learn from Complete Games"
# But wait, the intro text for "How much to observe" is currently in Section 3.
# Let's do a multi-replace.

old_sec3 = """## 3. Two Design Choices

Given these two ways to think about value, we need to decide how to design our learning algorithm. There are two dimensions to this decision:

**What to estimate: V or Q?**

V(s) is the value of a state. If we store one number per state, V is a vector of length |S| — one entry per state. It is cheaper to store, but requires you to simulate the game one step forward to choose actions.

Q(s, a) is the value of taking action a from state s. It is a matrix of dimensions |S| × |A| — one entry per (state, action) pair. Q is more expensive to store, but it is directly actionable: you just read off the row for your current state and take the column with the highest value, without needing a model of the environment.

How much of the game to observe before updating?

This is the deeper design choice, and it determines the entire character of the algorithm. You are in state s_t. You have taken some actions. At some point you must use your observations to update your value estimates. The question is: how much of the future do you observe before making that update?

One extreme: you play the entire game to completion, collect the final outcome, and then work backward attributing credit. This is the Monte Carlo approach.

The other extreme: you take a single step, observe the immediate reward and the new state, and update immediately. This is the Temporal Difference approach.

Both are legitimate strategies. They are not approximations to each other — they are genuinely different algorithms with different statistical properties. To understand why both exist and when each is preferred, we need to look at them in detail."""

new_sec3 = """## 3. Two Design Choices

Given these two ways to think about value, we need to decide how to design our learning algorithm. The algorithms we use are essentially determined by two dimensions: when we update our estimates, and what exactly we are estimating.

### 3.1 Dimension 1: When to Update (Monte Carlo vs TD)

This determines the entire character of the learning algorithm. You are in state s_t. You have taken some actions. At some point you must use your observations to update your value estimates. The question is: how much of the future do you observe before making that update?

One extreme is to play the entire game to completion, collect the final outcome, and then work backward attributing credit. This is the Monte Carlo approach. 

The other extreme is to take a single step, observe the immediate reward and the new state, and update immediately. This is the Temporal Difference approach. 

Both are legitimate strategies with genuinely different statistical properties."""

text = text.replace(old_sec3, new_sec3)

text = text.replace("## 4. Monte Carlo: Learn from Complete Games", "#### Monte Carlo: Learn from Complete Games")
text = text.replace("## 5. TD Learning: Learn from One Step", "#### TD Learning: Learn from One Step")

q_intro = """### 3.2 Dimension 2: What to Estimate (V vs Q)

The second choice is what our neural network or table is actually trying to predict. 

V(s) is the value of a state. It is cheaper to store (one number per state), but requires you to simulate the game one step forward to choose actions.

Q(s, a) is the value of taking action a from state s. It is more expensive to store (one number per action per state), but it is directly actionable: you just read off the row for your current state and take the column with the highest value, without needing a model of the environment.

#### Q-Learning"""

text = text.replace("## 6. Q-Learning", q_intro)
text = text.replace("## 7. Deep Q-Networks (DQN)", "#### Deep Q-Networks (DQN)")

text = text.replace("## 8. Policy Gradients and REINFORCE", "## 4. Policy Gradients and REINFORCE")
text = text.replace("## 9. Actor-Critic, GAE, and PPO", "## 5. Actor-Critic, GAE, and PPO")
text = text.replace("## 10. The Remaining Problem", "## 6. The Remaining Problem")

with open('substack/essay1b_rl_background.md', 'w') as f:
    f.write(text)

