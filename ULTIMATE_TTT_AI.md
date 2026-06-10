# Ultimate Tic-Tac-Toe AI — Claude Code Implementation Guide

## Project Overview

This project builds a self-learning AI for Ultimate Tic-Tac-Toe using an AlphaZero-style
architecture enhanced with domain-independent improvements from KataGo. The codebase is
structured as a YouTube lecture series — each file is one self-contained, teachable concept.
The repo structure IS the curriculum. Every function must be independently understandable.

---

## Non-Negotiable Design Rules

1. **One file = one concept = one YouTube episode.** No file should require understanding
   another file in the same folder to be taught independently.

2. **Explicit contracts everywhere.** Every function must have a full docstring specifying
   input shapes, output shapes, types, and what the function teaches conceptually.

3. **No hidden state.** Functions take inputs and return outputs. No global variables.
   No class attributes that persist unexpectedly across calls.

4. **No game-specific knowledge in the engine.** The network, MCTS, and training loop
   must be general. Only `01_game/` is allowed to know the rules of Ultimate TTT.

5. **PyTorch throughout.** All tensors are `torch.Tensor`. All networks are `nn.Module`.
   Use `float32` everywhere unless explicitly noted.

6. **Every file must be runnable standalone.** Each file ends with an
   `if __name__ == "__main__":` block that demonstrates the component with a minimal
   working example and prints/asserts correctness.

7. **Shapes in comments.** Every tensor operation must have a comment showing the shape
   after that operation, e.g. `# [B, 128, 9, 9]`.

---

## Repository Structure

```
ultimate-ttt-ai/
│
├── CLAUDE.md                    ← This file
├── requirements.txt
├── README.md
│
├── 01_game/
│   ├── __init__.py
│   ├── board.py                 # Episode 1
│   ├── rules.py                 # Episode 2
│   └── visualizer.py            # Episode 3
│
├── 02_network/
│   ├── __init__.py
│   ├── residual_block.py        # Episode 4
│   ├── policy_head.py           # Episode 5
│   ├── value_head.py            # Episode 6
│   └── network.py               # Episode 7
│
├── 03_mcts/
│   ├── __init__.py
│   ├── node.py                  # Episode 8
│   ├── search.py                # Episode 9
│   └── policy_target.py         # Episode 10
│
├── 04_training/
│   ├── __init__.py
│   ├── self_play.py             # Episode 11
│   ├── replay_buffer.py         # Episode 12
│   ├── loss.py                  # Episode 13
│   └── trainer.py               # Episode 14
│
├── 05_explainability/
│   ├── __init__.py
│   ├── gradcam.py               # Episode 15
│   ├── integrated_grads.py      # Episode 16
│   └── mcts_viz.py              # Episode 17
│
└── 06_evaluation/
    ├── __init__.py
    ├── elo.py                   # Episode 18
    ├── arena.py                 # Episode 19
    └── metrics.py               # Episode 20
```

---

## Episode Dependency Graph

Claude Code must enforce this dependency order. A file may only import from episodes
with lower numbers. No circular imports allowed.

```
Ep 1 (board) ──→ Ep 2 (rules) ──→ Ep 3 (visualizer)
                      │
                      ▼
Ep 4 (resblock) ──→ Ep 5 (policy_head) ──→ Ep 6 (value_head) ──→ Ep 7 (network)
                                                                         │
                                                                         ▼
Ep 8 (node) ──→ Ep 9 (search) ──→ Ep 10 (policy_target)
                                            │
                                            ▼
Ep 11 (self_play) ──→ Ep 12 (replay_buffer) ──→ Ep 13 (loss) ──→ Ep 14 (trainer)
                                                                         │
                                                                         ▼
Ep 15 (gradcam) ──→ Ep 16 (integrated_grads) ──→ Ep 17 (mcts_viz)
                                                                         │
                                                                         ▼
Ep 18 (elo) ──→ Ep 19 (arena) ──→ Ep 20 (metrics)
```

---

## requirements.txt

```
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
tqdm>=4.65.0
pytest>=7.3.0
```

---

## Episode 1 — `01_game/board.py`

**YouTube Title:** "Encoding a Game State as a Tensor — The Foundation of Board Game AI"

**Concept taught:** How to represent game state as a multi-channel tensor for neural network
input. Introduces the current-player convention from AlphaZero.

### Game State Representation

The game state is a Python dataclass (not a tensor — raw game logic stays as Python):

```python
@dataclass
class GameState:
    # Shape notation: (sub_board_idx, cell_idx) both in range [0,8]
    cells: np.ndarray          # shape (9, 9) — 0=empty, 1=player1, -1=player2
    sub_board_results: np.ndarray  # shape (9,) — 0=ongoing, 1=p1 won, -1=p2 won, 2=draw
    active_sub_board: int      # 0-8, or -1 meaning "free choice" (sent to won/drawn board)
    current_player: int        # 1 or -1
    move_count: int            # total moves played
    is_terminal: bool
    winner: int                # 1, -1, 0 (draw), or None (ongoing)
```

### Tensor Encoding — 9×9×7

`encode_state(game_state: GameState) -> torch.Tensor`

Returns shape `(7, 9, 9)` — 7 channels, always from current player's perspective.
The board is ALWAYS encoded as if the current player is "player 1".
If it is player -1's turn, flip the board before encoding.

Channel layout:

```
Channel 0: Current player's pieces        — 1.0 where current player has a piece, else 0.0
Channel 1: Opponent's pieces              — 1.0 where opponent has a piece, else 0.0
Channel 2: Active sub-board mask          — 1.0 for all 9 cells of the active sub-board,
                                            1.0 for ALL cells if active_sub_board == -1
Channel 3: Sub-boards won by current player — 1.0 for all 9 cells of sub-boards current player won
Channel 4: Sub-boards won by opponent       — 1.0 for all 9 cells of sub-boards opponent won
Channel 5: Drawn/dead sub-boards          — 1.0 for all 9 cells of drawn sub-boards
Channel 6: Constant turn indicator        — all 0.0 if current player is player 1 in raw game,
                                            all 1.0 if current player is player -1 in raw game
```

The spatial layout maps (sub_board_idx, cell_idx) to (row, col) in 9×9 as follows:
- sub_board_idx maps to the 3×3 block position: row_block = sub_board_idx // 3, col_block = sub_board_idx % 3
- cell_idx maps within the block: row_cell = cell_idx // 3, col_cell = cell_idx % 3
- Final 9×9 position: row = row_block * 3 + row_cell, col = col_block * 3 + col_cell

### Functions Required

```python
def create_initial_state() -> GameState:
    """Create a fresh game. Player 1 goes first. Active sub-board is -1 (free choice)
    on the very first move since no sub-board has been forced yet.
    Actually: first move is free choice since no move has directed us anywhere.
    Active sub-board starts as -1."""

def encode_state(state: GameState) -> torch.Tensor:
    """Encode game state as tensor for neural network input.
    
    Always encodes from current player's perspective (current player = 'me').
    This means the network sees the same representation regardless of which
    player it is — it always sees itself as the positive player.
    
    Args:
        state: GameState dataclass
    
    Returns:
        Tensor of shape (7, 9, 9), dtype=torch.float32
    """

def decode_move(move_idx: int) -> tuple[int, int]:
    """Convert flat move index (0-80) to (sub_board_idx, cell_idx).
    move_idx = sub_board_idx * 9 + cell_idx
    
    Returns:
        (sub_board_idx, cell_idx) both in range [0, 8]
    """

def encode_move(sub_board_idx: int, cell_idx: int) -> int:
    """Convert (sub_board_idx, cell_idx) to flat move index (0-80)."""
```

### Standalone Test

The `__main__` block must:
1. Create initial state
2. Encode it as tensor
3. Assert shape is `(7, 9, 9)`
4. Assert channel 6 is all zeros (player 1 to move)
5. Assert channel 2 is all ones (free choice on first move)
6. Print the tensor shape and channel sums

---

## Episode 2 — `01_game/rules.py`

**YouTube Title:** "Game Logic — Legal Moves, Win Detection, and State Transitions"

**Concept taught:** How to implement complete game rules as pure functions.
The directed-move mechanic of Ultimate TTT is the key teaching moment.

### Functions Required

```python
def get_legal_moves(state: GameState) -> list[int]:
    """Return list of legal move indices (0-80).
    
    Key rules:
    - You must play in the sub-board indicated by active_sub_board
    - If active_sub_board == -1, you may play in ANY non-terminal sub-board
    - You may never play in an already-occupied cell
    - You may never play in a sub-board that is already won or drawn
    
    Returns:
        List of flat move indices. Never empty unless is_terminal is True.
    """

def apply_move(state: GameState, move_idx: int) -> GameState:
    """Apply a move and return the NEW game state. Never mutates input state.
    
    Steps (in order):
    1. Validate move is legal
    2. Place piece in cell
    3. Check if that sub-board is now won or drawn
    4. Update sub_board_results if needed
    5. Check if the meta-board is now won or drawn (terminal check)
    6. Determine next active_sub_board:
       - next_sub_board = cell_idx of the move just played
       - If that sub-board is already terminal (won/drawn), set active_sub_board = -1
       - Otherwise set active_sub_board = next_sub_board
    7. Flip current_player
    8. Increment move_count
    
    Args:
        state: Current game state (not mutated)
        move_idx: Flat move index 0-80
    
    Returns:
        New GameState after move applied
    
    Raises:
        ValueError: If move_idx is not in get_legal_moves(state)
    """

def check_sub_board_winner(cells_3x3: np.ndarray) -> int:
    """Check if a 3x3 sub-board has a winner.
    
    Args:
        cells_3x3: shape (3, 3) with values 1, -1, or 0
    
    Returns:
        1 if player 1 wins, -1 if player 2 wins, 2 if draw, 0 if ongoing
    """

def check_meta_winner(sub_board_results: np.ndarray) -> int:
    """Check if the meta-board (9 sub-board results) has a winner.
    Treat sub_board_results as a 3x3 grid where:
    - 1 = player 1 won this sub-board
    - -1 = player 2 won this sub-board
    - 2 = draw (this sub-board does not count for either player in meta)
    - 0 = ongoing
    
    Returns:
        1, -1, 2 (full draw), or 0 (ongoing)
    """

def get_legal_move_mask(state: GameState) -> torch.Tensor:
    """Return binary mask of legal moves for network output masking.
    
    Returns:
        Tensor of shape (81,), dtype=torch.float32
        1.0 for legal moves, 0.0 for illegal moves
    """
```

### Standalone Test

The `__main__` block must:
1. Create initial state, verify 81 legal moves (free choice on first move)
2. Apply a move to sub-board 4 cell 0, verify next active_sub_board is 0
3. Simulate a full random game to terminal state
4. Verify terminal state has a winner or draw
5. Verify legal moves is empty on terminal state

---

## Episode 3 — `01_game/visualizer.py`

**YouTube Title:** "Visualizing the Board — Making the Game State Human-Readable"

**Concept taught:** How to render complex nested game states for debugging and teaching.

### Functions Required

```python
def render_board_ascii(state: GameState) -> str:
    """Return ASCII art of the full 9×9 board.
    
    Format: 9×9 grid with sub-board boundaries drawn as thicker lines.
    Use 'X' for player 1, 'O' for player 2, '.' for empty.
    Highlight the active sub-board with brackets or asterisks.
    Show sub-board win status (X wins shown as large X across sub-board etc.)
    
    Example output format:
    
    Sub-boards: [0][1][2]
                [3][4][5]
                [6][7][8]
    
    X . . | . . . | . . .
    . . . | . . . | . . .
    . . . | . . . | . . .
    ------+-------+------
    . . . |*. . .*| . . .   ← active sub-board shown with *
    . . . |*. . .*| . . .
    . . . |*. . .*| . . .
    ------+-------+------
    . . . | . . . | . . .
    . . . | . . . | . . .
    . . . | . . . | . . .
    """

def render_tensor_channels(tensor: torch.Tensor) -> str:
    """Render all 7 channels of an encoded state tensor as ASCII.
    
    Shows each channel as a 9x9 grid of 0s and 1s.
    Labels each channel with its meaning.
    Useful for debugging encode_state().
    
    Args:
        tensor: shape (7, 9, 9)
    """

def render_heatmap(state: GameState, values: torch.Tensor, title: str) -> None:
    """Render a heatmap overlaid on the board using matplotlib.
    
    Used by explainability modules to show Grad-CAM, visit counts, etc.
    
    Args:
        state: Current game state (for board context)
        values: shape (9, 9) or (81,) — heatmap values, will be normalized to [0,1]
        title: Plot title
    
    Saves to file and displays if possible. Never blocks execution.
    """

def render_mcts_visits(state: GameState, visit_counts: dict[int, int]) -> None:
    """Render MCTS visit counts as a heatmap over the board.
    
    Args:
        state: Current game state
        visit_counts: Dict mapping move_idx to visit count
    """
```

---

## Episode 4 — `02_network/residual_block.py`

**YouTube Title:** "The Residual Block with Global Pooling — KataGo's Key Innovation"

**Concept taught:** ResNets and why skip connections work. The KataGo global pooling
injection that lets local convolutions be conditioned on global board state.
This is the most technically dense episode — spend extra time on the global pooling math.

### Architecture

Each residual block has TWO branches that are added together:

**Branch 1 (local):** Standard residual path
```
input → Conv2d(C, C, 3, padding=1) → BatchNorm → ReLU → Conv2d(C, C, 3, padding=1) → BatchNorm
```

**Branch 2 (global pooling injection — KataGo innovation):**
```
input → GlobalAvgPool → Linear(C, C//8) → ReLU → Linear(C//8, C) → broadcast to (B, C, 9, 9)
```

Final output:
```
ReLU(Branch1 + Branch2 + input)   ← input is the skip connection
```

The global pooling branch allows every spatial position to be aware of what is
happening across the entire board — critical for a game where local moves have
global consequences (the directed-move mechanic).

### Class Required

```python
class ResidualBlock(nn.Module):
    """A single residual block with global average pooling injection.
    
    This is the core building block of the network trunk. It preserves
    spatial dimensions (9×9) while allowing global context to influence
    local features.
    
    The global pooling branch is the key KataGo domain-independent improvement:
    it allows the network to condition local features on the global board state
    without any game-specific knowledge being hardcoded.
    
    Args:
        channels: Number of feature channels (C). Must be consistent throughout trunk.
        global_channels: Size of global pooling bottleneck. Default: channels // 8.
    
    Input:  Tensor of shape (B, C, 9, 9)
    Output: Tensor of shape (B, C, 9, 9)  ← spatial dims preserved
    """
    
    def __init__(self, channels: int, global_channels: int = None):
        ...
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (B, C, 9, 9)
        
        Returns:
            shape (B, C, 9, 9)
        
        Internal shapes (must be shown in comments):
            local = conv_path(x)           # (B, C, 9, 9)
            pooled = x.mean(dim=[2,3])     # (B, C)
            global_vec = mlp(pooled)       # (B, C)
            global_broadcast = global_vec.unsqueeze(-1).unsqueeze(-1).expand_as(x)  # (B, C, 9, 9)
            out = relu(local + global_broadcast + x)  # (B, C, 9, 9)
        """
```

### Standalone Test

The `__main__` block must:
1. Create a ResidualBlock with channels=128
2. Pass a random tensor of shape (4, 128, 9, 9) through it
3. Assert output shape is (4, 128, 9, 9)
4. Assert output is not equal to input (the block is doing something)
5. Count and print the number of trainable parameters
6. Verify gradients flow through both branches

---

## Episode 5 — `02_network/policy_head.py`

**YouTube Title:** "The Policy Head — Teaching the Network to Choose Moves"

**Concept taught:** How to convert a spatial feature map into a probability distribution
over moves. The auxiliary opponent policy head for regularization (KataGo improvement).
How illegal move masking works and why it's applied AFTER the logits, not before.

### Architecture

```
input (B, C, 9, 9)
    → Conv2d(C, 2, kernel=1)          # (B, 2, 9, 9) — compress to 2 channels
    → BatchNorm2d(2)
    → ReLU
    → Flatten                          # (B, 162)
    → Linear(162, 81)                  # (B, 81) — policy logits
```

The auxiliary opponent head has the SAME architecture but separate weights:
```
input (B, C, 9, 9) → [same structure] → (B, 81) opponent policy logits
```

Illegal move masking (CRITICAL — must be taught carefully):
```python
# WRONG: masking before softmax causes numerical issues
probs = softmax(logits * legal_mask)   # DO NOT DO THIS

# CORRECT: set illegal logits to -inf, then softmax
masked_logits = logits.masked_fill(~legal_mask.bool(), float('-inf'))
probs = softmax(masked_logits, dim=-1)
```

### Class Required

```python
class PolicyHead(nn.Module):
    """Dual policy head: current player moves + opponent move prediction (auxiliary).
    
    The main policy output is used for move selection during play and MCTS.
    The auxiliary opponent policy is used only during training as a regularizer
    (KataGo domain-independent improvement: predicts opponent's likely next move
    to force the network to model both perspectives).
    
    During inference, only call forward() and use policy_logits.
    During training, use both policy_logits and opp_policy_logits.
    
    Args:
        in_channels: Number of channels from trunk (C)
    
    Input:  Tensor of shape (B, C, 9, 9)
    Output: Tuple of (policy_logits, opp_policy_logits), each shape (B, 81)
            These are RAW LOGITS. Apply legal move masking and softmax externally.
    """
    
    def __init__(self, in_channels: int):
        ...
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ...

def apply_legal_mask(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Apply legal move mask to policy logits and return probability distribution.
    
    This is a standalone function (not a method) so it can be taught independently
    and used both during play and training.
    
    Args:
        logits: Raw policy logits, shape (B, 81) or (81,)
        legal_mask: Binary mask, shape (B, 81) or (81,), 1.0=legal, 0.0=illegal
    
    Returns:
        Probability distribution over legal moves, same shape as logits.
        Illegal moves have probability 0.0.
        Legal moves sum to 1.0.
    
    Raises:
        ValueError: If no legal moves exist (all mask values are 0)
    """
```

---

## Episode 6 — `02_network/value_head.py`

**YouTube Title:** "The Value Head — Teaching the Network to Evaluate Positions"

**Concept taught:** Multi-task learning from a single position. How auxiliary targets
(score margin, sub-board ownership) give richer training signal than win/loss alone.
Why tanh is the right activation for value (bounded [-1, 1]).

### Architecture

The value head has ONE shared trunk then THREE separate output heads:

```
input (B, C, 9, 9)
    → Conv2d(C, 1, kernel=1)    # (B, 1, 9, 9) — compress to 1 channel
    → BatchNorm2d(1)
    → ReLU
    → Flatten                    # (B, 81)
    → Linear(81, 64)
    → ReLU
    → shared_features            # (B, 64)
         ├── Linear(64, 1) → Tanh  → win_value      (B, 1)   ← main output
         ├── Linear(64, 1) → Tanh  → score_margin   (B, 1)   ← auxiliary
         └── Linear(64, 9) → Sigmoid → ownership    (B, 9)   ← auxiliary
```

**win_value:** Probability current player wins. Range [-1, 1].
  -1 = certain loss, 0 = equal, 1 = certain win.

**score_margin:** Predicted sub-boards won minus sub-boards lost, normalized to [-1, 1].
  Formula: `(own_sub_boards_won - opp_sub_boards_won) / 9`

**ownership:** For each of the 9 sub-boards, probability current player will win it
  by game end. Range [0, 1] per sub-board. Shape (B, 9).

### Class Required

```python
class ValueHead(nn.Module):
    """Multi-output value head with auxiliary prediction targets.
    
    Main output: win_value — used for MCTS and move selection
    Auxiliary outputs: score_margin, ownership — used only during training
    
    The auxiliary outputs provide richer training signal per game
    (KataGo domain-independent improvement). They force the network to
    develop an internal model of board control, not just win probability.
    
    Args:
        in_channels: Number of channels from trunk (C)
    
    Input:  Tensor of shape (B, C, 9, 9)
    Output: ValueHeadOutput namedtuple with fields:
            - win_value:    (B, 1),  range [-1, 1]
            - score_margin: (B, 1),  range [-1, 1]
            - ownership:    (B, 9),  range [0, 1] per element
    """
    
    def __init__(self, in_channels: int):
        ...
    
    def forward(self, x: torch.Tensor) -> 'ValueHeadOutput':
        ...

@dataclass
class ValueHeadOutput:
    win_value: torch.Tensor     # (B, 1)
    score_margin: torch.Tensor  # (B, 1)
    ownership: torch.Tensor     # (B, 9)
```

---

## Episode 7 — `02_network/network.py`

**YouTube Title:** "Assembling the Full Network — Trunk, Policy Head, and Value Head"

**Concept taught:** How to compose the residual trunk with dual heads. The concept
of shared representations — the trunk learns features useful for BOTH policy and value.
How to count parameters and think about network capacity.

### Architecture Summary

```
Input: (B, 7, 9, 9)
    → Conv2d(7, 128, kernel=3, padding=1) → BN → ReLU   # (B, 128, 9, 9)
    → ResidualBlock(128) × num_blocks                    # (B, 128, 9, 9)
    → [splits into two heads]
    
    Policy Head → (policy_logits: B×81, opp_policy_logits: B×81)
    Value Head  → (win_value: B×1, score_margin: B×1, ownership: B×9)
```

### Class Required

```python
class UltimateTTTNetwork(nn.Module):
    """Complete neural network for Ultimate TTT.
    
    Architecture:
        - Input conv: 7 channels → C channels
        - Trunk: num_blocks residual blocks, each with global pooling
        - Policy head: outputs current + opponent move distributions
        - Value head: outputs win probability + auxiliary targets
    
    Default hyperparameters (justified by game complexity vs AlphaZero):
        - channels (C) = 128   (AlphaZero uses 256, our game is simpler)
        - num_blocks = 8        (AlphaZero uses 20, sufficient for 9×9)
    
    Args:
        channels: Feature channels throughout trunk. Default: 128.
        num_blocks: Number of residual blocks. Default: 8.
    
    Input:  Tensor of shape (B, 7, 9, 9)
    Output: NetworkOutput namedtuple
    """
    
    def __init__(self, channels: int = 128, num_blocks: int = 8):
        ...
    
    def forward(self, x: torch.Tensor) -> 'NetworkOutput':
        """
        Args:
            x: Board state tensor, shape (B, 7, 9, 9)
        
        Returns:
            NetworkOutput with fields:
            - policy_logits:     (B, 81)
            - opp_policy_logits: (B, 81)
            - win_value:         (B, 1)
            - score_margin:      (B, 1)
            - ownership:         (B, 9)
        """
    
    def predict(self, state: 'GameState', device: str = 'cpu') -> 'NetworkOutput':
        """Single-state inference with no gradient tracking.
        
        Convenience method for use during MCTS. Handles:
        - Encoding the game state to tensor
        - Adding batch dimension
        - Moving to device
        - Running forward pass in torch.no_grad()
        - Removing batch dimension from outputs
        
        Args:
            state: GameState (not a tensor)
            device: 'cpu' or 'cuda'
        
        Returns:
            NetworkOutput with batch dimension removed (shapes without B)
        """

@dataclass
class NetworkOutput:
    policy_logits:     torch.Tensor  # (B, 81) or (81,)
    opp_policy_logits: torch.Tensor  # (B, 81) or (81,)
    win_value:         torch.Tensor  # (B, 1)  or (1,)
    score_margin:      torch.Tensor  # (B, 1)  or (1,)
    ownership:         torch.Tensor  # (B, 9)  or (9,)
```

### Standalone Test

The `__main__` block must:
1. Create network with defaults
2. Print total parameter count
3. Run a forward pass with batch size 4
4. Assert all output shapes
5. Verify policy logits are not all equal (network is not degenerate)
6. Run `predict()` on a single GameState and verify output shapes without batch dim

---

## Episode 8 — `03_mcts/node.py`

**YouTube Title:** "The MCTS Node — Exploration vs Exploitation with UCT"

**Concept taught:** The UCT (Upper Confidence Bound for Trees) formula. Why we need
to balance exploring unknown moves vs exploiting known good moves. How visit counts
and value estimates are maintained per node.

### The UCT Formula (PUCT variant used by AlphaZero)

```
PUCT(s, a) = Q(s, a) + c_puct · P(s, a) · sqrt(N(s)) / (1 + N(s, a))

Where:
    Q(s, a)   = mean value of action a from state s (win rate estimate)
    P(s, a)   = prior probability from neural network policy
    N(s)      = total visits to parent node s
    N(s, a)   = visits to child via action a
    c_puct    = exploration constant (default 1.0, tune between 0.5 and 2.0)
```

Higher PUCT → more attractive to visit. The formula naturally balances:
- Exploitation: `Q(s, a)` favors moves with high observed win rate
- Exploration: `P(s, a) · sqrt(N(s)) / (1 + N(s, a))` favors unvisited moves with high prior

### Class Required

```python
class MCTSNode:
    """A single node in the Monte Carlo search tree.
    
    Each node represents a game state. It stores statistics about
    how often each action has been explored and what value was observed.
    
    Attributes:
        state: The GameState this node represents
        parent: Parent MCTSNode or None if root
        move_from_parent: The move_idx that led to this node (None for root)
        children: Dict mapping move_idx → MCTSNode
        
        # Per-action statistics (indexed by move_idx 0-80)
        N: Dict[int, int]    — visit count per action
        W: Dict[int, float]  — total value per action  
        Q: Dict[int, float]  — mean value = W[a] / N[a] (0 if N[a]==0)
        P: Dict[int, float]  — prior probability from network
        
        # Node statistics
        visit_count: int     — total visits to this node (sum of N values)
        is_expanded: bool    — whether children have been created
        is_terminal: bool    — whether this is a terminal game state
    """
    
    def __init__(
        self,
        state: 'GameState',
        parent: 'MCTSNode | None' = None,
        move_from_parent: int | None = None,
        prior: float = 0.0
    ):
        ...
    
    def puct_score(self, move_idx: int, c_puct: float = 1.0) -> float:
        """Compute PUCT score for a specific action.
        
        Args:
            move_idx: The action to score
            c_puct: Exploration constant
        
        Returns:
            PUCT score (higher = more attractive to visit)
        """
    
    def select_child(self, c_puct: float = 1.0) -> tuple[int, 'MCTSNode']:
        """Select the child with the highest PUCT score.
        
        Only selects among expanded children (children dict).
        Assumes node is already expanded.
        
        Returns:
            (move_idx, child_node) of the best child
        
        Raises:
            ValueError: If node has no children (not expanded or terminal)
        """
    
    def expand(self, prior_probs: torch.Tensor, legal_moves: list[int]) -> None:
        """Create child nodes for all legal moves.
        
        Args:
            prior_probs: Network policy output, shape (81,)
                         Values for illegal moves are ignored.
            legal_moves: List of legal move indices
        
        Sets is_expanded = True.
        Creates MCTSNode children with prior probabilities from network.
        Does NOT apply Dirichlet noise here — that happens at the root in search.py
        """
    
    def backup(self, value: float) -> None:
        """Backpropagate value up to the root.
        
        Updates W and N for the edge that led to this node,
        then recursively calls parent.backup(-value) because
        value is always from the perspective of the player who
        just moved, so we negate when going up to parent.
        
        Args:
            value: Outcome value from this node's perspective (+1 win, -1 loss)
        """
    
    def get_visit_counts(self) -> dict[int, int]:
        """Return {move_idx: visit_count} for all visited children."""
    
    def is_leaf(self) -> bool:
        """Return True if node has not been expanded yet."""
```

---

## Episode 9 — `03_mcts/search.py`

**YouTube Title:** "Monte Carlo Tree Search — The Four Phases"

**Concept taught:** The complete MCTS loop: Select → Expand → Evaluate → Backup.
How the neural network guides the search. Dirichlet noise for exploration at root.
The temperature parameter for move selection.

### The Four Phases

```
1. SELECT:   Start at root. Follow highest PUCT scores until reaching a leaf node.
2. EXPAND:   Call network to get (policy, value). Create children for all legal moves.
3. EVALUATE: Use network's value estimate (no rollouts — this is AlphaZero style).
4. BACKUP:   Propagate value up through ancestors, negating at each level.
```

### Functions Required

```python
def run_mcts(
    root_state: 'GameState',
    network: 'UltimateTTTNetwork',
    num_simulations: int = 800,
    c_puct: float = 1.0,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
    device: str = 'cpu'
) -> MCTSNode:
    """Run MCTS from the given state and return the root node with statistics.
    
    Dirichlet noise is added to the ROOT node's prior probabilities only.
    This encourages exploration during self-play. The formula is:
        P_noisy(a) = (1 - ε) · P(a) + ε · Dir(α)
        where Dir(α) is sampled from Dirichlet distribution
        α = 0.3 is appropriate for Ultimate TTT (similar to chess in AlphaZero)
        ε = 0.25 (KataGo default)
    
    Args:
        root_state: Starting game state
        network: Neural network for evaluation
        num_simulations: Number of MCTS simulations (default 800 per AlphaZero)
        c_puct: Exploration constant
        dirichlet_alpha: Dirichlet concentration parameter for root noise
        dirichlet_epsilon: Weight of Dirichlet noise (0 = no noise, 1 = all noise)
        device: Device for network inference
    
    Returns:
        Root MCTSNode with visit statistics populated after all simulations
    """

def select_move(
    root: MCTSNode,
    temperature: float = 1.0
) -> int:
    """Select a move from MCTS results using temperature-controlled sampling.
    
    Temperature controls exploration vs exploitation in move selection:
    
    temperature = 1.0: Sample proportional to visit counts (exploration)
        Used for first ~30 moves of self-play games
    
    temperature → 0.0: Select the move with highest visit count (exploitation)
        Used for evaluation games and later in self-play games
        Implemented as temperature = 0.0 → argmax of visit counts
    
    Formula for temperature > 0:
        π(a) = N(a)^(1/temperature) / Σ N(a')^(1/temperature)
        Then sample from distribution π
    
    Args:
        root: Root node after MCTS has been run
        temperature: Controls randomness. 0.0 = greedy, 1.0 = proportional.
    
    Returns:
        Selected move_idx (int, 0-80)
    """

def _select(node: MCTSNode, c_puct: float) -> MCTSNode:
    """Phase 1: Traverse tree following best PUCT scores until leaf."""

def _expand_and_evaluate(
    node: MCTSNode,
    network: 'UltimateTTTNetwork',
    device: str
) -> float:
    """Phase 2+3: Expand leaf node and return network value estimate.
    
    If node is terminal, return actual game outcome (+1, -1, or 0).
    Otherwise call network and expand with prior probabilities.
    
    Returns:
        Value from current player's perspective
    """

def _backup(node: MCTSNode, value: float) -> None:
    """Phase 4: Backpropagate value up the tree, negating at each level."""
```

---

## Episode 10 — `03_mcts/policy_target.py`

**YouTube Title:** "Policy Target Pruning — Why Raw MCTS Visits Are Noisy Training Targets"

**Concept taught:** The KataGo insight that MCTS visit distributions are optimized for
SEARCH (need exploration) but not for TRAINING (want clean signal). Policy target pruning
removes noise before the network learns from it.

### The Problem

During MCTS, we add Dirichlet noise and use temperature > 0. This means some moves
get visited just for exploration, not because they're actually good. If we train the
network to predict these noisy visit counts, we're training on noise.

### The Solution — Forced Playouts + Pruning

```python
def compute_policy_target(
    visit_counts: dict[int, int],
    num_legal_moves: int,
    pruning_threshold: float = 0.0
) -> torch.Tensor:
    """Convert MCTS visit counts to clean training target.
    
    Pruning strategy:
    1. Convert visit counts to probabilities: π(a) = N(a) / Σ N
    2. Prune moves with probability below threshold
    3. Renormalize remaining probabilities to sum to 1.0
    
    The threshold removes exploratory low-visit moves that provide
    noisy training signal. KataGo uses a softmax-like pruning rather
    than a hard threshold — implement both and choose via parameter.
    
    Args:
        visit_counts: Dict {move_idx: N(a)} from MCTS root node
        num_legal_moves: Number of legal moves (for context)
        pruning_threshold: Minimum probability to keep a move.
                          0.0 means no pruning (use all visits).
                          Recommended: 1 / (num_simulations) — prune single-visit moves.
    
    Returns:
        Tensor of shape (81,), dtype=float32
        Probability distribution over all 81 moves.
        Illegal/pruned moves have probability 0.0.
        All values sum to 1.0.
    """

def add_forced_playouts(
    visit_counts: dict[int, int],
    legal_moves: list[int],
    forced_playout_k: int = 2
) -> dict[int, int]:
    """Ensure every legal move gets a minimum number of visits.
    
    KataGo improvement: Before pruning, force at least k visits to every
    legal move. This prevents premature convergence where the network
    never explores moves it initially considered unlikely.
    
    Args:
        visit_counts: Existing visit counts from MCTS
        legal_moves: All legal moves that should be visited
        forced_playout_k: Minimum visits per legal move
    
    Returns:
        Updated visit counts with forced playouts added
    """
```

---

## Episode 11 — `04_training/self_play.py`

**YouTube Title:** "Self-Play — How the AI Generates Its Own Training Data"

**Concept taught:** The self-play loop. How MCTS + network generates game records.
What data we collect per move and why. Temperature scheduling.

### Data Collected Per Move

For each move in a self-play game, record:
```python
@dataclass
class MoveRecord:
    state_tensor: torch.Tensor     # encoded state at time of move, shape (7, 9, 9)
    policy_target: torch.Tensor    # pruned MCTS visit distribution, shape (81,)
    opp_policy_target: torch.Tensor # next player's MCTS target (from their turn), shape (81,)
    legal_mask: torch.Tensor       # legal moves at this state, shape (81,)
    current_player: int            # 1 or -1 (raw game player, before encoding flip)
```

For the whole game, record:
```python
@dataclass  
class GameRecord:
    moves: list[MoveRecord]
    winner: int                    # 1, -1, or 0 (draw)
    game_length: int
    
    # Derived targets (computed after game ends, added to each MoveRecord)
    # value_target[i]     = winner * current_player[i]  → +1 if winner, -1 if loser
    # score_target[i]     = final score margin from current_player[i]'s perspective
    # ownership_target[i] = which sub-boards each player won, from current_player[i]'s perspective
```

### Functions Required

```python
def play_self_play_game(
    network: 'UltimateTTTNetwork',
    num_simulations: int = 800,
    temperature_threshold: int = 30,
    device: str = 'cpu'
) -> GameRecord:
    """Play one complete self-play game and return the game record.
    
    Temperature schedule:
    - Moves 0 to temperature_threshold: temperature = 1.0 (exploration)
    - Moves after temperature_threshold: temperature = 0.0 (exploitation/greedy)
    
    After game ends, compute and attach value/score/ownership targets to
    each MoveRecord in the game.
    
    The opponent policy target for move i is the policy target from move i+1
    (the opponent's next move). For the final move, use a uniform distribution
    over legal moves as a placeholder.
    
    Args:
        network: Current best network (used for both players)
        num_simulations: MCTS simulations per move
        temperature_threshold: Move number after which temperature → 0
        device: Compute device
    
    Returns:
        Complete GameRecord with all targets computed
    """

def generate_self_play_batch(
    network: 'UltimateTTTNetwork',
    num_games: int,
    num_simulations: int = 800,
    device: str = 'cpu'
) -> list[GameRecord]:
    """Generate multiple self-play games.
    
    Args:
        network: Current network
        num_games: Number of games to play
        num_simulations: MCTS simulations per move
        device: Compute device
    
    Returns:
        List of GameRecord objects
    """
```

---

## Episode 12 — `04_training/replay_buffer.py`

**YouTube Title:** "The Replay Buffer — Stable Training with Experience Replay"

**Concept taught:** Why we can't just train on the most recent games (catastrophic
forgetting). How a circular buffer of past positions stabilizes training.
Data balancing to prevent opening positions from dominating.

### Design

The buffer stores individual `MoveRecord` objects (not full games) so we can
sample uniformly across positions. It's a circular buffer with a maximum capacity.

```python
class ReplayBuffer:
    """Circular buffer storing past self-play positions for training.
    
    Stores individual MoveRecord objects with their computed targets.
    When full, oldest positions are overwritten (circular behavior).
    
    Why experience replay?
    - Prevents overfitting to the current network's self-play distribution
    - Provides i.i.d. sampling assumption for gradient descent
    - Mixes recent positions (stronger play) with older ones (diverse openings)
    
    Args:
        capacity: Maximum number of positions to store.
                  Recommended: 500_000 (AlphaZero uses ~500K for small games)
    """
    
    def __init__(self, capacity: int = 500_000):
        ...
    
    def add_game(self, game_record: 'GameRecord') -> None:
        """Add all positions from a game record to the buffer.
        
        Flattens the GameRecord into individual training examples
        (one per move) and adds them to the circular buffer.
        """
    
    def sample(self, batch_size: int) -> 'TrainingBatch':
        """Sample a random batch of positions for training.
        
        Args:
            batch_size: Number of positions to sample
        
        Returns:
            TrainingBatch with all tensors stacked along batch dimension
        
        Raises:
            ValueError: If buffer has fewer positions than batch_size
        """
    
    def __len__(self) -> int:
        """Return current number of positions in buffer."""
    
    @property
    def is_ready(self) -> bool:
        """Return True if buffer has enough data to start training.
        Threshold: at least batch_size * 10 positions (default: 2560).
        """

@dataclass
class TrainingBatch:
    """A batch of training examples ready for loss computation."""
    state_tensors:      torch.Tensor  # (B, 7, 9, 9)
    policy_targets:     torch.Tensor  # (B, 81)
    opp_policy_targets: torch.Tensor  # (B, 81)
    value_targets:      torch.Tensor  # (B, 1)
    score_targets:      torch.Tensor  # (B, 1)
    ownership_targets:  torch.Tensor  # (B, 9)
    legal_masks:        torch.Tensor  # (B, 81)
```

---

## Episode 13 — `04_training/loss.py`

**YouTube Title:** "The Loss Function — Teaching the Network Five Things at Once"

**Concept taught:** Multi-task learning. Why each loss component matters.
How loss weights control what the network prioritizes learning.
How to interpret each loss component during training.

### The Complete Loss Function

```
L_total = L_policy 
        + λ_value     · L_value 
        + λ_score     · L_score 
        + λ_ownership · L_ownership 
        + λ_opp       · L_opp_policy

Default weights:
    λ_value     = 1.0
    λ_score     = 0.5
    λ_ownership = 0.5
    λ_opp       = 0.15   (KataGo's exact value for opponent policy)
```

### Functions Required

```python
def policy_loss(
    logits: torch.Tensor,      # (B, 81) raw logits
    targets: torch.Tensor,     # (B, 81) MCTS probability distribution
    legal_masks: torch.Tensor  # (B, 81) 1=legal
) -> torch.Tensor:
    """Cross-entropy loss between network policy and MCTS policy target.
    
    Formula: L = -Σ_a π_mcts(a) · log(softmax(logits)[a])
    
    Important: Apply legal mask before computing softmax.
    The MCTS target already has 0.0 for illegal moves, so they don't
    contribute to the loss numerically, but the logit masking ensures
    the softmax denominator is correct.
    
    Returns:
        Scalar mean loss over batch
    """

def value_loss(
    predictions: torch.Tensor,  # (B, 1) tanh output
    targets: torch.Tensor       # (B, 1) actual outcomes, values in {-1, 0, 1}
) -> torch.Tensor:
    """Mean squared error between predicted and actual game outcome.
    
    Formula: L = mean((z - v)²)
    
    Note: MSE is used (not cross-entropy) because value is continuous in [-1, 1]
    
    Returns:
        Scalar mean loss over batch
    """

def score_loss(
    predictions: torch.Tensor,  # (B, 1) tanh output
    targets: torch.Tensor       # (B, 1) normalized score margin in [-1, 1]
) -> torch.Tensor:
    """MSE between predicted score margin and actual score margin.
    
    Score margin = (sub_boards_won - sub_boards_lost) / 9
    Normalized to [-1, 1].
    
    Returns:
        Scalar mean loss over batch
    """

def ownership_loss(
    predictions: torch.Tensor,  # (B, 9) sigmoid output
    targets: torch.Tensor       # (B, 9) binary — did current player win each sub-board?
) -> torch.Tensor:
    """Binary cross-entropy between predicted and actual sub-board ownership.
    
    Formula: L = -mean(Σ_i [o_i·log(ô_i) + (1-o_i)·log(1-ô_i)])
    
    Returns:
        Scalar mean loss over batch
    """

def compute_total_loss(
    network_output: 'NetworkOutput',
    batch: 'TrainingBatch',
    lambda_value: float = 1.0,
    lambda_score: float = 0.5,
    lambda_ownership: float = 0.5,
    lambda_opp: float = 0.15,
) -> 'LossBreakdown':
    """Compute the complete weighted multi-task loss.
    
    Returns:
        LossBreakdown namedtuple — use total for .backward(), 
        use components for logging/visualization
    """

@dataclass
class LossBreakdown:
    total:          torch.Tensor  # scalar — call .backward() on this
    policy:         torch.Tensor  # scalar — for logging
    value:          torch.Tensor  # scalar — for logging
    score:          torch.Tensor  # scalar — for logging
    ownership:      torch.Tensor  # scalar — for logging
    opp_policy:     torch.Tensor  # scalar — for logging
```

---

## Episode 14 — `04_training/trainer.py`

**YouTube Title:** "The Training Loop — Putting It All Together"

**Concept taught:** The AlphaZero training cycle. How self-play and training
interleave. Learning rate scheduling. Gradient clipping. Checkpointing.

### The Training Cycle

```
For each iteration:
    1. SELF-PLAY:  Generate N games using current best network
    2. ADD TO BUFFER: Add all positions to replay buffer
    3. TRAIN:  Sample M batches from buffer, compute loss, update weights
    4. EVALUATE: Run arena every K iterations (see Episode 19)
    5. CHECKPOINT: Save network if it beats previous best
```

### Functions Required

```python
@dataclass
class TrainingConfig:
    # Self-play
    games_per_iteration:  int   = 100
    num_simulations:      int   = 800
    temperature_threshold: int  = 30
    
    # Training
    batch_size:           int   = 256
    batches_per_iteration: int  = 100
    learning_rate:        float = 0.001
    weight_decay:         float = 1e-4
    grad_clip_norm:       float = 1.0
    
    # Loss weights
    lambda_value:         float = 1.0
    lambda_score:         float = 0.5
    lambda_ownership:     float = 0.5
    lambda_opp:           float = 0.15
    
    # Evaluation
    arena_every_n:        int   = 10    # run arena every N iterations
    arena_games:          int   = 100   # games in each arena
    win_rate_threshold:   float = 0.55  # must win 55% to update best network
    
    # Buffer
    buffer_capacity:      int   = 500_000
    
    # Checkpointing
    checkpoint_dir:       str   = 'checkpoints/'
    device:               str   = 'cpu'

def train(config: TrainingConfig) -> None:
    """Main training loop. Runs indefinitely until interrupted.
    
    Logs all metrics to stdout and saves loss curves as JSON.
    Saves checkpoint after every iteration with improved network.
    
    Metric logging format (one line per iteration):
    {
        "iteration": 5,
        "games_played": 500,
        "loss_total": 2.341,
        "loss_policy": 1.823,
        "loss_value": 0.312,
        "loss_score": 0.124,
        "loss_ownership": 0.051,
        "loss_opp_policy": 0.031,
        "elo": 412.3,
        "avg_game_length": 38.2,
        "policy_entropy": 2.14,
        "arena_win_rate": 0.61   (only when arena was run)
    }
    """

def train_step(
    network: 'UltimateTTTNetwork',
    optimizer: torch.optim.Optimizer,
    batch: 'TrainingBatch',
    config: TrainingConfig,
    device: str
) -> 'LossBreakdown':
    """Single gradient update step.
    
    Steps:
    1. Move batch to device
    2. Forward pass
    3. Compute loss
    4. Backward pass
    5. Clip gradients (prevents exploding gradients)
    6. Optimizer step
    
    Returns:
        LossBreakdown with all components (detached, for logging)
    """

def save_checkpoint(
    network: 'UltimateTTTNetwork',
    optimizer: torch.optim.Optimizer,
    iteration: int,
    elo: float,
    checkpoint_dir: str
) -> str:
    """Save network and optimizer state.
    
    Filename format: checkpoint_iter{iteration:05d}_elo{elo:.0f}.pt
    
    Saves:
        - network.state_dict()
        - optimizer.state_dict()
        - iteration number
        - elo rating
        - timestamp
    
    Returns:
        Path to saved checkpoint file
    """

def load_checkpoint(
    path: str,
    network: 'UltimateTTTNetwork',
    optimizer: torch.optim.Optimizer | None = None
) -> dict:
    """Load network (and optionally optimizer) from checkpoint.
    
    Returns:
        Dict with 'iteration', 'elo', 'timestamp'
    """
```

---

## Episode 15 — `05_explainability/gradcam.py`

**YouTube Title:** "Grad-CAM — Which Cells Is the AI Looking At?"

**Concept taught:** How gradients flowing into the last convolutional layer reveal
what spatial regions the network focuses on for a given decision.
Why preserving 9×9 spatial dimensions throughout the network was architecturally critical.

### Implementation Notes

- Hook onto the LAST ResidualBlock in the trunk (before policy/value heads split)
- Compute gradient of the chosen action's logit w.r.t. that layer's feature maps
- Weight feature maps by their gradient importance and sum
- Apply ReLU (only positive contributions matter)
- Upsample to 9×9 if necessary (should already be 9×9 due to our architecture)

```python
def compute_gradcam(
    network: 'UltimateTTTNetwork',
    state: 'GameState',
    target_move: int | None = None,
    device: str = 'cpu'
) -> torch.Tensor:
    """Compute Grad-CAM heatmap for a specific move decision.
    
    Args:
        network: Trained network
        state: Game state to analyze
        target_move: Move index (0-80) to explain. If None, uses the
                     move with highest policy probability (best move).
        device: Compute device
    
    Returns:
        Heatmap tensor of shape (9, 9), values in [0, 1].
        Higher values = cells that most influenced the decision.
    
    Algorithm:
        1. Encode state to tensor, enable gradients
        2. Forward pass through network
        3. Select target logit (policy_logits[target_move])
        4. Backward pass to get gradients at last conv layer
        5. α_k = mean of gradients over spatial dims (9×9) → shape (C,)
        6. heatmap = ReLU(Σ_k α_k · feature_map_k) → shape (9, 9)
        7. Normalize heatmap to [0, 1]
    """

def visualize_gradcam(
    state: 'GameState',
    heatmap: torch.Tensor,
    target_move: int,
    save_path: str | None = None
) -> None:
    """Overlay Grad-CAM heatmap on board visualization.
    
    Shows:
    - Board state (pieces, sub-board boundaries, active sub-board)
    - Heatmap overlay (red = high importance, blue = low)
    - Highlighted target move
    - Title with move description
    """
```

---

## Episode 16 — `05_explainability/integrated_grads.py`

**YouTube Title:** "Integrated Gradients — Provably Faithful Cell Attribution"

**Concept taught:** Why Grad-CAM can be misleading (not guaranteed to sum to output).
The completeness axiom of Integrated Gradients. The baseline (empty board) concept.
Why this method gives a stronger theoretical guarantee than Grad-CAM.

```python
def compute_integrated_gradients(
    network: 'UltimateTTTNetwork',
    state: 'GameState',
    target_move: int | None = None,
    baseline_state: 'GameState | None' = None,
    num_steps: int = 50,
    device: str = 'cpu'
) -> torch.Tensor:
    """Compute Integrated Gradients attribution for a move decision.
    
    Integrated Gradients formula:
        IG(x) = (x - x') × ∫[0→1] ∂F(x' + α(x-x')) / ∂x dα
        
    Approximated with num_steps discrete steps along the interpolation path.
    
    The completeness axiom guarantees:
        Σ IG(x_i) = F(x) - F(x')
    Where F is the network output for target_move, x is the encoded state,
    and x' is the encoded baseline (empty board).
    
    Args:
        network: Trained network
        state: Game state to explain
        target_move: Move to explain. If None, uses highest probability move.
        baseline_state: Reference state. If None, uses empty board (all zeros tensor).
        num_steps: Number of interpolation steps (more = more accurate, slower)
        device: Compute device
    
    Returns:
        Attribution tensor of shape (7, 9, 9).
        Sum over channels gives per-cell importance: shape (9, 9).
        Positive values: cells that increase probability of target_move.
        Negative values: cells that decrease probability of target_move.
    
    Note:
        Call .abs().sum(dim=0) to get unsigned per-cell importance for visualization.
    """

def verify_completeness(
    attributions: torch.Tensor,
    network_output_diff: float,
    tolerance: float = 1e-3
) -> bool:
    """Verify the completeness axiom: sum of attributions ≈ network output difference.
    
    This is a correctness check for the implementation.
    If this fails, num_steps is too low or there's a bug.
    """
```

---

## Episode 17 — `05_explainability/mcts_viz.py`

**YouTube Title:** "Visualizing the AI's Thinking — MCTS Visit Counts as a Heatmap"

**Concept taught:** How MCTS visit counts naturally give interpretable explanations.
This is the most intuitive explainability method — no gradient math required.
Value delta per move as causal attribution.

```python
def compute_visit_heatmap(
    root: 'MCTSNode',
    normalize: bool = True
) -> torch.Tensor:
    """Convert MCTS visit counts to a 9×9 heatmap.
    
    Args:
        root: Root MCTS node after search has been run
        normalize: If True, normalize to [0, 1]. If False, return raw counts.
    
    Returns:
        Tensor of shape (9, 9). Zero for unvisited/illegal moves.
    """

def compute_value_delta_heatmap(
    state: 'GameState',
    network: 'UltimateTTTNetwork',
    device: str = 'cpu'
) -> torch.Tensor:
    """Compute value delta for each legal move.
    
    For each legal move m:
        Δvalue(m) = V(state_after_m) - V(state)
    
    Where V is the network's win_value estimate.
    
    This shows CAUSAL attribution — which moves the AI expects to
    most improve (or worsen) the position.
    
    Returns:
        Tensor of shape (9, 9).
        Positive: move improves position.
        Negative: move worsens position.
        Zero: illegal move.
    """

def render_three_panel_explanation(
    state: 'GameState',
    root: 'MCTSNode',
    network: 'UltimateTTTNetwork',
    chosen_move: int,
    device: str = 'cpu',
    save_path: str | None = None
) -> None:
    """Render a three-panel explanation figure using matplotlib.
    
    Panel 1: MCTS visit count heatmap — "What the AI considered"
    Panel 2: Grad-CAM heatmap — "What features drove the choice"
    Panel 3: Value delta heatmap — "What the AI expects to gain"
    
    All three panels show the board state with the chosen move highlighted.
    """
```

---

## Episode 18 — `06_evaluation/elo.py`

**YouTube Title:** "The Elo Rating System — Measuring AI Playing Strength"

**Concept taught:** How Elo works mathematically. Why it's the right metric for
comparing AI checkpoints. The K-factor and what it controls.

```python
def expected_score(rating_a: float, rating_b: float) -> float:
    """Compute expected score for player A against player B.
    
    Formula: E_A = 1 / (1 + 10^((R_B - R_A) / 400))
    
    Returns value in [0, 1] — probability player A wins.
    """

def update_elo(
    rating: float,
    opponent_rating: float,
    actual_score: float,
    k_factor: float = 32.0
) -> float:
    """Update Elo rating after a game result.
    
    Formula: R_new = R_old + K · (actual_score - expected_score)
    
    Args:
        rating: Current player's Elo rating
        opponent_rating: Opponent's Elo rating
        actual_score: 1.0 for win, 0.5 for draw, 0.0 for loss
        k_factor: Controls rating volatility (higher = faster changes)
    
    Returns:
        Updated Elo rating
    """

class EloTracker:
    """Tracks Elo ratings for multiple agents across training.
    
    Maintains a history of Elo ratings over time for plotting.
    
    Args:
        initial_rating: Starting Elo for new agents. Default: 1000.
        k_factor: Elo K-factor. Default: 32.
    """
    
    def __init__(self, initial_rating: float = 1000.0, k_factor: float = 32.0):
        ...
    
    def add_agent(self, name: str) -> None:
        """Register a new agent with the initial rating."""
    
    def record_result(self, winner: str, loser: str, draw: bool = False) -> None:
        """Record game result and update both ratings."""
    
    def get_rating(self, name: str) -> float:
        """Get current Elo rating for an agent."""
    
    def get_history(self, name: str) -> list[tuple[int, float]]:
        """Get (game_number, elo) history for plotting."""
```

---

## Episode 19 — `06_evaluation/arena.py`

**YouTube Title:** "The Arena — Pitting Networks Against Each Other to Measure Progress"

**Concept taught:** Why loss decrease doesn't always mean the model is actually better.
How to use head-to-head games as the true measure of improvement.
The 55% win rate threshold from AlphaZero.

```python
def run_arena(
    network_new: 'UltimateTTTNetwork',
    network_old: 'UltimateTTTNetwork',
    num_games: int = 100,
    num_simulations: int = 400,   # fewer than training for speed
    device: str = 'cpu'
) -> 'ArenaResult':
    """Pit two networks against each other over num_games games.
    
    Each network plays as both player 1 and player 2 (alternates).
    Temperature = 0 during arena (greedy play, no exploration noise).
    No Dirichlet noise at root (deterministic evaluation).
    
    Args:
        network_new: Candidate new network to evaluate
        network_old: Current best network (baseline)
        num_games: Total games to play (should be even for fair alternation)
        num_simulations: MCTS simulations per move during arena
        device: Compute device
    
    Returns:
        ArenaResult with win/loss/draw counts and win rate
    """

@dataclass
class ArenaResult:
    wins:     int     # new network wins
    losses:   int     # new network losses  
    draws:    int     # draws
    win_rate: float   # wins / (wins + losses + draws)
    
    @property
    def new_is_better(self) -> bool:
        """Return True if new network should replace old (win_rate > 0.55)."""
        return self.win_rate > 0.55

def play_single_game(
    player1_network: 'UltimateTTTNetwork',
    player2_network: 'UltimateTTTNetwork',
    num_simulations: int,
    device: str
) -> int:
    """Play one game between two networks.
    
    Returns:
        1 if player1 wins, -1 if player2 wins, 0 if draw
    """
```

---

## Episode 20 — `06_evaluation/metrics.py`

**YouTube Title:** "Reading the Training Metrics — How to Know Your AI Is Learning"

**Concept taught:** What each metric means and how to interpret it.
What healthy training curves look like vs warning signs.

### Functions Required

```python
def compute_policy_entropy(policy_probs: torch.Tensor) -> float:
    """Compute entropy of policy distribution.
    
    H = -Σ p(a) · log(p(a))
    
    High entropy (early training): network is uncertain, explores broadly
    Low entropy (late training): network is decisive, plays with conviction
    
    Expected range:
        Random policy over 81 moves: H = log(81) ≈ 4.4
        Well-trained policy: H ≈ 1.0–2.0
        Degenerate policy (all mass on one move): H ≈ 0
    
    Args:
        policy_probs: shape (81,) or (B, 81), probability distribution
    
    Returns:
        Scalar entropy value (or mean entropy over batch)
    """

def compute_value_accuracy(
    value_preds: torch.Tensor,
    value_targets: torch.Tensor
) -> float:
    """Fraction of positions where sign(prediction) == sign(target).
    
    Measures whether the network correctly predicts who is winning,
    even if the exact probability is off.
    
    Args:
        value_preds:  (B, 1) predicted win values in [-1, 1]
        value_targets: (B, 1) actual outcomes in {-1, 0, 1}
    
    Returns:
        Accuracy in [0, 1]. Random baseline is ~0.5.
        Good network: > 0.7 after sufficient training.
    """

class MetricsLogger:
    """Logs and persists training metrics to JSON for later analysis.
    
    Maintains running averages of loss components within an iteration.
    Saves full history to JSON file for plotting.
    """
    
    def __init__(self, log_path: str = 'training_metrics.json'):
        ...
    
    def log_iteration(self, iteration: int, metrics: dict) -> None:
        """Log all metrics for one training iteration. Appends to JSON."""
    
    def plot_training_curves(self, save_path: str = 'training_curves.png') -> None:
        """Generate a 6-panel matplotlib figure:
        
        Panel 1: All 5 loss components over iterations (log scale y-axis)
        Panel 2: Elo rating over iterations
        Panel 3: Average game length over iterations
        Panel 4: Policy entropy over iterations
        Panel 5: Value head accuracy over iterations
        Panel 6: Arena win rate over iterations (scatter, only when available)
        """
    
    def get_summary(self) -> dict:
        """Return summary statistics of the most recent 10 iterations."""
```

---

## Training Diagnostics — What Healthy Training Looks Like

This section is reference material for the YouTube series. Include as comments
in `trainer.py` and expand in `metrics.py`.

### Loss Curves — Expected Behavior

| Loss | At iteration 0 | Healthy progress | Red flags |
|---|---|---|---|
| L_policy | ~4.4 (log 81) | Steady decrease | Plateau before iter 50 |
| L_value | ~0.5–1.0 | Slow, noisy decrease | Wild oscillation |
| L_score | ~0.3–0.5 | Correlated with L_value | Moving opposite to L_value |
| L_ownership | ~0.6–0.7 | Fast early drop | Not moving at all |
| L_opp_policy | ~4.4 | Tracks L_policy, slightly behind | Diverging from L_policy |

### Elo Curve — Expected Behavior

- First 10 iterations: Fast rise from 1000 to ~1200 (network learns basic tactics)
- Iterations 10–50: Steady rise, ~20–30 Elo per iteration
- After 50+: Slower, diminishing returns
- Flat Elo for 10+ iterations = training is stuck (try adjusting lr or c_puct)

### Arena Win Rate — Expected Behavior

- Should be ~0.5 at random (against older checkpoint from last iteration)
- If consistently > 0.7: learning is fast, consider reducing games_per_iteration
- If consistently < 0.55: network is not improving, check loss curves for issues

### Warning Signs

1. **L_policy increases after initial decrease:** Network is forgetting — reduce learning rate
2. **L_value stuck at ~1.0:** Value head isn't learning — check that targets are correct sign
3. **Policy entropy hits 0:** Network has collapsed to one move — something is very wrong
4. **Arena win rate always < 0.5:** New network is consistently worse — check replay buffer

---

## Testing Requirements

Every episode file must pass its standalone `__main__` block as a smoke test.
Additionally, create `tests/` with pytest files:

```
tests/
├── test_board.py       # Test encode_state, decode_move, encode_move
├── test_rules.py       # Test all rule edge cases
├── test_network.py     # Test forward pass shapes
├── test_mcts.py        # Test node expansion, UCT, backup
├── test_loss.py        # Test loss values at known inputs
└── test_arena.py       # Test arena terminates and returns valid result
```

Key test cases for `test_rules.py`:
- Winning on meta-board is detected correctly
- Sent-to-won-sub-board correctly gives free choice (active = -1)
- Draw on meta-board (all sub-boards decided, no winner) detected
- Illegal move raises ValueError
- apply_move never mutates input state (copy semantics)

Key test cases for `test_loss.py`:
- policy_loss with perfect prediction returns ~0
- policy_loss with uniform prediction returns log(num_legal_moves)
- value_loss with perfect prediction returns 0
- compute_total_loss returns LossBreakdown where all components are non-negative scalars

---

## Code Style Requirements

1. **Type hints everywhere.** All function signatures must have complete type hints.

2. **Docstrings in Google style.** Args, Returns, Raises sections. Include shape annotations.

3. **No magic numbers.** All constants must be named and documented.

4. **Shape comments on every tensor operation.** 
   ```python
   x = self.conv(x)  # (B, 128, 9, 9)
   x = x.mean(dim=[2, 3])  # (B, 128)
   ```

5. **Immutable game states.** `apply_move` must never mutate its input. Use copy.

6. **Device-agnostic.** All network code must work on both CPU and CUDA.
   Never hardcode device. Always accept `device` as a parameter.

7. **Reproducibility.** Trainer must accept and set a random seed.

---

## Final Notes for Claude Code

- Build all files before running any tests
- Run each file's `__main__` block after implementing it to verify correctness
- The episode numbering in folder names is intentional — preserve it
- When in doubt about a design decision, choose the more explicit/verbose option
  over the clever/concise one. This is pedagogy code, not production code.
- Every `TODO` or open design decision should be left as a `# NOTE:` comment
  explaining the tradeoff, since those become teaching moments in the videos
