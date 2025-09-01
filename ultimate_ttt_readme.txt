
Ultimate Tic Tac Toe Simulator

How the Game Works

Ultimate Tic Tac Toe is an advanced variant of classic Tic Tac Toe. It consists of a 3x3 grid where each cell contains another 3x3 Tic Tac Toe board, making a total of 81 cells.

Objective

The goal is to win three small boards in a row on the big 3x3 board by playing standard Tic Tac Toe inside each small board. A player wins a small board by making three of their marks in a row, column, or diagonal within that board.

Game Structure

- The overall game board contains 9 small boards.
- Each small board has 9 cells.
- The total number of playable cells is 81.

Each small board is indexed from 0 to 8 and laid out as:

0 | 1 | 2
---------
3 | 4 | 5
---------
6 | 7 | 8

The 9 small boards are arranged in a big board, also indexed 0 to 8:

[0][1][2]
[3][4][5]
[6][7][8]

Turn Rules

1. The first move can be made in any cell of any small board.
2. After that, the opponent must play in the small board corresponding to the index of the last move's small cell.
3. If the target small board is full or already won, the opponent may play in any uncompleted board.

Example:
If Player 1 plays in small board 4, cell 6, Player 2 must play in small board 6.

Winning a Small Board

A player wins a small board if they occupy all cells in any of the following combinations:

- Rows: [0, 1, 2], [3, 4, 5], [6, 7, 8]
- Columns: [0, 3, 6], [1, 4, 7], [2, 5, 8]
- Diagonals: [0, 4, 8], [2, 4, 6]

Once won, the board is marked as complete and cannot be played in again.

Winning the Game

A player wins the game by winning three small boards in a row on the big board using the same set of win conditions (rows, columns, diagonals). If all 81 cells are filled without a 3-in-a-row win on the big board, the game ends in a draw.

State Representation

The board state is stored as a 9x9x3 tensor:
- The first two dimensions (9x9) represent the 81 small cells.
- The third dimension is a one-hot encoding of the cell state:
  - [1, 0, 0] = empty
  - [0, 1, 0] = Player 1
  - [0, 0, 1] = Player 2

Each game is represented as a sequence of such states over time:
- Shape: (9, 9, 3, T), where T <= 81 (number of moves)

This structure is designed to be compatible with reinforcement learning pipelines.

Game Constraints

- Players alternate turns.
- Moves must follow the navigation rule (current cell index determines opponent's board).
- If the target board is full or won, the next player may move anywhere.
- A small board cannot be played in once it is won or full.
- The game ends when a player wins the big board or all 81 cells are filled.

Intended Repository Structure

ultimate_ttt/
├── game/
│   ├── board.py          # Core logic: board state, game rules, move validation
│   ├── utils.py          # Helper functions: win checking, board flattening
│   └── config.py         # Index maps and constants
│
├── play/
│   ├── play_console.py   # Interactive game with human or random player
│   └── play_random.py    # Simulation using random agents
│
├── data/
│   └── saved_games/      # Saved game history in .npy, .json, or .pt format
│
├── tests/
│   └── test_board.py     # Unit tests for move validation and win detection
│
├── scripts/
│   └── simulate_games.py # Generate multiple game histories for training
│
├── main.py               # CLI entry point for running games or simulations
├── README.md             # This file
└── requirements.txt      # Dependencies (numpy, optionally torch)

Setup Instructions

1. Clone the repository
2. Install dependencies:
pip install -r requirements.txt

3. To play a game or run simulations:
python main.py

Output Format

Each played game can be exported as:
- A tensor: shape (9, 9, 3, T), saved as .npy or .pt
- A move log: list of (big_index, small_index, player) tuples
- Metadata: winner, total moves, draw flag

These can be used for supervised learning, reinforcement learning, or evaluation purposes.

Next Steps

- Add baseline RL agent
- Create Gym or PettingZoo wrapper
- Export board images for visualisation or debugging
