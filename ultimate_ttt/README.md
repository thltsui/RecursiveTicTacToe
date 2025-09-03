# Ultimate Tic Tac Toe

An advanced variant of classic Tic Tac Toe where each cell contains another 3x3 Tic Tac Toe board, creating a total of 81 playable cells.

## How the Game Works

Ultimate Tic Tac Toe consists of a 3x3 grid where each cell contains another 3x3 Tic Tac Toe board. The goal is to win three small boards in a row on the big 3x3 board by playing standard Tic Tac Toe inside each small board.

### Game Rules

1. **First Move**: Can be made in any cell of any small board
2. **Navigation Rule**: After the first move, the opponent must play in the small board corresponding to the index of the last move's small cell
3. **Board Constraints**: If the target small board is full or already won, the opponent may play in any uncompleted board
4. **Winning Small Boards**: A player wins a small board by making three of their marks in a row, column, or diagonal
5. **Winning the Game**: Win three small boards in a row on the big board using the same win conditions

### Board Layout

```
Big Board Indices (0-8):
[0][1][2]
[3][4][5]
[6][7][8]

Small Cell Indices within each board (0-8):
0 | 1 | 2
---------
3 | 4 | 5
---------
6 | 7 | 8
```

## Installation

### Option 1: Using Poetry (Recommended)

1. **Install Poetry** (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ultimate_ttt
   ```

3. **Install dependencies**:
   ```bash
   poetry install
   ```

4. **Play the game**:
   ```bash
   poetry run python main.py
   ```

### Option 2: Using pip

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ultimate_ttt
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Play the game**:
   ```bash
   python main.py
   ```

### Option 3: Manual Installation

1. **Install Python dependencies manually**:
   ```bash
   pip install numpy torch
   ```

2. **Run the game**:
   ```bash
   python main.py
   ```

## Usage

### Interactive Game

Start the game with an interactive menu:
```bash
# Using Poetry
poetry run python main.py

# Using pip
python main.py
```

Or directly play a specific mode:
```bash
# Human vs Human
poetry run python main.py play --mode human-human
# or
python main.py play --mode human-human

# Human vs Random AI
poetry run python main.py play --mode human-random
# or
python main.py play --mode human-random

# Random vs Human
poetry run python main.py play --mode random-human
# or
python main.py play --mode random-human

# Random vs Random
poetry run python main.py play --mode random-random
# or
python main.py play --mode random-random
```

### Simulation

Simulate multiple random games:
```bash
# Simulate 1000 games
python main.py simulate --num-games 1000 --verbose --save

# Or use the interactive menu
python main.py
# Then choose option 2
```

### Training Data Generation

Generate training data for machine learning:
```bash
# Generate 5000 games as training data
python main.py generate --num-games 5000 --format both

# Or use the interactive menu
python main.py
# Then choose option 3
```

### Running Tests

```bash
python main.py test
```

### Demo

Watch a quick demo game:
```bash
python main.py demo
```

## Repository Structure

```
ultimate_ttt/
├── game/                    # Core game logic
│   ├── board.py            # Board state, game rules, move validation
│   ├── utils.py            # Helper functions, win checking
│   └── config.py           # Constants and configuration
├── play/                    # Game interfaces
│   ├── play_console.py     # Interactive console gameplay
│   └── play_random.py      # Random agent simulation
├── data/                    # Data storage
│   └── saved_games/        # Saved game histories
├── tests/                   # Unit tests
│   └── test_board.py       # Board logic tests
├── scripts/                 # Utility scripts
│   └── simulate_games.py   # Training data generation
├── main.py                  # CLI entry point
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Game Mechanics

### State Representation

The board state is stored as a 9x9x3 tensor:
- First two dimensions (9x9) represent the 81 small cells
- Third dimension is one-hot encoding: `[1,0,0]` = empty, `[0,1,0]` = Player 1, `[0,0,1]` = Player 2

### Move Validation

- Moves must follow the navigation rule
- Cannot play in won or full small boards
- Game ends when big board is won or all 81 cells are filled

### Win Detection

- Small boards: Standard Tic Tac Toe win patterns
- Big board: Three small boards in a row, column, or diagonal

## Data Export

Games can be exported in multiple formats:

- **JSON**: Human-readable format with game metadata
- **NPY**: NumPy arrays for machine learning pipelines
- **Summary**: Text files with game statistics

## Machine Learning Integration

The repository is designed to be compatible with reinforcement learning pipelines:

- State tensors in standard format
- Move history tracking
- Win/loss outcomes
- Configurable data generation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Testing

Run the full test suite:
```bash
python main.py test
```

Or run individual test files:
```bash
python -m unittest tests.test_board
```

## Examples

### Basic Game Play

```python
from game.board import UltimateTTTBoard

# Create a new game
board = UltimateTTTBoard()

# Make moves
board.make_move(4, 6)  # Play in board 4, cell 6
board.make_move(6, 2)  # Must play in board 6, cell 2

# Check game state
print(f"Game over: {board.game_over}")
print(f"Winner: {board.winner}")
```

### Simulation

```python
from play.play_random import simulate_multiple_games

# Simulate 100 games
stats = simulate_multiple_games(100, verbose=True)
print(f"Player 1 wins: {stats['wins_player1']}")
print(f"Player 2 wins: {stats['wins_player2']}")
print(f"Draws: {stats['draws']}")
```

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Inspired by the Ultimate Tic Tac Toe variant
- Designed for educational and research purposes
- Compatible with reinforcement learning frameworks
