# Quick Start Guide

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Installation

### Option 1: Using Poetry (Recommended)
```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Play the game
poetry run python main.py
```

### Option 2: Install from requirements.txt
```bash
pip install -r requirements.txt
```

### Option 3: Install using setup.py
```bash
pip install -e .
```

### Option 4: Install dependencies manually
```bash
pip install numpy torch
```

### Option 5: Use the installation script
```bash
# On macOS/Linux
./install_poetry.sh

# On Windows
install_poetry.bat
```

## Quick Test

Run the basic functionality test to ensure everything is working:
```bash
# Using Poetry
poetry run python basic_test.py

# Using pip
python3 basic_test.py
```

## Play a Game

### Interactive Menu
```bash
# Using Poetry
poetry run python main.py

# Using pip
python3 main.py
```
Then choose option 1 to play a game.

### Direct Command
```bash
# Human vs Human
poetry run python main.py play --mode human-human
# or
python3 main.py play --mode human-human

# Human vs Random AI
poetry run python main.py play --mode human-random
# or
python3 main.py play --mode human-random

# Random vs Human
poetry run python main.py play --mode random-human
# or
python3 main.py play --mode random-human

# Random vs Random
poetry run python main.py play --mode random-random
# or
python3 main.py play --mode random-random
```

## Watch a Demo

See a quick game between two random players:
```bash
# Using Poetry
poetry run python demo.py

# Using pip
python3 demo.py
```

## Run Tests

```bash
# Using Poetry
poetry run python main.py test

# Using pip
python3 main.py test
```

## Game Controls

- **Moves**: Enter as `big_index small_index` (e.g., `4 6`)
- **Quit**: Type `q` during a move
- **Board indices**: 0-8 for both big and small boards
- **Navigation**: Follow the constraint shown after each move

## Example Game Session

```
Ultimate Tic Tac Toe
==============================
Choose an option:
1. Play a game
2. Simulate random games
3. Generate training data
4. Run tests
5. Show demo
6. Exit

Enter choice (1-6): 1

Choose game mode:
1. Human vs Human
2. Human vs Random
3. Random vs Human
4. Random vs Random

Enter choice (1-4): 1

Player X's turn
Enter move as 'big_index small_index' (e.g., '4 6')
Or enter 'q' to quit
Move: 4 6
```

## Troubleshooting

### Import Errors
If you get import errors, make sure you're in the correct directory:
```bash
cd ultimate_ttt
python3 main.py
```

### Missing Dependencies
Install required packages:
```bash
pip install numpy torch
```

### Permission Issues
On some systems, you might need to use `pip3` instead of `pip`:
```bash
pip3 install -r requirements.txt
```

## Next Steps

- Try different game modes
- Run simulations to see game statistics
- Generate training data for machine learning
- Explore the code structure in the `game/` directory
- Check out the comprehensive README.md for more details
