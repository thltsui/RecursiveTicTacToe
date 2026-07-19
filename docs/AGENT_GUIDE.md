# Agent Guide: Ultimate Tic-Tac-Toe AI Project

Welcome, fellow AI agent! This guide serves as your entry point to understanding the architecture, workflows, and deployment mechanisms of the Ultimate Tic-Tac-Toe AI project.

## 1. Core Mechanism and AI Training
This project implements a self-learning AI for Ultimate Tic-Tac-Toe using an **AlphaZero-style architecture**, heavily inspired by DeepMind's AlphaGo/AlphaZero and incorporating domain-independent improvements from KataGo.

### Game Mechanics (`01_game/`)
- Ultimate Tic-Tac-Toe is played on a 9x9 grid, divided into nine 3x3 sub-boards.
- The previous move dictates which sub-board the next player must play in.
- The rules are encoded purely in `01_game/rules.py` and `01_game/board.py`. The neural network and MCTS are completely general and rules-agnostic.

### Neural Network (`02_network/`)
- **Architecture**: A PyTorch-based Convolutional/ResNet architecture that processes the 9x9 grid state and outputs:
  - **Policy**: A probability distribution over the 81 possible moves.
  - **Value**: An evaluation of the board state from the current player's perspective (win/loss prediction).

### Training Loop (`03_mcts/`, `04_training/`, `train.py`)
- **Self-Play**: The AI plays thousands of games against itself using Monte Carlo Tree Search (MCTS) guided by the neural network's policy and value estimates.
- **MCTS**: Evaluates moves by simulating future trajectories. It balances exploration (trying new moves) and exploitation (picking known good moves). 
- **Training**: The outcomes of these self-play games (who won, and what the MCTS search probabilities were) are saved into a replay buffer. The neural network is then trained via supervised learning on this replay buffer to minimize the value loss and policy cross-entropy.

---

## 2. Web App and Deployment (`web_app/`)
The web application provides a visual interface for humans to play against the trained AI or against other humans online.

### Tech Stack
- **Backend**: Python, Flask, Flask-SocketIO (for real-time multiplayer).
- **Frontend**: Vanilla HTML/JS/CSS (`index.html`, `main.js`, `style.css`).
- **State Management**: Upstash Redis is used as a pub/sub message queue and state store to keep game rooms synchronized across multiple server instances.

### Multiplayer Architecture
- Uses `eventlet` with `eventlet.monkey_patch()` for asynchronous socket handling without blocking the main event loop.
- The frontend forces a direct WebSocket connection (`transports: ['websocket']`) to bypass load balancer long-polling handshake scattering.

### Deployment (Fly.io)
- **Containerized**: Built via a Dockerfile running Python 3.11-slim. Dependency management is handled blazingly fast using `uv` (`uv sync --frozen --extra web`).
- **Checkpoints**: The Docker context excludes the massive `.git` history and the bulky intermediate `.pt` checkpoints. Only the `best_model.pt` from the `checkpoints/` directory is whitelisted in `.dockerignore` to keep the Docker image slim (~194MB context).
- **Hosting**: Deployed on Fly.io with a minimum of 2 instances for high availability. 

---

## 3. Git Branches Overview

The repository's evolution and various workstreams are separated into distinct branches (both locally and on `origin`). Before making structural changes, ensure you are on the correct branch for the task at hand.

- `main`
  The primary backbone of the project containing the foundational game logic, PyTorch network, and MCTS training loop.

- `webapp` **(Current Active Branch for Deployment)**
  Contains all web application code, Flask-SocketIO multiplayer integration, and Docker/Fly.toml deployment configuration. This branch hosts the production code that players interact with.

- `Substack`
  Dedicated solely to educational content. This branch stores the Substack articles under the `substack/` directory (note: historically referred to as `tthl`). These articles explain the project's progression, math, and AI concepts for a broader audience.

- `refactor/alphazero-ai-episodes`
  An experimental/refactoring branch aimed at restructuring the AI training pipeline into episodic scopes intended for a YouTube educational lecture series.

- `flask-app`
  An older or alternative iteration of the web server implementation. Usually deprecated in favor of `webapp`.
