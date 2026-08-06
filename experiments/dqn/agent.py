"""DQN Agent for Ultimate Tic-Tac-Toe.

Contains:
  ReplayBuffer  — circular buffer of transitions
  DQNAgent      — epsilon-greedy selection, experience storage, training step
"""
from __future__ import annotations

import os
import random
import sys
from collections import deque

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module

_board = import_module("01_game.board")
_rules = import_module("01_game.rules")
encode_state        = _board.encode_state
get_legal_move_mask = _rules.get_legal_move_mask
get_legal_moves     = _rules.get_legal_moves

from model import QNetwork  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Replay Buffer
# ──────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Fixed-size circular buffer storing (s, a, r, s', done, legal_mask') tuples.

    Tensors are stored on CPU and moved to the training device only at sample time.
    """

    def __init__(self, capacity: int = 100_000) -> None:
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state_t:    torch.Tensor,   # (7, 9, 9)
        move:       int,
        reward:     float,
        next_t:     torch.Tensor,   # (7, 9, 9)
        done:       bool,
        next_mask:  torch.Tensor,   # (81,) binary float
    ) -> None:
        self.buffer.append((
            state_t,
            move,
            reward,
            next_t,
            float(done),
            next_mask,
        ))

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.buffer, batch_size)
        states, moves, rewards, nexts, dones, masks = zip(*batch)
        return (
            torch.stack(states).to(device),
            torch.tensor(moves,   dtype=torch.long,    device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device),
            torch.stack(nexts).to(device),
            torch.tensor(dones,   dtype=torch.float32, device=device),
            torch.stack(masks).to(device),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ──────────────────────────────────────────────────────────────────────────────
# DQN Agent
# ──────────────────────────────────────────────────────────────────────────────

class DQNAgent:
    """Deep Q-Network agent using experience replay and a target network.

    Args:
        lr:                  Adam learning rate.
        gamma:               Discount factor.
        epsilon:             Initial exploration rate (decayed externally).
        batch_size:          Minibatch size for each gradient step.
        target_update_freq:  Every N gradient steps, copy online→target weights.
        buffer_capacity:     Maximum replay buffer size.
        device:              'cpu', 'cuda', or 'mps'.
    """

    def __init__(
        self,
        lr: float   = 1e-4,
        gamma: float = 0.99,
        epsilon: float = 0.5,
        batch_size: int = 32,
        target_update_freq: int = 500,
        buffer_capacity: int = 100_000,
        device: str = "cpu",
    ) -> None:
        self.gamma      = gamma
        self.epsilon    = epsilon
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device     = torch.device(device)
        self._grad_steps = 0
        self._environment_steps = 0

        self.online_net = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(capacity=buffer_capacity)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, state) -> int:
        """Epsilon-greedy: random with prob epsilon, else argmax Q."""
        legal = get_legal_moves(state)
        if not legal:
            raise ValueError("No legal moves.")
        if self.epsilon > 0.0 and random.random() < self.epsilon:
            return random.choice(legal)

        s    = encode_state(state).unsqueeze(0).to(self.device)   # (1,7,9,9)
        mask = get_legal_move_mask(state).unsqueeze(0).to(self.device)  # (1,81)
        q    = self.online_net.q_masked(s, mask).squeeze(0)       # (81,)
        return int(q.argmax().item())

    # ------------------------------------------------------------------
    # Experience storage
    # ------------------------------------------------------------------

    def store(self, state, move: int, reward: float, next_state, done: bool) -> None:
        """Encode and push one transition to the replay buffer."""
        s    = encode_state(state).cpu()
        ns   = encode_state(next_state).cpu()
        mask = get_legal_move_mask(next_state).cpu()
        self.buffer.push(s, move, reward, ns, done, mask)

    def observe(
        self,
        state,
        move: int,
        reward: float,
        next_state,
        done: bool,
        *,
        train_every: int = 4,
        learning_starts: int = 1_000,
    ) -> float | None:
        """Store one transition and train at the configured environment cadence."""
        self.store(state, move, reward, next_state, done)
        self._environment_steps += 1
        if (
            self._environment_steps >= learning_starts
            and self._environment_steps % train_every == 0
        ):
            return self.train_step()
        return None

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self) -> float | None:
        """Sample a minibatch and do one gradient step. Returns loss or None."""
        if len(self.buffer) < self.batch_size:
            return None

        states, moves, rewards, nexts, dones, masks = self.buffer.sample(
            self.batch_size, self.device
        )

        # Q(s, a) for the actions actually taken
        q_current = self.online_net(states).gather(1, moves.unsqueeze(1)).squeeze(1)

        # Target: r + γ · max_a' Q_target(s', a')  (0 if terminal).
        # Training transitions span from one agent decision to its next decision,
        # after the random opponent has moved, so both states use the agent's
        # perspective and the standard positive bootstrap sign applies.
        with torch.no_grad():
            q_next = self.target_net.q_masked(nexts, masks)          # (B, 81)
            q_next_max = q_next.max(dim=1).values                     # (B,)
            # Terminal masks contain no legal actions, so their maximum is -inf.
            # Replace it explicitly instead of multiplying by zero (which would
            # produce NaN), while preserving legitimate negative nonterminal Qs.
            q_next_max = torch.where(
                dones.bool(), torch.zeros_like(q_next_max), q_next_max
            )
            targets = rewards + self.gamma * q_next_max

        loss = F.mse_loss(q_current, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

        self._grad_steps += 1
        if self._grad_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save({
            "online_state_dict": self.online_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "grad_steps": self._grad_steps,
            "environment_steps": self._environment_steps,
            "epsilon": self.epsilon,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_state_dict"])
        self.target_net.load_state_dict(ckpt["target_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self._grad_steps = ckpt["grad_steps"]
        self._environment_steps = ckpt.get("environment_steps", 0)
        self.epsilon     = ckpt["epsilon"]
