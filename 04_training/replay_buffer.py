"""Episode 12 — The Replay Buffer — Stable Training with Experience Replay

Concept taught: Why we can't just train on the most recent games (catastrophic
forgetting). How a circular buffer of past positions stabilizes training.
Data balancing to prevent opening positions from dominating.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from .self_play import MoveRecord, GameRecord


@dataclass
class TrainingBatch:
    """A batch of training examples ready for loss computation.

    All tensors are stacked along the batch dimension (B).
    """
    state_tensors:      torch.Tensor  # (B, 7, 9, 9)
    policy_targets:     torch.Tensor  # (B, 81)
    opp_policy_targets: torch.Tensor  # (B, 81)
    value_targets:      torch.Tensor  # (B, 1)
    score_targets:      torch.Tensor  # (B, 1)
    ownership_targets:  torch.Tensor  # (B, 9)
    legal_masks:        torch.Tensor  # (B, 81)
    opp_legal_masks:    torch.Tensor  # (B, 81)


class ReplayBuffer:
    """Circular buffer storing past self-play positions for training.

    Stores individual MoveRecord objects with their computed targets.
    When full, oldest positions are overwritten (circular behavior).

    Why experience replay?
        - Prevents overfitting to the current network's self-play distribution.
        - Provides i.i.d. sampling assumption for gradient descent.
        - Mixes recent positions (stronger play) with older ones (diverse openings).

    Args:
        capacity: Maximum number of positions to store.
            Recommended: 500_000 (AlphaZero uses ~500K for small games).
    """

    def __init__(self, capacity: int = 500_000):
        self.capacity = capacity
        self._buffer: list[MoveRecord] = []
        self._position: int = 0  # write position for circular behavior

    def add_game(self, game_record: GameRecord) -> None:
        """Add all positions from a game record to the buffer.

        Flattens the GameRecord into individual training examples
        (one per move) and adds them to the circular buffer.

        Args:
            game_record: Complete game record with targets computed.
        """
        for move_record in game_record.moves:
            if len(self._buffer) < self.capacity:
                self._buffer.append(move_record)
            else:
                self._buffer[self._position] = move_record
            self._position = (self._position + 1) % self.capacity

    def sample(self, batch_size: int) -> TrainingBatch:
        """Sample a random batch of positions for training.

        Args:
            batch_size: Number of positions to sample.

        Returns:
            TrainingBatch with all tensors stacked along batch dimension.

        Raises:
            ValueError: If buffer has fewer positions than batch_size.
        """
        if len(self._buffer) < batch_size:
            raise ValueError(
                f"Buffer has {len(self._buffer)} positions, need {batch_size}"
            )

        samples = random.sample(self._buffer, batch_size)

        return TrainingBatch(
            state_tensors=torch.stack([s.state_tensor for s in samples]),       # (B, 7, 9, 9)
            policy_targets=torch.stack([s.policy_target for s in samples]),     # (B, 81)
            opp_policy_targets=torch.stack([s.opp_policy_target for s in samples]),  # (B, 81)
            value_targets=torch.tensor(
                [[s.value_target] for s in samples], dtype=torch.float32       # (B, 1)
            ),
            score_targets=torch.tensor(
                [[s.score_target] for s in samples], dtype=torch.float32       # (B, 1)
            ),
            ownership_targets=torch.stack([s.ownership_target for s in samples]),  # (B, 9)
            legal_masks=torch.stack([s.legal_mask for s in samples]),           # (B, 81)
            opp_legal_masks=torch.stack([s.opp_legal_mask for s in samples]),   # (B, 81)
        )

    def __len__(self) -> int:
        """Return current number of positions in buffer."""
        return len(self._buffer)

    @property
    def is_ready(self, min_positions: int = 2560) -> bool:
        """Return True if buffer has enough data to start training.

        Threshold: at least batch_size * 10 positions (default: 2560).
        """
        return len(self._buffer) >= min_positions

    def save_to_file(self, path: str) -> None:
        """Save buffer contents to disk for persistence across restarts.

        Serializes all MoveRecord tensors and scalars into a single file.

        Args:
            path: File path to save the buffer to.
        """
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        data = {
            'capacity': self.capacity,
            'position': self._position,
            'records': [{
                'state_tensor': rec.state_tensor,
                'policy_target': rec.policy_target,
                'opp_policy_target': rec.opp_policy_target,
                'opp_legal_mask': rec.opp_legal_mask,
                'legal_mask': rec.legal_mask,
                'current_player': rec.current_player,
                'value_target': rec.value_target,
                'score_target': rec.score_target,
                'ownership_target': rec.ownership_target,
            } for rec in self._buffer],
        }
        torch.save(data, path)

    @classmethod
    def load_from_file(cls, path: str) -> 'ReplayBuffer':
        """Load buffer contents from disk.

        Args:
            path: File path to load the buffer from.

        Returns:
            ReplayBuffer with restored contents.
        """
        data = torch.load(path, weights_only=False)
        buf = cls(capacity=data['capacity'])
        buf._position = data['position']
        buf._buffer = [
            MoveRecord(
                state_tensor=rec['state_tensor'],
                policy_target=rec['policy_target'],
                opp_policy_target=rec['opp_policy_target'],
                opp_legal_mask=rec['opp_legal_mask'],
                legal_mask=rec['legal_mask'],
                current_player=rec['current_player'],
                value_target=rec['value_target'],
                score_target=rec['score_target'],
                ownership_target=rec['ownership_target'],
            )
            for rec in data['records']
        ]
        return buf


if __name__ == "__main__":
    print("=== Episode 12: Replay Buffer ===\n")

    buf = ReplayBuffer(capacity=1000)
    print(f"Buffer created with capacity 1000")
    assert len(buf) == 0

    # Create fake game records
    for game_i in range(5):
        moves = []
        for m in range(10):
            rec = MoveRecord(
                state_tensor=torch.randn(7, 9, 9),
                policy_target=torch.softmax(torch.randn(81), dim=0),
                opp_policy_target=torch.softmax(torch.randn(81), dim=0),
                opp_legal_mask=torch.ones(81),
                legal_mask=torch.ones(81),
                current_player=1 if m % 2 == 0 else -1,
                value_target=1.0,
                score_target=0.5,
                ownership_target=torch.zeros(9),
            )
            moves.append(rec)
        game = GameRecord(moves=moves, winner=1, game_length=10)
        buf.add_game(game)

    print(f"After 5 games: buffer size = {len(buf)}")
    assert len(buf) == 50

    # Sample a batch
    batch = buf.sample(16)
    assert batch.state_tensors.shape == (16, 7, 9, 9)
    assert batch.policy_targets.shape == (16, 81)
    assert batch.value_targets.shape == (16, 1)
    assert batch.score_targets.shape == (16, 1)
    assert batch.ownership_targets.shape == (16, 9)
    assert batch.legal_masks.shape == (16, 81)
    assert batch.opp_legal_masks.shape == (16, 81)
    print(f"Sampled batch: shapes correct")

    # Test circular overwrite
    big_buf = ReplayBuffer(capacity=20)
    for _ in range(5):
        game = GameRecord(
            moves=[MoveRecord(
                state_tensor=torch.randn(7, 9, 9),
                policy_target=torch.softmax(torch.randn(81), dim=0),
                opp_policy_target=torch.softmax(torch.randn(81), dim=0),
                opp_legal_mask=torch.ones(81),
                legal_mask=torch.ones(81),
                current_player=1,
            ) for _ in range(10)],
            winner=1, game_length=10,
        )
        big_buf.add_game(game)
    assert len(big_buf) == 20, f"Circular buffer should cap at 20, got {len(big_buf)}"
    print(f"Circular overwrite: PASSED")

    print("\n=== Episode 12 PASSED ===")
