from __future__ import annotations

import importlib
import random
import sys
from pathlib import Path

import pytest
import torch


DQN_DIR = Path(__file__).resolve().parents[1] / "experiments" / "dqn"
sys.path.insert(0, str(DQN_DIR))
dqn_train = importlib.import_module("train")


class RecordingAgent:
    def __init__(self) -> None:
        self.transitions = []
        self.observe_calls = 0

    def select_action(self, state) -> int:
        return dqn_train.get_legal_moves(state)[0]

    def observe(
        self,
        state,
        move,
        reward,
        next_state,
        done,
        **_kwargs,
    ) -> None:
        self.transitions.append((state, move, reward, next_state, done))
        self.observe_calls += 1


@pytest.mark.parametrize("agent_player", [1, -1])
def test_training_transitions_span_agent_decisions(agent_player: int) -> None:
    agent = RecordingAgent()

    winner = dqn_train.play_training_game(
        agent,
        agent_player=agent_player,
        rng=random.Random(7),
    )

    assert winner in (-1, 0, 1)
    assert agent.transitions
    assert agent.observe_calls == len(agent.transitions)
    assert agent.transitions[-1][-1] is True

    for state, move, _reward, next_state, done in agent.transitions:
        assert state.current_player == agent_player
        assert move in dqn_train.get_legal_moves(state)
        assert next_state.is_terminal is done
        if not done:
            assert next_state.current_player == agent_player


def test_invalid_agent_player_is_rejected() -> None:
    with pytest.raises(ValueError, match="agent_player"):
        dqn_train.play_training_game(
            RecordingAgent(),
            agent_player=0,
            rng=random.Random(0),
        )


def test_real_dqn_training_step_produces_finite_weights() -> None:
    agent = dqn_train.DQNAgent(batch_size=2, epsilon=1.0)

    dqn_train.play_training_game(
        agent,
        agent_player=1,
        rng=random.Random(11),
        train_every=1,
        learning_starts=2,
    )

    assert agent._environment_steps > 2
    assert agent._grad_steps > 0
    assert all(
        torch.isfinite(parameter).all()
        for parameter in agent.online_net.parameters()
    )
