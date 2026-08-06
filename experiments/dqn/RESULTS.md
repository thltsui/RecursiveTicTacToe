# Practical DQN experiment

This experiment trains one DQN player against a uniform-random opponent,
alternating X and O between episodes. Replay transitions span from one DQN
decision to its next decision after the opponent replies. Both states are
therefore encoded from the learning agent's perspective, and non-terminal
targets use the standard positive bootstrap:

`target = reward + gamma * max(Q_target(next_state))`

Only the DQN player's decisions are stored. Training begins after 1,000
transitions, samples batches of 32 every four DQN decisions, and copies the
online network to the target network every 500 gradient steps.

## Recorded runs

Both runs used seed 0, 20,000 training episodes, a 100,000-transition replay
buffer, gamma 0.99, epsilon decay from 0.5 to 0.05, and 300 fixed evaluation
games every 1,000 episodes.

| Adam learning rate | Training time | Best 300-game checkpoint | Final 300-game result | Fresh 2,000-game final evaluation |
|---|---:|---:|---:|---:|
| 0.001 | 2,451 s | 54.3% wins at 5k | 37.3% W / 27.0% D / 35.7% L | 35.1% W / 27.3% D / 37.6% L |
| 0.0001 | 2,286 s | 63.7% wins at 17k | 59.3% W / 15.3% D / 25.3% L | 60.1% W / 13.1% D / 26.8% L |

The higher learning rate repeatedly collapsed after apparently strong
checkpoints. The lower rate learned more gradually and sustained its gains, so
`0.0001` is now the default.

## Reproduce

```bash
experiments/dqn/run_experiment.sh
```

The runner accepts environment-variable overrides such as `LEARNING_RATE`,
`EPISODES`, `SEED`, and `DEVICE`. It writes the full configuration and
checkpoint evaluations to `training_log.json`.

The recorded logs, 2,000-game evaluations, and comparison chart are under
`experiments/dqn/results/`.
