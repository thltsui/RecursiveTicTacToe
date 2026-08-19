# ADR 0002: JSON-configured lightweight Transformer and batched self-play

- Status: accepted
- Date: 2026-08-19
- Owners: model architecture, training, evaluation, and deployment
- Supersedes: the hard-coded architecture settings in `train.py` and
  `transformer/train_transformer.py`

## Context

Ultimate Tic-Tac-Toe has non-local structure. A move in one cell determines
the sub-board of the next move, while wins are decided on both the local and
macro boards. Full token-to-token attention is therefore a useful inductive
bias: it can relate a particular cell to a particular remote threat without
first reducing the rest of the board to a global average.

The old repository had two launchers. `train.py` hard-coded a 192-channel,
10-block CNN even though its header described a small model, while
`transformer/train_transformer.py` hard-coded a 128-channel, four-layer
Transformer and monkey-patched the CNN constructor. This made the executable,
not a saved configuration, the source of truth.

The existing value head also dominated small architectures. Its
`2592 -> 512 -> 128` dense path contributed about 1.4 million parameters even
when the trunk was reduced. Shrinking only Transformer width therefore did not
produce a genuinely small deployment model.

Finally, mixed self-play generated games sequentially. The repository already
had a correctness-tested batched MCTS primitive, but no production data path
used it. The neural network consequently evaluated one leaf at a time during
self-play, leaving the M2's batched compute underused.

## Decision

### One launcher and one JSON source of truth

All training starts through:

```bash
uv run train.py --config configs/training/lite_transformer.json
```

There is no architecture-specific training script. The JSON `model` object
defines every shape that affects checkpoint compatibility:

- architecture;
- channel/embedding width;
- layer count and attention-head count;
- feed-forward multiplier and dropout;
- positional encoding;
- value-head convolution, hidden, and feature widths.

The loader rejects unknown keys instead of silently accepting misspellings.
The command below validates and prints the canonical configuration without
starting training:

```bash
uv run train.py \
  --config configs/training/lite_transformer.json \
  --print-config
```

### Lightweight Transformer shape

The initial profile uses:

- 96-dimensional tokens;
- three encoder layers;
- four attention heads;
- a two-times-width feed-forward layer;
- hierarchical positional encoding;
- an `8 -> 128 -> 64` spatial value head.

The hierarchical encoding adds a learned macro-board position and a learned
within-sub-board cell position to every token. This preserves full attention
while explicitly representing the game's repeated 3-by-3 structure.

The value head still flattens a small spatial map before its dense layer. It is
therefore compact without discarding which sub-board produced a feature. The
profile is constrained by tests to remain below 500,000 parameters.

### Batched self-play

`self_play_batch_size` controls how many independent games contribute one leaf
to a network evaluation batch. Every game retains its own:

- MCTS tree and exactly one backup per simulation;
- Dirichlet noise sample and early-ply noise schedule;
- move temperature;
- forced opening;
- terminal result and W/D/L perspective targets.

Pure self-play and random-prefix strong reanalysis use batching. Games against
a distinct frozen best network remain sequential for now because alternating
two evaluators cannot be represented by the existing single-evaluator batched
MCTS interface without grouping and rescheduling turns. This does not weaken
target semantics; it only leaves a smaller performance optimization for later.

### Checkpoint metadata

New checkpoints have format version 2 and contain `model_config` alongside the
network and optimizer states. Training resume, W/D/L calibration, candidate
evaluation, and web deployment all construct the network through the same
model factory.

Old checkpoints remain loadable through shape inference. That path is marked
as legacy because head count and dropout cannot be recovered from weights.
New artifacts never depend on this inference.

### Release gates

Training loss alone does not authorize deployment. A candidate must pass the
versioned gates in
`configs/evaluation/lite_transformer_acceptance.json`:

- no more than 500,000 parameters;
- median 400-simulation search below the deployment latency budget;
- at least 100 alternating-color arena games and a non-inferior point score
  against the current champion, with draws worth half a point;
- at least 10,000 held-out deployment-representative positions;
- an evaluation buffer whose SHA-256 fingerprint differs from the buffer used
  to fit temperature scaling;
- a checkpoint explicitly marked as calibrated;
- W/D/L negative log-likelihood, Brier score, and classwise ECE limits.

The latency budget must be measured on deployment-equivalent CPU hardware. A
Mac result is useful for regression comparison but does not substitute for a
Fly measurement.

The non-training tools are:

```bash
# Architecture or checkpoint latency only
uv run python scripts/benchmark_model.py \
  --config configs/training/lite_transformer.json

# Combined latency, arena, and held-out calibration report
uv run python scripts/evaluate_candidate.py \
  --candidate checkpoints/lite_transformer_calibrated.pt \
  --baseline checkpoints/best_ever_model.pt \
  --held-out-buffer checkpoints/calibration_evaluation_buffer.pt \
  --output reports/lite_transformer_candidate.json

# Deterministic pass/fail decision
uv run python scripts/check_model_acceptance.py \
  --report reports/lite_transformer_candidate.json
```

These commands evaluate existing weights. None performs gradient updates.
`calibration_evaluation_buffer.pt` must not be the buffer previously passed to
`scripts/calibrate_wdl.py`; the checkpoint and evaluation report carry file
fingerprints and the acceptance checker enforces that separation.

## Consequences

### Positive

- Model size is reviewable in JSON and persisted in every new checkpoint.
- The selected Transformer preserves global relational reasoning while being
  substantially smaller than both the deployed CNN and old Transformer.
- Batched self-play amortizes neural evaluation across independent games.
- Calibration, strength, size, and latency are explicit deployment gates.
- A printed config and a saved checkpoint describe the same architecture.

### Costs and limitations

- This architecture requires training from scratch; its compact head and
  hierarchical positional parameters do not match old weight shapes.
- Batching does not reduce the number of MCTS simulations. It improves hardware
  utilization, so wall-clock gain must be measured rather than assumed.
- `vs_best` generation and the arena are still sequential.
- The first acceptance thresholds are engineering hypotheses. They are
  versioned so measured deployment behavior can justify later changes.
- The JSON profile selects a starting simulation budget, not proof of optimal
  hyperparameters.

## Training boundary

This ADR and its implementation prepare the pipeline only. No training,
calibration fitting, arena run, or latency benchmark is started automatically.
