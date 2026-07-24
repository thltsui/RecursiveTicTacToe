# MLflow for this project — analysis and recommendation

Status: **reference document, not a plan of record.** Nothing here is being implemented now. This is a "come back to this if the project outgrows its current tracking" note, written after auditing what's actually logged today (`training_metrics.json`, `EloTracker`, `06_evaluation/metrics.py`, `web_app/app.py`'s checkpoint loading) rather than as a generic pitch for the tool.

## Recommendation up front

- **Full MLflow tracking server: not now.** The existing JSON-file logging is adequate at the current scale (4 checkpoint lineages, ~200 iterations each). The added ceremony (server, dependency, instrumentation) isn't paid back yet.
- **Model registry for `best_model.pt`: worth doing if/when convenient.** This is the one piece that fixes a real fragility in `web_app/app.py`'s deployment logic, independent of whether tracking is ever adopted.
- Revisit the "not now" call if the project starts running many parallel architecture/hyperparameter variants, or if cross-run comparison starts happening often enough that manually opening JSON files becomes the bottleneck.

## What's tracked today, mapped to MLflow concepts

`trainer.py`'s training loop writes one flat JSON list per run to `checkpoints/<run>/training_metrics.json`, one dict per iteration containing `loss_total` and its five components (`loss_policy`, `loss_value`, `loss_score`, `loss_ownership`, `loss_opp_policy`), `buffer_size`, `learning_rate`, `time_seconds`, and — only on iterations where the arena runs (every `arena_every_n`) — `arena_win_rate`, `arena_wins`, `arena_losses`, `arena_draws`.

| Current artifact | MLflow concept |
|---|---|
| Per-iteration entries in `training_metrics.json` | `mlflow.log_metric(name, value, step=iteration)` |
| `TrainingConfig` dataclass fields (`learning_rate`, `num_simulations`, `dirichlet_alpha`, `dirichlet_epsilon`, `temperature_threshold`, `channels`, `num_blocks`, `win_rate_threshold`, `arena_games`, loss weights `lambda_value`/`lambda_score`/`lambda_ownership`/`lambda_opp`) | `mlflow.log_params(asdict(config))` — one call at run start |
| A `checkpoints/large_v*` directory (e.g. `large_v3_pure_self_play`, `large_v5_fixed_mcts`) | One MLflow **run** (arguably one **experiment**, with training iterations as steps) |
| `best_model.pt`, `checkpoint_iter00200_elo1.pt`, `replay_buffer.pt` | MLflow **artifacts** |

Two things are *computed* somewhere in the codebase but never make it into the per-iteration record:

- **Elo.** `06_evaluation/elo.py`'s `EloTracker` keeps its own in-memory rating history, but its output never reaches `training_metrics.json`. The only surviving trace of Elo is the bucket baked into checkpoint filenames (`checkpoint_iter00050_elo1.pt`), which is far coarser than an actual per-iteration Elo curve.
- **Policy entropy.** `06_evaluation/metrics.py` has `compute_policy_entropy()` with guidance in its own docstring ("random ~4.4, well-trained 1.0–2.0"), and `trainer.py`'s own training-diagnostics comment block lists "policy entropy hits 0 → collapsed" as a warning sign to watch for. Nothing in the training loop actually calls it — there's a documented failure mode with no metric watching for it.

Worth noting: `06_evaluation/metrics.py` also defines a `MetricsLogger` class with its own JSON history log and a 6-panel matplotlib figure (loss components, Elo, game length, policy entropy, value accuracy, arena win rate). It is never invoked from `trainer.py`. It's effectively a hand-built precursor to what MLflow's run-comparison UI would give for free — evidence the need was already felt once.

## Where MLflow would actually help: cross-run comparison

Single-run tracking is already fine — `training_metrics.json` is a flat list you can load into pandas in two lines. The real gap is comparing *across* runs. Right now, understanding how `large_v3_pure_self_play`, `large_v4_deep_value`, and `large_v5_fixed_mcts` differ means opening multiple JSON files and, for the hyperparameters, grepping the stdout banner in files like `logs/training_v5_fixed_mcts.txt` — because no checkpoint directory persists a `config.json` next to its weights. The meaning encoded in names like "fixed_mcts" or "pure_self_play" exists only as a mental note, not as queryable data.

MLflow's comparison view (parallel-coordinates plot, filterable run table, overlaid metric charts) would replace that manual process — e.g. filtering to "runs where `dirichlet_alpha = 0.3`" or overlaying `loss_value` curves for all four lineages on one chart. At 4 runs this is a nice-to-have; it becomes a real time-saver once there are enough concurrent variants that eyeballing filenames stops working.

## The model registry case — this one stands on its own

`web_app/app.py`'s `load_network()` is genuinely fragile, independent of the tracking-server question:

```python
search_dirs = [
    os.path.join(PROJECT_ROOT, 'checkpoints/large_v4_deep_value'),
    os.path.join(PROJECT_ROOT, 'checkpoints/large_v5_fixed_mcts'),
    os.path.join(PROJECT_ROOT, 'checkpoints/large_v3_pure_self_play'),
]
```

Deployment currently means: a hardcoded, order-dependent list of directory names, checked in sequence for a `best_model.pt`, falling back to sorting `checkpoint_iter*.pt` by filename if that's missing. There's no stored config to confirm architecture either — `load_network()` reverse-engineers `channels` and `num_blocks` by inspecting tensor shapes in the loaded state dict rather than reading them from anywhere. "Promoting" a new best model today means renaming or reordering folders and hoping the search order still resolves correctly.

Registering `best_model.pt` under an MLflow Model Registry entry (e.g. `ultimate-ttt-alphazero`) with a `Production` stage/alias would let `app.py` load via `mlflow.pytorch.load_model("models:/ultimate-ttt-alphazero/Production")`. Promotion becomes an explicit, auditable action (a stage transition) instead of implicit directory-priority ordering. This is a correctness fix, not tooling polish — it's worth doing on its own merits whenever it's convenient, regardless of whether full experiment tracking ever gets adopted.

## Honest tradeoff on the full server

This is a solo project with four lineages and roughly 200 iterations per run. Standing up `mlflow server` (or even local file-store tracking) means a new dependency, a UI process to keep running, and instrumentation work across `trainer.py` — in exchange for a comparison workflow that, at this scale, takes about as long to do by hand (open two JSON files) as it would to spin up the MLflow UI and navigate to the right run. That calculus flips once there's enough run volume or enough parallel experimentation that manual comparison actually becomes the bottleneck — at that point, revisit this document and start with `mlflow.log_metric`/`log_params` calls dropped directly into the existing loop in `trainer.py` (no restructuring needed), plus wiring `EloTracker` and `compute_policy_entropy()` into the per-iteration metrics dict since both already exist and just aren't called.
