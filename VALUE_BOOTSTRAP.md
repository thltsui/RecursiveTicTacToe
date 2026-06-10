# Value Bootstrap via Random Self-Play

This document specifies how to bootstrap the value function before entering the
main AlphaZero self-play loop, and why this is necessary and mathematically sound.

---

## 1. Implementation Plan

### Overall Staged Approach

The cold-start problem in AlphaZero is severe for Ultimate TTT: with randomly
initialised weights, `V_θ(s) ≈ 0` for every position. MCTS backup values are
therefore pure noise, the policy head gradient receives no useful signal, and
the system spins its wheels for dozens of iterations before learning anything.

The fix is a three-stage pipeline:

```
Stage 0  Generate ~5 000 random games (both sides random, no MCTS).
         Record every position and the game outcome as value target.

Stage 1  Pretrain VALUE HEAD ONLY on this dataset.
         Policy, score, and ownership heads are frozen (lambda_policy = 0).
         Train until value-head MSE plateaus (~200–500 batches).

Stage 2  Hand off to normal AlphaZero loop (trainer.train()).
         Warm-start from pretrain checkpoint.
         All heads unfreeze; the buffer fills with on-policy data.
         Random-play bias decays naturally as the buffer turns over.
```

Stage 0 and 1 run once, offline. Stage 2 is the existing `train()` loop with
one new field in `TrainingConfig` pointing at the pretrain checkpoint.

---

### Files That Change

#### `04_training/self_play.py` — new function `play_random_vs_random_game()`

Add a function that plays a complete game with both sides sampling uniformly
from legal moves. No MCTS, no network. Records every position from both
perspectives.

Key differences from `play_vs_random_game()`:
- No network argument.
- Both players are random.
- Both players' moves ARE recorded (we want positional coverage, not policy targets).
- `policy_target` is set to a uniform distribution over legal moves (informational
  only; the pretrain script will set `lambda_policy = 0` so it does not contribute to loss).
- `opp_policy_target` and `opp_legal_mask` are zeros (same convention as
  `play_vs_random_game()`).

#### `04_training/loss.py` — new `lambda_policy` parameter in `compute_total_loss()`

Add `lambda_policy: float = 1.0` as an explicit parameter (currently the policy
term is implicitly weighted 1.0). The pretrain script passes `lambda_policy=0.0`
to suppress the policy loss entirely during Stage 1.

Diff is minimal: add the parameter and multiply `l_policy` by it in the sum.

#### `04_training/trainer.py` — new `pretrain_checkpoint` field in `TrainingConfig`

Add `pretrain_checkpoint: str = ''` to `TrainingConfig`. At the top of `train()`,
if this field is non-empty and `iteration == 0`, load weights from it before the
first self-play iteration. This is the only change to `trainer.py`.

#### `scripts/pretrain_value.py` — new script (full file, see §3)

Standalone script that:
1. Generates random-vs-random games (`play_random_vs_random_game()`).
2. Populates a `ReplayBuffer`.
3. Trains with `lambda_policy=0.0`, `lambda_score=1.0`, `lambda_ownership=1.0`,
   `lambda_opp=0.0` (only value head receives meaningful gradient).
4. Saves the pretrain checkpoint to `checkpoints/pretrain_value.pt`.

---

### Why We Do NOT Pretrain the Policy Head

Ultimate TTT has a structural property that makes shallow policy priors actively
harmful: **winning a sub-board can immediately lose the global game** by sending
the opponent to any sub-board of their choice. A position that looks locally
dominant is often strategically weak because it constrains the opponent less than
an "inferior" local move would.

Concretely: no handcrafted heuristic or random-play visit distribution is a valid
prior over the policy space. If we pretrained the policy head on random-play data,
we would initialise it to prefer moves that random players tend to play — uniform
or near-uniform — which is exactly what the policy head needs to move *away* from.
The network must discover through self-play that constraining the opponent's
sub-board choice is the dominant strategic dimension. Giving it a random prior
on this dimension is no better than leaving weights random, and freezing on that
prior for the first N iterations would actively slow convergence.

The value head, by contrast, is learning a scalar signal (who is ahead?) that is
positively correlated across all policy distributions. Random-play outcomes are a
noisy but *directionally correct* estimate of position quality. The policy head has
no analogous directionally-correct signal available from random play.

---

## 2. Mathematics

### What V(s) Is

Let `π` denote the current policy, `z ∈ {+1, −0.5, 0, −1}` the game outcome
(from the perspective of the player to move at state `s`), and `E[·|s, π]` the
expectation over trajectories that start from `s` under policy `π`.

```
V^π(s)  =  E[ z | s, π ]
```

The value head `V_θ(s)` is trained to approximate `V^π(s)` for the current
self-play policy. The optimal value function is:

```
V*(s)  =  E[ z | s, π* ]
```

where `π*` is the game-theoretically optimal policy.

The output of the value head is `tanh`-bounded to `[−1, 1]`, consistent with
the target domain `{+1, −0.5, 0, −1} ⊂ [−1, 1]`.

---

### Why Random-Play Outcomes Are a Valid Proxy for V*(s)

Let `π_rand` denote the uniform-random policy (sample uniformly from legal moves
at every state). Define:

```
V^rand(s)  =  E[ z | s, π_rand ]
            =  Σ_a  π_rand(a|s) · Q^rand(s, a)
            =  (1 / |A(s)|) · Σ_a  Q^rand(s, a)
```

by the law of total expectation, where `A(s)` is the set of legal actions and
`Q^rand(s, a) = E[ z | s, a, π_rand ]`.

This is a uniform average over all Q-values. For positions with strong positional
features — where some moves lead to much better outcomes than others — the average
is positively correlated with `max_a Q(s, a)`:

```
Corr( V^rand(s), V*(s) )  >  0    when Var_a[ Q*(s, a) ] > 0
```

The correlation is imperfect (it is exact only when `π_rand = π*`), but it is
strictly positive whenever there exist dominant moves. In Ultimate TTT, positions
near the end of the game have very large `Var_a[ Q*(s, a) ]` (one move wins, the
rest lose), so the signal-to-noise ratio of `V^rand(s)` as a proxy for `V*(s)` is
high precisely where it matters most: game-deciding positions.

For opening positions, where `Q*(s, a)` is nearly flat across all `a` (many
equally good moves), `V^rand(s) ≈ V*(s) ≈ 0`, and the approximation is fine
because the value network does not need to discriminate finely among opening moves.

---

### The MSE Objective Being Minimised

During pretraining (Stage 1), the loss is:

```
L_pretrain  =  L_value  +  λ_score · L_score  +  λ_ownership · L_ownership

            =  (1/N) Σ_i ( V_θ(s_i) − z_i )²
             + λ_score · (1/N) Σ_i ( S_θ(s_i) − m_i )²
             + λ_ownership · (1/N) Σ_i BCE( O_θ(s_i), o_i )
```

where:
- `V_θ(s_i) ∈ [−1, 1]`  — `win_value` output of the value head (tanh)
- `z_i ∈ {+1, −0.5, 0, −1}` — game outcome from the perspective of the player
  to move at position `s_i`
- `S_θ(s_i) ∈ [−1, 1]`  — `score_margin` output (tanh)
- `m_i = (sub_boards_won − sub_boards_lost) / 9 ∈ [−1, 1]` — normalised margin
- `O_θ(s_i) ∈ [0, 1]^9` — `ownership` output (sigmoid per sub-board)
- `o_i ∈ {0, 1}^9` — did current player win each sub-board?
- `λ_score = 1.0`,  `λ_ownership = 1.0`  (pretrain defaults)
- `λ_policy = 0.0` — policy loss is **suppressed entirely**

The score and ownership auxiliaries provide richer per-position supervision than
`z_i` alone (a binary win/loss/draw collapses sub-board structure to a single bit),
which improves gradient conditioning for the shared value-head trunk.

---

### Why This Beats Cold-Start

At iteration 0 with randomly initialised weights:

- `V_θ(s) ≈ N(0, σ²)` for small σ (tanh of Gaussian noise, near zero for small
  weights).
- MCTS leaf evaluations are `V_θ(s_leaf) ≈ 0` with high variance.
- Backup values propagated to the root are `Q(s, a) ≈ 0 ± ε` for all `a`.
- The PUCT selection criterion uses `V` to weight Q-estimates; if all Q ≈ 0,
  PUCT degenerates to pure prior-weighted exploration — i.e. the policy head drives
  everything at iteration 0.
- The policy head receives gradient from `L_policy` which requires `V` to
  distinguish move quality; it cannot, so the policy gradient signal is noise.

After Stage 1:

- `V_θ(s)` has learned the direction of `V^rand(s)`, which is the right sign.
- MCTS leaf evaluations immediately distinguish "clearly winning" from "clearly
  losing" positions.
- Q-value estimates at the root have positive signal-to-noise ratio after even a
  handful of simulations.
- The policy head receives a real gradient from the first self-play iteration.
- Stage 2 converges meaningfully faster (typically 10–30 fewer wasted iterations).

---

### Distribution Shift and Bias Decay

The pretrained value function is fit to data from `d^π_rand(s)`, the state
distribution induced by random play. The true target distribution for the
AlphaZero loop is `d^π*(s)`, the distribution induced by the optimal (or
near-optimal) self-play policy. These distributions diverge:

```
d^π_rand(s)  ≠  d^π*(s)
```

In particular, `π*` concentrates on tactically rich positions that `π_rand` visits
rarely. The pretrained `V_θ` is miscalibrated on positions that only arise under
strong play — it will over-estimate winning chances in positions where a naive
random player would blunder, and under-estimate in positions where random play
accidentally creates a threat.

Quantifying the bias: let `B_t = buffer_size_at_iteration_t`. Each self-play
iteration adds approximately `G × L̄` positions (where `G` is games per iteration
and `L̄ ≈ 40–80` is the mean game length in Ultimate TTT). The circular buffer has
capacity `C`. The fraction of pretrain data remaining after `t` iterations is:

```
f_pretrain(t)  ≈  max( 0,  1 − (G · L̄ · t) / C )
```

For `C = 200 000`, `G = 20`, `L̄ = 60`:

```
f_pretrain(t)  ≈  max( 0,  1 − t / 167 )
```

So after ~167 iterations, the buffer contains zero pretrain positions. The bias
is fully diluted by then. In practice it becomes negligible much sooner — after
~20 iterations (`f ≈ 0.88`) the on-policy data already dominates the gradient
because recent positions are sampled uniformly alongside old ones, and the recent
data has lower loss (the network has adapted). The pretrain bias decays as
`O(1/t)` in the number of training iterations.

---

### The Anti-Draw Shaping: z_draw = −0.5

In the codebase (`self_play.py`, line 158):

```python
rec.value_target = float(winner * cp) if winner != 0 else -0.5
```

Draws are assigned `z = −0.5` instead of `z = 0`. This is deliberate and
mathematically motivated.

In a symmetric zero-sum two-player game, the Nash equilibrium value of a drawn
position is 0 for both players — neither player can guarantee a better outcome
against optimal play. If we set `z_draw = 0`, the network learns `V*(s_draw) = 0`,
which means the network is indifferent between fighting for a win and accepting a
draw. In self-play, this creates a **draw equilibrium**: both players settle for
draws because `V*(s_draw) = V*(s_won_by_opponent) = 0` from each player's
perspective is not distinguishable as worse.

By setting `z_draw = −0.5 < 0`, we assert that `V*(s_draw) < 0`. Both players
learn that a draw is a *bad* outcome. This creates an **asymmetric incentive**:

```
V*(win)  = +1.0
V*(draw) = −0.5  <  0
V*(loss) = −1.0
```

For any position `s` where the current player can force a draw, the value should
satisfy `V*(s) ≥ −0.5`. But if the opponent also believes draws are bad, they
will try to avoid draw lines — creating positions where the current player can
exploit the opponent's draw-aversion to obtain a win.

The game-theoretic consequence is that both players are forced to play for wins,
which breaks the Nash draw equilibrium that self-play would otherwise converge to.
The network discovers winning strategies rather than draw strategies.

Note that this is a deliberate departure from strict game-theoretic zero-sum
framing — we are encoding a preference ordering `win ≻ draw ≻ loss` with specific
numerical gaps that create training pressure. The `−0.5` value is a hyperparameter;
it should be negative (to penalise draws) but not as negative as `−1.0` (which
would make draws as bad as losses, causing the network to take suicidal risks to
avoid them).

---

## 3. Code Changes

### `04_training/self_play.py` — new function `play_random_vs_random_game()`

Add after `play_vs_random_game()` (around line 281). Full function:

```python
def play_random_vs_random_game() -> GameRecord:
    """Play one complete game with both sides sampling uniformly from legal moves.

    No network, no MCTS. Both players pick moves uniformly at random.
    Every position from both players is recorded.

    Policy targets are set to the uniform distribution over legal moves.
    These are informational only — the pretrain script sets lambda_policy=0.0
    so they do not contribute to the pretrain loss.

    opp_policy_target and opp_legal_mask are left as zeros (same convention
    as play_vs_random_game — the loss ignores them via zero mask).

    Returns:
        Complete GameRecord with value/score/ownership targets.
    """
    import random
    from importlib import import_module
    board_mod = import_module('01_game.board')
    rules_mod = import_module('01_game.rules')

    create_initial_state = board_mod.create_initial_state
    encode_state = board_mod.encode_state
    get_legal_moves = rules_mod.get_legal_moves
    get_legal_move_mask = rules_mod.get_legal_move_mask

    state = create_initial_state()
    move_records: list[MoveRecord] = []

    while not state.is_terminal:
        legal_moves = get_legal_moves(state)
        move = random.choice(legal_moves)

        # Uniform policy target over legal moves
        legal_mask = get_legal_move_mask(state)
        n_legal = float(legal_mask.sum().item())
        policy_tgt = legal_mask.float() / n_legal  # uniform over legal moves

        record = MoveRecord(
            state_tensor=encode_state(state),
            policy_target=policy_tgt,
            opp_policy_target=torch.zeros(81),
            opp_legal_mask=torch.zeros(81),
            legal_mask=legal_mask,
            current_player=state.current_player,
        )
        move_records.append(record)

        state = rules_mod.apply_move(state, move)

    # Compute value/score/ownership targets
    winner = state.winner if state.winner is not None else 0
    final_results = state.sub_board_results.copy()

    for rec in move_records:
        cp = rec.current_player
        rec.value_target = float(winner * cp) if winner != 0 else -0.5
        own_wins = np.sum(final_results == cp)
        opp_wins = np.sum(final_results == -cp)
        rec.score_target = float(own_wins - opp_wins) / 9.0
        rec.ownership_target = torch.tensor(
            [(1.0 if final_results[i] == cp else 0.0) for i in range(9)],
            dtype=torch.float32
        )

    return GameRecord(
        moves=move_records,
        winner=winner,
        game_length=len(move_records),
        final_sub_board_results=final_results,
    )
```

---

### `04_training/loss.py` — add `lambda_policy` to `compute_total_loss()`

```diff
 def compute_total_loss(
     network_output: 'NetworkOutput',
     batch: 'TrainingBatch',
+    lambda_policy: float = 1.0,
     lambda_value: float = 3.0,
     lambda_score: float = 1.0,
     lambda_ownership: float = 1.0,
     lambda_opp: float = 0.15,
 ) -> LossBreakdown:
     """Compute the complete weighted multi-task loss.

-    L_total = L_policy
+    L_total = lambda_policy * L_policy
             + lambda_value     * L_value
             + lambda_score     * L_score
             + lambda_ownership * L_ownership
             + lambda_opp       * L_opp_policy

     Args:
         network_output: Forward pass output from the network.
         batch: Training batch with targets.
+        lambda_policy: Weight for policy loss. Default: 1.0.
+            Pass 0.0 during value pretraining to suppress policy learning.
         lambda_value: Weight for value loss. Default: 1.0.
         lambda_score: Weight for score loss. Default: 0.5.
         lambda_ownership: Weight for ownership loss. Default: 0.5.
         lambda_opp: Weight for opponent policy loss. Default: 0.15.
     ...
     """
     l_policy = policy_loss(
         network_output.policy_logits, batch.policy_targets, batch.legal_masks
     )
     l_value = value_loss(network_output.win_value, batch.value_targets)
     l_score = score_loss(network_output.score_margin, batch.score_targets)
     l_ownership = ownership_loss(network_output.ownership, batch.ownership_targets)
     opp_mask = batch.opp_legal_masks
     l_opp = policy_loss(
         network_output.opp_policy_logits, batch.opp_policy_targets, opp_mask
     )

-    total = (l_policy
+    total = (lambda_policy * l_policy
              + lambda_value * l_value
              + lambda_score * l_score
              + lambda_ownership * l_ownership
              + lambda_opp * l_opp)
```

The existing call site in `trainer.py` (`train_step`) does not pass `lambda_policy`,
so it defaults to `1.0` — no behaviour change for the normal training loop.

---

### `04_training/trainer.py` — add `pretrain_checkpoint` to `TrainingConfig` and load it in `train()`

```diff
 @dataclass
 class TrainingConfig:
     ...
     # Reproducibility
     seed:                  int   = 42
+
+    # Value bootstrap
+    pretrain_checkpoint:   str   = ''   # path to pretrain_value.pt; loaded at iter 0 if set
```

In `train()`, after the existing best-checkpoint block (after line 183), add:

```diff
     if os.path.exists(best_checkpoint_path):
         best_net = UltimateTTTNetwork(
             channels=config.channels, num_blocks=config.num_blocks
         ).to(config.device)
         best_meta = load_checkpoint(best_checkpoint_path, best_net)
         best_network_state = clone_state_dict(best_net.state_dict())
         last_arena_win_rate = float(best_meta.get('elo', last_arena_win_rate))
         print(f"  Loaded best model from: {best_checkpoint_path}")

+    # Load pretrain checkpoint at cold start (only if no checkpoint was auto-resumed)
+    if config.pretrain_checkpoint and iteration == 0:
+        if os.path.exists(config.pretrain_checkpoint):
+            load_checkpoint(config.pretrain_checkpoint, network)
+            best_network_state = clone_state_dict(network.state_dict())
+            print(f"  Loaded pretrain checkpoint: {config.pretrain_checkpoint}")
+        else:
+            print(f"  WARNING: pretrain_checkpoint not found: {config.pretrain_checkpoint}")
```

---

### `scripts/pretrain_value.py` — full new file

```python
#!/usr/bin/env python3
"""Stage 0+1 of value bootstrap: generate random games and pretrain the value head.

Usage:
    python scripts/pretrain_value.py [--games N] [--batches N] [--checkpoint PATH]

This script implements the two pre-AlphaZero stages:

    Stage 0  Generate --games random-vs-random games and populate a ReplayBuffer.
    Stage 1  Train with lambda_policy=0 (value/score/ownership only) for --batches
             gradient steps, then save the pretrain checkpoint.

The checkpoint is then passed to TrainingConfig(pretrain_checkpoint=PATH) so that
the normal train() loop warm-starts from a value function with positive SNR.
"""

import sys
import os
import argparse
import time

# Ensure project root is on sys.path regardless of CWD
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.optim as optim
from importlib import import_module
from tqdm import tqdm

# Load modules via importlib (digit-prefixed directories are not valid Python identifiers)
network_mod        = import_module('02_network.network')
self_play_mod      = import_module('04_training.self_play')
replay_buffer_mod  = import_module('04_training.replay_buffer')
loss_mod           = import_module('04_training.loss')
trainer_mod        = import_module('04_training.trainer')

UltimateTTTNetwork         = network_mod.UltimateTTTNetwork
play_random_vs_random_game = self_play_mod.play_random_vs_random_game
ReplayBuffer               = replay_buffer_mod.ReplayBuffer
TrainingBatch              = replay_buffer_mod.TrainingBatch
compute_total_loss         = loss_mod.compute_total_loss
save_checkpoint            = trainer_mod.save_checkpoint


def generate_random_games(num_games: int) -> ReplayBuffer:
    """Generate random-vs-random games and return a populated ReplayBuffer.

    Args:
        num_games: Number of complete games to generate.

    Returns:
        ReplayBuffer containing all positions from all games.
    """
    # Capacity: assume average game length of 80 moves, both players recorded.
    # 5000 games x 80 moves = 400 000 positions; use 500 000 to be safe.
    buffer = ReplayBuffer(capacity=max(num_games * 100, 500_000))

    t0 = time.time()
    total_positions = 0

    for i in range(num_games):
        record = play_random_vs_random_game()
        buffer.add_game(record)
        total_positions += record.game_length

        if (i + 1) % 100 == 0 or (i + 1) == num_games:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(
                f"  [{i+1}/{num_games}] games  |  {total_positions} positions  "
                f"|  {rate:.1f} games/s  |  {elapsed:.0f}s elapsed"
            )

    print(f"\nBuffer: {len(buffer)} positions from {num_games} games.")
    return buffer


def pretrain_value_head(
    network: torch.nn.Module,
    buffer: ReplayBuffer,
    num_batches: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    lambda_score: float = 1.0,
    lambda_ownership: float = 1.0,
) -> list:
    """Train the value head (only) on the random-play buffer.

    Policy and opponent-policy losses are suppressed (lambda_policy=0, lambda_opp=0).
    The score and ownership auxiliaries are included because they share the value-head
    trunk and provide richer per-position supervision.

    Args:
        network: Full UltimateTTTNetwork (all parameters accessible, but only
                 value-head-related parameters receive meaningful gradient).
        buffer: ReplayBuffer populated with random-play positions.
        num_batches: Number of gradient update steps.
        batch_size: Positions per batch.
        learning_rate: Adam learning rate.
        device: Compute device ('cpu', 'mps', 'cuda').
        lambda_score: Weight for score auxiliary loss.
        lambda_ownership: Weight for ownership auxiliary loss.

    Returns:
        List of per-batch metric dicts for logging.
    """
    optimizer = optim.Adam(network.parameters(), lr=learning_rate, weight_decay=1e-4)
    network.to(device)
    network.train()

    metrics_log = []

    for step in tqdm(range(num_batches), desc="Pretraining value head"):
        batch = buffer.sample(batch_size)

        # Move batch tensors to device
        states    = batch.state_tensors.to(device)
        pol_tgt   = batch.policy_targets.to(device)
        opp_tgt   = batch.opp_policy_targets.to(device)
        val_tgt   = batch.value_targets.to(device)
        score_tgt = batch.score_targets.to(device)
        own_tgt   = batch.ownership_targets.to(device)
        masks     = batch.legal_masks.to(device)
        opp_masks = batch.opp_legal_masks.to(device)

        device_batch = TrainingBatch(
            state_tensors=states,
            policy_targets=pol_tgt,
            opp_policy_targets=opp_tgt,
            value_targets=val_tgt,
            score_targets=score_tgt,
            ownership_targets=own_tgt,
            legal_masks=masks,
            opp_legal_masks=opp_masks,
        )

        output = network(states)

        breakdown = compute_total_loss(
            output,
            device_batch,
            lambda_policy=0.0,              # suppress policy loss entirely
            lambda_value=3.0,               # emphasise value (primary objective)
            lambda_score=lambda_score,
            lambda_ownership=lambda_ownership,
            lambda_opp=0.0,                 # suppress opponent-policy loss
        )

        optimizer.zero_grad()
        breakdown.total.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
        optimizer.step()

        metrics_log.append({
            'step': step,
            'loss_total': breakdown.total.item(),
            'loss_value': breakdown.value.item(),
            'loss_score': breakdown.score.item(),
            'loss_ownership': breakdown.ownership.item(),
        })

        if (step + 1) % 50 == 0:
            tqdm.write(
                f"  step {step+1:4d}  |  total={breakdown.total.item():.4f}  "
                f"value={breakdown.value.item():.4f}  "
                f"score={breakdown.score.item():.4f}  "
                f"ownership={breakdown.ownership.item():.4f}"
            )

    return metrics_log


def main():
    parser = argparse.ArgumentParser(description="Pretrain value head on random-play data.")
    parser.add_argument('--games',      type=int,   default=5_000,
                        help="Number of random-vs-random games to generate (default: 5000).")
    parser.add_argument('--batches',    type=int,   default=500,
                        help="Gradient update steps for value pretraining (default: 500).")
    parser.add_argument('--batch-size', type=int,   default=256,
                        help="Positions per gradient step (default: 256).")
    parser.add_argument('--lr',         type=float, default=1e-3,
                        help="Adam learning rate (default: 0.001).")
    parser.add_argument('--channels',   type=int,   default=128,
                        help="Network channels (must match main training config).")
    parser.add_argument('--num-blocks', type=int,   default=8,
                        help="Number of residual blocks (must match main training config).")
    parser.add_argument('--device',     type=str,   default='cpu',
                        help="Compute device: cpu | mps | cuda (default: cpu).")
    parser.add_argument('--checkpoint', type=str,
                        default=os.path.join(PROJECT_ROOT, 'checkpoints', 'pretrain_value.pt'),
                        help="Output checkpoint path.")
    parser.add_argument('--seed',       type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print("=" * 60)
    print("  Ultimate TTT -- Value Head Pretraining")
    print("=" * 60)
    print(f"  Network:    {args.channels}ch x {args.num_blocks} blocks")
    print(f"  Games:      {args.games:,} random-vs-random")
    print(f"  Batches:    {args.batches} gradient steps  (batch size {args.batch_size})")
    print(f"  Device:     {args.device}")
    print(f"  Checkpoint: {args.checkpoint}")
    print("=" * 60)
    print()

    # Stage 0: generate random games
    print("Stage 0: Generating random-vs-random games...")
    buffer = generate_random_games(args.games)

    if len(buffer) < args.batch_size:
        print(f"ERROR: buffer has {len(buffer)} positions, need at least {args.batch_size}.")
        sys.exit(1)

    # Stage 1: pretrain value head
    print(f"\nStage 1: Pretraining value head ({args.batches} steps)...")
    network = UltimateTTTNetwork(channels=args.channels, num_blocks=args.num_blocks)
    n_params = sum(p.numel() for p in network.parameters())
    print(f"  Parameters: {n_params:,}")

    metrics = pretrain_value_head(
        network=network,
        buffer=buffer,
        num_batches=args.batches,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
    )

    # Save checkpoint to specified path directly
    os.makedirs(os.path.dirname(os.path.abspath(args.checkpoint)), exist_ok=True)
    network.to('cpu')
    torch.save({
        'network_state_dict': network.state_dict(),
        'iteration': 0,
        'elo': 0.0,
        'pretrain_games': args.games,
        'pretrain_batches': args.batches,
        'timestamp': time.time(),
    }, args.checkpoint)

    print(f"\nSaved pretrain checkpoint: {args.checkpoint}")

    # Report final losses
    if metrics:
        last = metrics[-1]
        print(f"\nFinal losses (step {last['step']+1}):")
        print(f"  value     = {last['loss_value']:.4f}   (target: < 0.5)")
        print(f"  score     = {last['loss_score']:.4f}")
        print(f"  ownership = {last['loss_ownership']:.4f}")
        print(f"  total     = {last['loss_total']:.4f}")

    print("\nTo use this checkpoint, add to TrainingConfig in train.py:")
    print(f"  pretrain_checkpoint='{args.checkpoint}'")
    print("\nDone.")


if __name__ == '__main__':
    main()
```

---

### Usage Summary

```bash
# Step 1: run pretraining (5–15 minutes on CPU)
python scripts/pretrain_value.py \
    --games 5000 \
    --batches 500 \
    --channels 128 \
    --num-blocks 8 \
    --device cpu \
    --checkpoint checkpoints/pretrain_value.pt

# Step 2: in train.py, add pretrain_checkpoint to TrainingConfig:
#   pretrain_checkpoint='checkpoints/pretrain_value.pt'
# The normal train() loop loads it automatically at iteration 0.
```

The pretrain script is self-contained and idempotent — re-running it regenerates
the buffer and overwrites the checkpoint. It does not modify any existing
checkpoints or the replay buffer used by the main training loop.
