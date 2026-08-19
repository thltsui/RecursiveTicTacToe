# ADR-0001: Explicit W/D/L value semantics and strong-play reanalysis

- Status: Accepted
- Date: 2026-08-19
- Owners: training, MCTS, evaluation, and web deployment

## Context

The old system represented a position with one scalar `win_value` and converted
it to two displayed percentages with `(1 + value) / 2`. That representation had
four incompatible meanings in different parts of the system:

1. wins and losses were usually labelled `+1` and `-1`, but some self-play
   paths labelled a draw `-0.5`;
2. negamax MCTS changes player perspective by negating value, so a draw must be
   its own inverse (`0`), while `-0.5` becomes a false opponent advantage;
3. backup multiplied values by `0.85` at every ply even though the terminal
   result of a finite board game is undiscounted; and
4. the web UI forced every non-human percentage into the AI percentage, so a
   genuine draw estimate was displayed as an AI win.

The training distribution also mixed different questions. A position completed
by a random opponent was labelled with the result of that random continuation,
while MCTS needs an estimate under strong continuation. Those labels are not
wrong observations, but they estimate a different value function. Mixing them
makes novel positions especially unreliable and can teach the value head that a
state is winning or losing merely because one continuation policy is weak.

The deployment symptom was a confidently negative Q-value and a correspondingly
high displayed AI win percentage even in positions where a human had good
chances. A larger search cannot repair a systematically biased leaf evaluator;
more simulations may simply repeat and reinforce that evaluator's error.

## Decision

### 1. One outcome contract everywhere

Every outcome is from the player-to-move perspective and uses the stable class
order:

| Class | Index | Scalar expectation |
|---|---:|---:|
| Win | 0 | `+1` |
| Draw | 1 | `0` |
| Loss | 2 | `-1` |

The network predicts three logits and is trained with categorical
cross-entropy. Its scalar search value is derived, never learned separately:

`V(s) = P(win | s) - P(loss | s)`

This is a zero-sum expectation. It is not a win probability. A draw can have
large probability while `V(s)` remains close to zero.

Historical replay records are migrated as follows: `+1 -> win`, `-1 -> loss`,
and both `0` and `-0.5 -> draw`. Migrated scalar draw targets are normalized to
zero. Historical one-output checkpoints are expanded into symmetric win/loss
logits with a neutral draw logit so they remain loadable, but they are explicitly
marked `legacy_migrated_uncalibrated`. Their old optimizer state is discarded
because its moment tensors have incompatible shapes.

### 2. Negamax MCTS preserves that contract

On every change of player perspective, MCTS performs exactly these transforms:

- scalar: `V_parent = -V_child`;
- categorical: `[win, draw, loss]_parent = [loss, draw, win]_child`;
- no temporal discount (`gamma = 1`).

MCTS stores an average W/D/L vector as well as the scalar Q-value on every edge.
The web estimate uses the W/D/L vector for the most-visited recommended move.
It does not average all root edges, because search intentionally spends some
visits evaluating inferior moves and that mixture is not the expected outcome
of the move the player will choose.

Single-sided ownership probabilities are not backed up through the tree.
`1 - P(current player owns a sub-board)` is not the opponent ownership
probability when the sub-board can draw. Ownership remains an auxiliary raw
network explanation only.

### 3. Separate win, draw, and loss in the product

The API returns `wdl_probs` and retains `win_value` only for backward
compatibility. The UI displays human win, draw, and AI win separately. Its
balance bar uses expected score, `P(win) + 0.5 * P(draw)`, rather than pretending
a draw belongs to either player.

The UI also states whether the estimate is:

- `Uncalibrated`: a W/D/L head without post-hoc calibration;
- `Legacy estimate`: a migrated scalar checkpoint; or
- `Temp-scaled`: a W/D/L head whose temperature was fitted on a held-out set.

Temperature scaling minimizes held-out categorical negative log likelihood and
is persisted as part of the value head. We report classwise expected calibration
error and multiclass Brier score. “Temp-scaled” is intentionally narrower than
claiming perfect calibration: MCTS changes the evaluated state distribution and
aggregates many leaf estimates, so search-level reliability must still be
measured on positions representative of the web app.

Any subsequent gradient update resets the temperature to one and clears the
calibration marker, because calibration fitted to the previous logits is no
longer valid.

### 4. Diversity states do not create their own labels

Random moves may be used to reach underrepresented, legal, non-terminal states.
From that point, both sides are adjudicated with no-noise, greedy MCTS using the
normal strong simulation budget. Only that strong continuation is recorded.

This same `play_reanalyzed_game` entry point can accept future human-game states.
The random prefix or human move history chooses *where to evaluate*; it never
chooses the value label. The default mixed batch therefore contains:

- ordinary self-play;
- random-prefix states followed by strong two-sided reanalysis; and
- current-network versus frozen-best games.

The old `num_vs_random` setting is accepted only as a deprecated alias for the
new strong-reanalysis count, so an old configuration cannot silently restore
random-opponent targets. Random-vs-random pretraining is disabled by default and
kept solely as an explicitly acknowledged legacy representation experiment.

## Consequences

Benefits:

- a forced draw can no longer become an apparent AI advantage through target,
  backup, or presentation arithmetic;
- Q-value perspective transforms are testable invariants;
- the product can distinguish “likely draw” from “50/50 decisive game”;
- novel regions receive training coverage without assigning random-policy
  outcomes to the strong-play value function; and
- calibration quality becomes measurable rather than inferred from raw logits.

Costs and limitations:

- the final value layer changes from one output to three, so meaningful W/D/L
  quality requires new self-play training even though old checkpoints load;
- a single played result is still a noisy one-hot sample of the outcome
  distribution, so balanced held-out evaluation is required;
- temperature scaling corrects confidence, not ranking or blind spots; and
- random-prefix reanalysis consumes MCTS time comparable to self-play.

## Rollout and validation

This change only modifies code and documentation; it does not start training or
calibration.

Before deploying a new checkpoint:

1. start a fresh W/D/L replay buffer, or allow legacy loading only through the
   documented normalization;
2. train using the updated mixed batch and W/D/L cross-entropy;
3. reserve a held-out set representative of deployment, containing self-play,
   random-prefix, and human positions with enough examples of every outcome;
4. fit one temperature on that held-out set and compare negative log likelihood,
   classwise ECE, and Brier score before and after scaling;
5. stratify those metrics by game phase and search budget, especially the
   100-simulation analysis used by the web app; and
6. deploy only a native three-output checkpoint with the persisted calibration
   marker. A migrated legacy checkpoint may run for compatibility, but its
   percentages must remain labelled as a legacy estimate.

Automated invariants cover draw normalization, player-perspective swapping,
undiscounted multi-ply backup, W/D/L probability normalization, legacy-head
migration, and calibration metrics.

## Out of scope

Inference throughput work—batched leaf evaluation, caching, a smaller distilled
deployment model, quantization, and changing Fly machine resources—is a separate
decision. Those changes can increase the simulation budget, but should be
evaluated only after the value semantics in this ADR are stable.
