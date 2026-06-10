"""Episode 20 — Reading the Training Metrics — How to Know Your AI Is Learning

Concept taught: What each metric means and how to interpret it.
What healthy training curves look like vs warning signs.

Expected ranges:
    Random policy entropy: log(81) ~ 4.4
    Well-trained policy entropy: 1.0 - 2.0
    Value accuracy baseline: ~0.5 (random)
    Good value accuracy: > 0.7
"""

from __future__ import annotations

import json
import os

import torch
import numpy as np


def compute_policy_entropy(policy_probs: torch.Tensor) -> float:
    """Compute entropy of policy distribution.

    H = -sum p(a) * log(p(a))

    High entropy (early training): network is uncertain, explores broadly.
    Low entropy (late training): network is decisive, plays with conviction.

    Args:
        policy_probs: shape (81,) or (B, 81), probability distribution.

    Returns:
        Scalar entropy value (or mean entropy over batch).
    """
    if policy_probs.dim() == 1:
        policy_probs = policy_probs.unsqueeze(0)  # (1, 81)

    # Clamp to avoid log(0)
    p = policy_probs.clamp(min=1e-8)
    entropy = -(p * p.log()).sum(dim=-1)  # (B,)
    return entropy.mean().item()


def compute_value_accuracy(
    value_preds: torch.Tensor,
    value_targets: torch.Tensor,
) -> float:
    """Fraction of positions where sign(prediction) == sign(target).

    Measures whether the network correctly predicts who is winning,
    even if the exact probability is off.

    Args:
        value_preds: (B, 1) predicted win values in [-1, 1].
        value_targets: (B, 1) actual outcomes in {-1, 0, 1}.

    Returns:
        Accuracy in [0, 1]. Random baseline is ~0.5.
    """
    # For draws (target=0), count as correct if prediction is near 0
    pred_sign = torch.sign(value_preds)
    target_sign = torch.sign(value_targets)

    # Handle draws: target_sign == 0 is correct if pred is within [-0.5, 0.5]
    correct = (pred_sign == target_sign).float()

    # For draw positions, be more lenient
    draw_mask = (value_targets == 0).float()
    near_zero = (value_preds.abs() < 0.5).float()
    correct = correct * (1 - draw_mask) + near_zero * draw_mask

    return correct.mean().item()


class MetricsLogger:
    """Logs and persists training metrics to JSON for later analysis.

    Maintains running averages of loss components within an iteration.
    Saves full history to JSON file for plotting.
    """

    def __init__(self, log_path: str = 'training_metrics.json'):
        self.log_path = log_path
        self._history: list[dict] = []

        # Load existing history if present
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r') as f:
                    content = f.read().strip()
                    if content:
                        self._history = json.loads(content)
            except (json.JSONDecodeError, IOError):
                pass

    def log_iteration(self, iteration: int, metrics: dict) -> None:
        """Log all metrics for one training iteration. Appends to JSON.

        Args:
            iteration: Iteration number.
            metrics: Dict of metric name -> value.
        """
        entry = {'iteration': iteration}
        entry.update(metrics)
        self._history.append(entry)

        with open(self.log_path, 'w') as f:
            json.dump(self._history, f, indent=2)

    def plot_training_curves(self, save_path: str = 'training_curves.png') -> None:
        """Generate a 6-panel matplotlib figure.

        Panel 1: All 5 loss components over iterations (log scale y-axis).
        Panel 2: Elo rating over iterations.
        Panel 3: Average game length over iterations.
        Panel 4: Policy entropy over iterations.
        Panel 5: Value head accuracy over iterations.
        Panel 6: Arena win rate over iterations (scatter, only when available).
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if not self._history:
            print("No metrics to plot")
            return

        iters = [h['iteration'] for h in self._history]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # Panel 1: Loss components
        ax = axes[0, 0]
        for key, label in [('loss_policy', 'Policy'), ('loss_value', 'Value'),
                           ('loss_score', 'Score'), ('loss_ownership', 'Ownership'),
                           ('loss_opp_policy', 'Opp Policy')]:
            vals = [h.get(key) for h in self._history]
            if any(v is not None for v in vals):
                valid_iters = [i for i, v in zip(iters, vals) if v is not None]
                valid_vals = [v for v in vals if v is not None]
                ax.semilogy(valid_iters, valid_vals, label=label)
        ax.set_title("Loss Components")
        ax.legend()
        ax.grid(True)

        # Panel 2: Elo
        ax = axes[0, 1]
        elo_vals = [h.get('elo') for h in self._history]
        if any(v is not None for v in elo_vals):
            valid = [(i, v) for i, v in zip(iters, elo_vals) if v is not None]
            ax.plot([x[0] for x in valid], [x[1] for x in valid])
        ax.set_title("Elo Rating")
        ax.grid(True)

        # Panel 3: Game length
        ax = axes[0, 2]
        gl_vals = [h.get('avg_game_length') for h in self._history]
        if any(v is not None for v in gl_vals):
            valid = [(i, v) for i, v in zip(iters, gl_vals) if v is not None]
            ax.plot([x[0] for x in valid], [x[1] for x in valid])
        ax.set_title("Avg Game Length")
        ax.grid(True)

        # Panel 4: Policy entropy
        ax = axes[1, 0]
        pe_vals = [h.get('policy_entropy') for h in self._history]
        if any(v is not None for v in pe_vals):
            valid = [(i, v) for i, v in zip(iters, pe_vals) if v is not None]
            ax.plot([x[0] for x in valid], [x[1] for x in valid])
        ax.set_title("Policy Entropy")
        ax.grid(True)

        # Panel 5: Value accuracy
        ax = axes[1, 1]
        va_vals = [h.get('value_accuracy') for h in self._history]
        if any(v is not None for v in va_vals):
            valid = [(i, v) for i, v in zip(iters, va_vals) if v is not None]
            ax.plot([x[0] for x in valid], [x[1] for x in valid])
        ax.set_title("Value Accuracy")
        ax.grid(True)

        # Panel 6: Arena win rate (scatter)
        ax = axes[1, 2]
        wr_vals = [h.get('arena_win_rate') for h in self._history]
        if any(v is not None for v in wr_vals):
            valid = [(i, v) for i, v in zip(iters, wr_vals) if v is not None]
            ax.scatter([x[0] for x in valid], [x[1] for x in valid], c='green', s=50)
            ax.axhline(0.55, color='red', linestyle='--', label='Threshold')
            ax.legend()
        ax.set_title("Arena Win Rate")
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.close(fig)
        print(f"Training curves saved to {save_path}")

    def get_summary(self) -> dict:
        """Return summary statistics of the most recent 10 iterations.

        Returns:
            Dict with mean values of key metrics over the last 10 iterations.
        """
        recent = self._history[-10:]
        if not recent:
            return {}

        summary = {}
        keys = ['loss_total', 'loss_policy', 'loss_value', 'loss_score',
                'loss_ownership', 'loss_opp_policy', 'elo', 'policy_entropy',
                'value_accuracy', 'arena_win_rate']

        for key in keys:
            vals = [h.get(key) for h in recent if h.get(key) is not None]
            if vals:
                summary[f'{key}_mean'] = sum(vals) / len(vals)
                summary[f'{key}_latest'] = vals[-1]

        return summary


if __name__ == "__main__":
    print("=== Episode 20: Metrics ===\n")

    # Test policy entropy
    uniform = torch.ones(81) / 81.0
    h_uniform = compute_policy_entropy(uniform)
    print(f"Uniform entropy: {h_uniform:.3f} (expected ~{np.log(81):.3f})")
    assert abs(h_uniform - np.log(81)) < 0.01

    peaked = torch.zeros(81)
    peaked[0] = 1.0
    h_peaked = compute_policy_entropy(peaked)
    print(f"Peaked entropy: {h_peaked:.6f} (expected ~0)")
    assert h_peaked < 0.01

    # Test batch entropy
    batch = torch.softmax(torch.randn(8, 81), dim=-1)
    h_batch = compute_policy_entropy(batch)
    print(f"Batch entropy (B=8): {h_batch:.3f}")

    # Test value accuracy
    preds = torch.tensor([[0.8], [-0.5], [0.1], [-0.9]])
    targets = torch.tensor([[1.0], [-1.0], [0.0], [-1.0]])
    acc = compute_value_accuracy(preds, targets)
    print(f"\nValue accuracy: {acc:.3f}")
    assert acc > 0.5

    # Test MetricsLogger
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        tmp_path = f.name

    logger = MetricsLogger(log_path=tmp_path)
    for i in range(5):
        logger.log_iteration(i, {
            'loss_total': 5.0 - i * 0.5,
            'loss_policy': 4.0 - i * 0.3,
            'loss_value': 0.8 - i * 0.1,
        })

    summary = logger.get_summary()
    print(f"\nSummary: {summary}")
    assert 'loss_total_mean' in summary

    os.unlink(tmp_path)
    print("MetricsLogger: PASSED")

    print("\n=== Episode 20 PASSED ===")
