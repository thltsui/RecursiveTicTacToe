"""Plot comparable DQN training logs produced by run_experiment.sh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_curve(path: Path) -> tuple[list[int], list[float], str]:
    record = json.loads(path.read_text())
    episodes = [row["episode"] for row in record["results"]]
    win_rates = [100.0 * row["win_rate"] for row in record["results"]]
    learning_rate = record["config"]["learning_rate"]
    return episodes, win_rates, f"Adam learning rate {learning_rate:g}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for log_path in args.logs:
        episodes, win_rates, label = load_curve(log_path)
        ax.plot(episodes, win_rates, marker="o", markersize=3, linewidth=2, label=label)

    ax.axhline(50, color="#777777", linewidth=1, linestyle="--", label="50% reference")
    ax.set_title("DQN performance against a random opponent")
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("Win rate over 300 evaluation games (%)")
    ax.set_xlim(left=0)
    ax.set_ylim(25, 70)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
