"""Calibration metrics and post-hoc temperature fitting for W/D/L outputs.

These utilities operate on a held-out set.  They are never called by the
training loop automatically, which keeps model fitting and calibration data
strictly separated.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def wdl_brier_score(probs: torch.Tensor, targets: torch.Tensor) -> float:
    """Return multiclass Brier score (lower is better)."""
    if probs.ndim != 2 or probs.shape[-1] != 3:
        raise ValueError("probs must have shape (N, 3)")
    if targets.ndim != 1 or targets.shape[0] != probs.shape[0]:
        raise ValueError("targets must have shape (N,)")
    if targets.numel() == 0:
        raise ValueError("calibration set must not be empty")
    one_hot = F.one_hot(targets.long(), num_classes=3).to(probs.dtype)
    return ((probs - one_hot) ** 2).sum(dim=-1).mean().item()


def classwise_expected_calibration_error(
    probs: torch.Tensor,
    targets: torch.Tensor,
    num_bins: int = 10,
) -> dict[str, float]:
    """Compute one-vs-rest ECE independently for win, draw, and loss."""
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    if probs.ndim != 2 or probs.shape[-1] != 3:
        raise ValueError("probs must have shape (N, 3)")
    if targets.ndim != 1 or targets.shape[0] != probs.shape[0]:
        raise ValueError("targets must have shape (N,)")
    if targets.numel() == 0:
        raise ValueError("calibration set must not be empty")

    names = ("win", "draw", "loss")
    edges = torch.linspace(0.0, 1.0, num_bins + 1, device=probs.device)
    result: dict[str, float] = {}
    total = max(int(targets.numel()), 1)

    for class_idx, name in enumerate(names):
        confidence = probs[:, class_idx]
        observed = (targets == class_idx).to(probs.dtype)
        ece = torch.zeros((), dtype=probs.dtype, device=probs.device)
        for bin_idx in range(num_bins):
            lower = edges[bin_idx]
            upper = edges[bin_idx + 1]
            if bin_idx == num_bins - 1:
                in_bin = (confidence >= lower) & (confidence <= upper)
            else:
                in_bin = (confidence >= lower) & (confidence < upper)
            if in_bin.any():
                gap = (confidence[in_bin].mean() - observed[in_bin].mean()).abs()
                ece = ece + gap * (in_bin.sum() / total)
        result[name] = float(ece.item())
    return result


def fit_temperature(
    logits: torch.Tensor,
    targets: torch.Tensor,
    max_iter: int = 50,
) -> float:
    """Fit one positive temperature by held-out negative log likelihood."""
    if logits.ndim != 2 or logits.shape[-1] != 3:
        raise ValueError("logits must have shape (N, 3)")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError("targets must have shape (N,)")
    if targets.numel() == 0:
        raise ValueError("calibration set must not be empty")

    frozen_logits = logits.detach()
    frozen_targets = targets.detach().long()
    log_temperature = torch.zeros((), device=logits.device, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = F.cross_entropy(frozen_logits / temperature, frozen_targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0).item())


def set_model_temperature(network, temperature: float) -> None:
    """Persist a fitted temperature and calibration marker on a network."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    value_head = network.value_head
    with torch.no_grad():
        value_head.wdl_temperature.fill_(float(temperature))
        value_head.wdl_is_calibrated.fill_(True)
