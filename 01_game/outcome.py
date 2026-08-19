"""Canonical win/draw/loss semantics shared by training, search, and UI.

The class order is intentionally stable because it is persisted in replay buffers
and learned checkpoints.  Values are always from the player-to-move's
perspective.
"""

from __future__ import annotations


WDL_WIN = 0
WDL_DRAW = 1
WDL_LOSS = 2
WDL_LABELS = ("win", "draw", "loss")


def outcome_class(winner: int, perspective_player: int) -> int:
    """Return the W/D/L class for ``perspective_player``.

    Args:
        winner: Raw game winner: ``1``, ``-1``, or ``0`` for draw.
        perspective_player: Raw player identity, ``1`` or ``-1``.
    """
    if winner not in (-1, 0, 1):
        raise ValueError(f"winner must be -1, 0, or 1; got {winner}")
    if perspective_player not in (-1, 1):
        raise ValueError(
            f"perspective_player must be -1 or 1; got {perspective_player}"
        )
    if winner == 0:
        return WDL_DRAW
    return WDL_WIN if winner == perspective_player else WDL_LOSS


def outcome_value(winner: int, perspective_player: int) -> float:
    """Return the zero-sum scalar value ``+1 / 0 / -1``."""
    result = outcome_class(winner, perspective_player)
    if result == WDL_WIN:
        return 1.0
    if result == WDL_LOSS:
        return -1.0
    return 0.0


def class_value(wdl_class: int) -> float:
    """Return the scalar expectation for a hard W/D/L class."""
    if wdl_class == WDL_WIN:
        return 1.0
    if wdl_class == WDL_DRAW:
        return 0.0
    if wdl_class == WDL_LOSS:
        return -1.0
    raise ValueError(f"wdl_class must be 0, 1, or 2; got {wdl_class}")


def legacy_value_to_class(value: float) -> int:
    """Migrate old replay targets to W/D/L classes.

    Historical buffers used both ``0`` and ``-0.5`` for draws, while losses
    were exactly ``-1``.  The thresholds preserve those old draws rather than
    silently converting them into losses.
    """
    if value >= 0.5:
        return WDL_WIN
    if value <= -0.75:
        return WDL_LOSS
    return WDL_DRAW
