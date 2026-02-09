"""Implicit Reference metric (Sec. 4.1.2)."""

from __future__ import annotations

import math
from typing import Callable

from .common import names_equal
from .types import ContextWindow

DecayFunction = Callable[[int], float]


def geometric_decay(lmbda: float = 0.6) -> DecayFunction:
    if not 0.0 < lmbda <= 1.0:
        raise ValueError("lambda must be in (0, 1]")

    def _decay(distance: int) -> float:
        return float(lmbda * ((1 - lmbda) ** distance))

    return _decay


def exponential_decay(alpha: float = 1.0) -> DecayFunction:
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    def _decay(distance: int) -> float:
        return float(math.exp(-alpha * distance))

    return _decay


def inverse_distance_decay() -> DecayFunction:
    def _decay(distance: int) -> float:
        return 1.0 / (1.0 + distance)

    return _decay


def implicit_reference_score(
    predicted_speaker: str,
    context: ContextWindow,
    decay_fn: DecayFunction | None = None,
) -> float:
    """Score the likelihood that the speaker re-enters via turn-taking."""

    if not predicted_speaker or len(context) < 2:
        return 0.0

    decay_fn = decay_fn or geometric_decay()
    best = 0.0
    prior_turns = context[:-1]
    for distance, turn in enumerate(reversed(prior_turns)):
        if names_equal(turn.speaker, predicted_speaker):
            best = max(best, decay_fn(distance))
    return float(best)


__all__ = [
    "implicit_reference_score",
    "geometric_decay",
    "exponential_decay",
    "inverse_distance_decay",
]
