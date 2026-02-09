"""Participation Frequency metric (Sec. 4.1.3)."""

from __future__ import annotations

from .common import names_equal
from .types import ContextWindow


def participation_frequency_score(predicted_speaker: str, context: ContextWindow) -> float:
    """Compute |{u | s = s_p}| / k for the provided window."""

    if not predicted_speaker or not context:
        return 0.0

    match_count = sum(1 for turn in context if names_equal(turn.speaker, predicted_speaker))
    return match_count / len(context)


__all__ = ["participation_frequency_score"]
