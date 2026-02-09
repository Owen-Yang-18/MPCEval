"""Implementation of the Direct Name Reference metric (Sec. 4.1.1)."""

from __future__ import annotations

from typing import Sequence

from .common import contains_direct_mention
from .types import ContextWindow, TurnRecord


def direct_name_reference_score(
    predicted_speaker: str,
    context: ContextWindow,
    mention_prefixes: Sequence[str] | None = None,
) -> float:
    """Return 1.0 if the speaker is explicitly mentioned in the window."""

    if not predicted_speaker or not context:
        return 0.0

    for turn in context:
        if contains_direct_mention(turn.utterance, predicted_speaker, mention_prefixes):
            return 1.0
    return 0.0


__all__ = ["direct_name_reference_score"]
