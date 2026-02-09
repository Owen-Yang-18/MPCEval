"""Shared dataclasses for local content unsupervised evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence


@dataclass(frozen=True)
class TurnRecord:
    """Simplified representation of a conversation turn."""

    speaker: str
    utterance: str


ContextWindow = Sequence[TurnRecord]


def to_turn_records(raw_turns: Sequence[Dict[str, Any]]) -> List[TurnRecord]:
    """Convert processed MPDD dictionaries into :class:`TurnRecord` objects."""

    records: List[TurnRecord] = []
    for turn in raw_turns:
        speaker = str(turn.get("speaker", "")).strip()
        utterance = str(turn.get("utterance", "")).strip()
        if not speaker or not utterance:
            continue
        records.append(TurnRecord(speaker=speaker, utterance=utterance))
    return records

