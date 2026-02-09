"""Helper utilities shared across local speaker metrics."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List, Set


def normalize_text(value: str | None) -> str:
    """Unicode-normalize and trim textual input."""

    if value is None:
        return ""
    return unicodedata.normalize("NFKC", value).strip()


def normalize_name(name: str | None) -> str:
    return normalize_text(name).casefold()


def names_equal(lhs: str | None, rhs: str | None) -> bool:
    return normalize_name(lhs) == normalize_name(rhs)


def contains_direct_mention(
    utterance: str | None,
    speaker_name: str,
    mention_prefixes: Iterable[str] | None = None,
    min_partial_token_length: int = 2,
) -> bool:
    """Return True if `speaker_name` (or its variants) appears in `utterance`."""

    if not utterance:
        return False

    normalized_utterance = normalize_text(utterance).casefold()
    normalized_name = normalize_name(speaker_name)
    if not normalized_name:
        return False

    prefixes = list(mention_prefixes or ("@", "#"))
    for prefix in prefixes:
        candidate = f"{normalize_text(prefix)}{normalized_name}"
        if candidate and candidate in normalized_utterance:
            return True

    variants = _name_variants(normalized_name, min_partial_token_length)
    for variant in variants:
        boundary_pattern = re.compile(
            rf"(?<![\w@#]){re.escape(variant)}(?![\w@#])"
        )
        if boundary_pattern.search(normalized_utterance):
            return True
    return False


def _name_variants(name: str, min_partial_token_length: int) -> Set[str]:
    variants: Set[str] = set()
    stripped = normalize_text(name).casefold()
    if not stripped:
        return variants

    variants.add(stripped)
    compact = re.sub(r"\s+", "", stripped)
    if compact and compact != stripped:
        variants.add(compact)

    tokens = _extract_name_tokens(stripped)
    for token in tokens:
        if len(token) >= min_partial_token_length or len(stripped) <= min_partial_token_length:
            variants.add(token)
    return variants


def _extract_name_tokens(name: str) -> List[str]:
    if not name:
        return []
    return re.findall(r"[\w]+", name)
