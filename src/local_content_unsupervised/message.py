"""Message-level semantic novelty for local unsupervised content flow (Sec. 4.2.2).

Implements Message-Level Semantic Novelty (M-SNS) using sentence embeddings:

Eq. (17) (minimum distance):
    Flowing_M-SNS = min_{c ∈ C^{-k}} (1 - cos(e(u_p), e(c)))

Eq. (18) (average distance):
    Nov_sem^avg = (1 / |C^{-k}|) * Σ_{c ∈ C^{-k}} (1 - cos(e(u_p), e(c)))

We apply punctuation removal only (no stopword removal).
"""

from __future__ import annotations

import math
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np

Array = np.ndarray


def remove_punctuation(text: str) -> str:
    """Remove Unicode punctuation characters and collapse whitespace."""

    if not text:
        return ""
    kept: list[str] = []
    for ch in text:
        if unicodedata.category(ch).startswith("P"):
            continue
        kept.append(ch)
    cleaned = "".join(kept)
    # Collapse whitespace
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()


class SentenceEmbedder:
    """Sentence-transformers backed embedder."""

    def __init__(self, model_name: str, *, batch_size: int = 64) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is required for message-level semantic novelty. "
                "Install via `pip install sentence-transformers`."
            ) from exc

        self.model_name = model_name
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> Array:
        if not texts:
            dim = int(self._model.get_sentence_embedding_dimension())
            return np.zeros((0, dim), dtype=float)
        arr = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        mat = np.asarray(arr, dtype=float)
        if mat.ndim == 1:
            mat = mat.reshape(1, -1)
        return mat


@dataclass
class CachedSentenceEmbedder:
    """LRU cached sentence embedder keyed by normalized message string."""

    embedder: SentenceEmbedder
    max_cache_items: int = 20_000

    def __post_init__(self) -> None:
        if self.max_cache_items <= 0:
            raise ValueError("max_cache_items must be > 0")
        self._cache: "OrderedDict[str, Array]" = OrderedDict()
        self._hits = 0
        self._misses = 0

    @property
    def cache_info(self) -> Dict[str, int]:
        return {
            "max_items": int(self.max_cache_items),
            "items": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
        }

    def encode(self, texts: Sequence[str]) -> Array:
        if not texts:
            return self.embedder.encode(texts)

        normalized = list(texts)
        unique: list[str] = []
        seen: set[str] = set()
        for t in normalized:
            if t in seen:
                continue
            seen.add(t)
            unique.append(t)

        missing: list[str] = []
        for t in unique:
            if t in self._cache:
                self._hits += 1
                self._cache.move_to_end(t)
            else:
                self._misses += 1
                missing.append(t)

        if missing:
            batch_emb = self.embedder.encode(missing)
            for t, vec in zip(missing, batch_emb):
                self._cache[t] = vec
            while len(self._cache) > self.max_cache_items:
                self._cache.popitem(last=False)

        dim = next(iter(self._cache.values())).shape[0] if self._cache else 0
        out = np.zeros((len(texts), dim), dtype=float)
        for i, t in enumerate(normalized):
            out[i, :] = self._cache[t]
        return out


def message_semantic_novelty(
    context_messages: Sequence[str],
    predicted_message: str,
    *,
    embedder: SentenceEmbedder | CachedSentenceEmbedder,
    k: int | None = None,
    mode: str = "min",
) -> float:
    """Compute Eq. (17) or Eq. (18) for a single (context, u_p) pair."""

    if mode not in {"min", "avg"}:
        raise ValueError("mode must be 'min' or 'avg'")

    up = remove_punctuation(predicted_message)
    if not up:
        return 0.0

    ctx = [remove_punctuation(m) for m in context_messages]
    ctx = [m for m in ctx if m]
    if k is not None and k > 0:
        ctx = ctx[-k:]

    if not ctx:
        return 0.0

    mat = embedder.encode([up, *ctx])
    if mat.shape[0] < 2:
        return 0.0

    u = mat[0]
    c = mat[1:]
    sims = c @ u
    dists = 1.0 - sims
    if dists.size == 0:
        return 0.0
    return float(dists.min() if mode == "min" else dists.mean())

