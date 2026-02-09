"""Embedding-based speaker-content consistency metrics (Sec. 4.3.1)."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from src.local_speaker.utils.common import names_equal, normalize_text
from src.local_speaker.utils.embedding_similarity import (
    SentenceTransformerEmbedder,
    TfidfEmbedder,
)
from src.local_speaker.utils.types import ContextWindow, TurnRecord

Array = np.ndarray
EncodeFn = Callable[[Sequence[str]], Array]


def _cosine_similarities(vec: Array, mat: Array, eps: float = 1e-12) -> Array:
    if mat.size == 0 or vec.size == 0:
        return np.zeros((0,), dtype=float)

    vec = vec.reshape(-1)
    vec_norm = float(np.linalg.norm(vec) + eps)
    mat_norm = np.linalg.norm(mat, axis=1) + eps
    return np.clip((mat @ vec) / (mat_norm * vec_norm), -1.0, 1.0)


def _resolve_encode_fn(
    embedder: EncodeFn | object,
) -> EncodeFn:
    if isinstance(embedder, (TfidfEmbedder, SentenceTransformerEmbedder)):
        return embedder.encode
    if hasattr(embedder, "encode") and callable(getattr(embedder, "encode")):
        return getattr(embedder, "encode")
    if callable(embedder):
        return embedder  # type: ignore[return-value]
    raise TypeError(f"Unsupported embedder type: {type(embedder)!r}")


@dataclass
class CachedTextEmbedder:
    """A lightweight LRU cache wrapper around an embedding `encode(texts)` function."""

    encode_fn: EncodeFn
    max_cache_items: int = 20000

    def __post_init__(self) -> None:
        self._cache: OrderedDict[str, Array] = OrderedDict()

    def encode(self, texts: Sequence[str]) -> Array:
        if not texts:
            return np.zeros((0, 0), dtype=float)

        normalized = [normalize_text(text) for text in texts]
        cached_vectors: list[Array | None] = []
        missing: list[str] = []
        missing_positions: list[int] = []

        for idx, text in enumerate(normalized):
            if text in self._cache:
                vec = self._cache.pop(text)
                self._cache[text] = vec
                cached_vectors.append(vec)
            else:
                cached_vectors.append(None)
                missing.append(text)
                missing_positions.append(idx)

        if missing:
            missing_matrix = np.asarray(self.encode_fn(missing), dtype=float)
            if missing_matrix.ndim == 1:
                missing_matrix = missing_matrix.reshape(1, -1)

            for pos, text, vec in zip(missing_positions, missing, missing_matrix):
                vec1d = np.asarray(vec, dtype=float).reshape(-1)
                cached_vectors[pos] = vec1d
                self._cache[text] = vec1d
                while len(self._cache) > self.max_cache_items:
                    self._cache.popitem(last=False)

        matrix = np.stack([vec for vec in cached_vectors if vec is not None], axis=0)
        return np.asarray(matrix, dtype=float)


def _speaker_history_texts(
    predicted_speaker: str,
    context: ContextWindow,
    window_size: int,
) -> list[str]:
    if not predicted_speaker or not context:
        return []
    window = list(context[-window_size:]) if window_size > 0 else list(context)
    return [turn.utterance for turn in window if names_equal(turn.speaker, predicted_speaker) and turn.utterance]


def _embed_pair(
    predicted_message: str,
    speaker_texts: Sequence[str],
    *,
    embedder: EncodeFn | object,
) -> tuple[Array, Array]:
    predicted_message = normalize_text(predicted_message)
    if not predicted_message or not speaker_texts:
        return np.zeros((0, 0), dtype=float), np.zeros((0,), dtype=float)

    encode_fn = _resolve_encode_fn(embedder)
    texts = list(speaker_texts) + [predicted_message]
    matrix = np.asarray(encode_fn(texts), dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)

    speaker_matrix = matrix[:-1]
    predicted_vec = matrix[-1].reshape(-1)
    return speaker_matrix, predicted_vec


def scc_avg(
    predicted_message: str,
    predicted_speaker: str,
    context: ContextWindow,
    *,
    window_size: int = 5,
    embedder: EncodeFn | object,
) -> float | None:
    """Eq. (22): mean_c sim(e(u_p), e(c)) over c in C_sp."""

    speaker_texts = _speaker_history_texts(predicted_speaker, context, window_size)
    if not speaker_texts or not normalize_text(predicted_message):
        return None
    speaker_matrix, predicted_vec = _embed_pair(
        predicted_message, speaker_texts, embedder=embedder
    )
    if speaker_matrix.size == 0:
        return None
    sims = _cosine_similarities(predicted_vec, speaker_matrix)
    return float(sims.mean()) if sims.size else None


def scc_max(
    predicted_message: str,
    predicted_speaker: str,
    context: ContextWindow,
    *,
    window_size: int = 5,
    embedder: EncodeFn | object,
) -> float | None:
    """Eq. (23): max_c sim(e(u_p), e(c)) over c in C_sp."""

    speaker_texts = _speaker_history_texts(predicted_speaker, context, window_size)
    if not speaker_texts or not normalize_text(predicted_message):
        return None
    speaker_matrix, predicted_vec = _embed_pair(
        predicted_message, speaker_texts, embedder=embedder
    )
    if speaker_matrix.size == 0:
        return None
    sims = _cosine_similarities(predicted_vec, speaker_matrix)
    return float(sims.max()) if sims.size else None


def scc_min(
    predicted_message: str,
    predicted_speaker: str,
    context: ContextWindow,
    *,
    window_size: int = 5,
    embedder: EncodeFn | object,
) -> float | None:
    """Eq. (24): min_c sim(e(u_p), e(c)) over c in C_sp."""

    speaker_texts = _speaker_history_texts(predicted_speaker, context, window_size)
    if not speaker_texts or not normalize_text(predicted_message):
        return None
    speaker_matrix, predicted_vec = _embed_pair(
        predicted_message, speaker_texts, embedder=embedder
    )
    if speaker_matrix.size == 0:
        return None
    sims = _cosine_similarities(predicted_vec, speaker_matrix)
    return float(sims.min()) if sims.size else None


def build_embedder(
    model_name: str,
    *,
    backend: str = "auto",
) -> SentenceTransformerEmbedder | TfidfEmbedder:
    """Build an embedder.

    backend:
      - auto: try SentenceTransformer, fall back to TF-IDF
      - sentence-transformer: require SentenceTransformer
      - tfidf: always use TF-IDF
    """

    backend_norm = (backend or "auto").lower()
    if backend_norm == "tfidf":
        return TfidfEmbedder()

    if backend_norm not in {"auto", "sentence-transformer"}:
        raise ValueError(f"Unsupported embedding backend: {backend!r}")

    try:
        return SentenceTransformerEmbedder(model_name=model_name)
    except Exception:
        if backend_norm == "sentence-transformer":
            raise
        return TfidfEmbedder()


def fit_tfidf_on_pairs(
    embedder: TfidfEmbedder,
    *,
    pairs: Iterable[Mapping[str, object]],
    window_size: int,
    max_pairs: int = 2000,
) -> None:
    """Fit a TF-IDF embedder on a sample of the pairs file for stability."""

    corpus: list[str] = []
    seen = 0
    for record in pairs:
        context = record.get("context")
        if isinstance(context, list):
            window = context[-window_size:] if window_size > 0 else context
            for turn in window:
                if isinstance(turn, dict):
                    corpus.append(str(turn.get("utterance", "")).strip())
        corpus.append(str(record.get("next_message", "")).strip())
        seen += 1
        if seen >= max_pairs:
            break
    embedder.fit(corpus)


__all__ = [
    "CachedTextEmbedder",
    "build_embedder",
    "fit_tfidf_on_pairs",
    "scc_avg",
    "scc_max",
    "scc_min",
    "TurnRecord",
]
