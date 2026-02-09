"""Embedding-based speaker alignment (Sec. 4.1.4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

try:  # Optional heavy dependency for neural embeddings.
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore

from .common import names_equal
from .types import ContextWindow

Array = np.ndarray
EmbedderFn = Callable[[Sequence[str]], Array]


@dataclass
class TfidfEmbedder:
    """Simple TF-IDF embedder that can be re-used across calls."""

    max_features: int = 5000
    ngram_range: tuple[int, int] = (1, 2)
    analyzer: str = "word"

    def __post_init__(self) -> None:
        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features, ngram_range=self.ngram_range, analyzer=self.analyzer
        )
        self._is_fitted = False

    def fit(self, corpus: Iterable[str]) -> None:
        texts = [text for text in corpus if text]
        if not texts:
            return
        self._vectorizer.fit(texts)
        self._is_fitted = True

    def encode(self, texts: Sequence[str]) -> Array:
        if not texts:
            if not self._is_fitted:
                return np.zeros((0, 0), dtype=float)
            return np.zeros((0, len(self._vectorizer.get_feature_names_out())), dtype=float)

        if not self._is_fitted:
            self.fit(texts)
        return self._vectorizer.transform(texts).toarray()


class SentenceTransformerEmbedder:
    """SentenceTransformer-backed embedder that loads Hugging Face checkpoints."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str | None = None,
                 normalize: bool = False) -> None:
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEmbedder"
            )

        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self._model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: Sequence[str]) -> Array:
        if not texts:
            return np.zeros((0, self._model.get_sentence_embedding_dimension()), dtype=float)
        embeddings = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        return np.asarray(embeddings, dtype=float)


def average_embedding_similarity_score(
    predicted_speaker: str,
    context: ContextWindow,
    background_text: str | None = None,
    window_size: int | None = None,
    embedder: EmbedderFn | TfidfEmbedder | None = None,
    use_augmentation: bool = True,
) -> float:
    """Compute Eq. (4) with optional augmentation using speaker metadata."""

    matrices = _prepare_embeddings(predicted_speaker, context, background_text, embedder)
    if matrices is None:
        return 0.0

    context_matrix, speaker_matrix, background_vector, speaker_history = matrices
    if context_matrix.size == 0:
        return 0.0

    window = window_size or max(len(context), 1)
    speaker_repr, _ = _speaker_representation(
        speaker_matrix,
        background_vector,
        history_len=speaker_history,
        window_size=window,
        use_augmentation=use_augmentation,
    )
    if speaker_repr is None or speaker_repr.size == 0:
        return 0.0

    sims = _cosine_similarity_matrix(context_matrix, speaker_repr)
    if sims.size == 0:
        return 0.0
    per_context = sims.max(axis=1)
    return float(per_context.mean())


def maximum_embedding_similarity_score(
    predicted_speaker: str,
    context: ContextWindow,
    background_text: str | None = None,
    window_size: int | None = None,
    embedder: EmbedderFn | TfidfEmbedder | None = None,
    use_augmentation: bool = True,
) -> float:
    """Compute Eq. (5) with optional metadata augmentation."""

    matrices = _prepare_embeddings(predicted_speaker, context, background_text, embedder)
    if matrices is None:
        return 0.0

    context_matrix, speaker_matrix, background_vector, speaker_history = matrices
    if context_matrix.size == 0:
        return 0.0

    window = window_size or max(len(context), 1)
    speaker_repr, _ = _speaker_representation(
        speaker_matrix,
        background_vector,
        history_len=speaker_history,
        window_size=window,
        use_augmentation=use_augmentation,
    )
    if speaker_repr is None or speaker_repr.size == 0:
        return 0.0

    sims = _cosine_similarity_matrix(context_matrix, speaker_repr)
    if sims.size == 0:
        return 0.0
    per_context = sims.max(axis=1)
    return float(per_context.max()) if per_context.size else 0.0


def _prepare_embeddings(
    predicted_speaker: str,
    context: ContextWindow,
    background_text: str | None,
    embedder: EmbedderFn | TfidfEmbedder | None,
):
    other_texts: list[str] = []
    speaker_texts: list[str] = []
    for turn in context:
        if not turn.utterance:
            continue
        if names_equal(turn.speaker, predicted_speaker):
            speaker_texts.append(turn.utterance)
        else:
            other_texts.append(turn.utterance)

    if not other_texts:
        return None

    embed_fn = _resolve_embedder(embedder, context, background_text)
    context_matrix, speaker_matrix, background_vector = _embed_texts(
        embed_fn, other_texts, speaker_texts, background_text
    )

    return context_matrix, speaker_matrix, background_vector, len(speaker_texts)


def _resolve_embedder(
    embedder: EmbedderFn | TfidfEmbedder | None,
    context: ContextWindow,
    background_text: str | None,
) -> EmbedderFn:
    if embedder is None:
        default_embedder = _default_embedder(context, background_text)
        if isinstance(default_embedder, (TfidfEmbedder, SentenceTransformerEmbedder)):
            return default_embedder.encode
        return default_embedder

    if isinstance(embedder, (TfidfEmbedder, SentenceTransformerEmbedder)):
        return embedder.encode
    return embedder


def _default_embedder(
    context: ContextWindow,
    background_text: str | None,
) -> EmbedderFn | TfidfEmbedder | SentenceTransformerEmbedder:
    try:
        return SentenceTransformerEmbedder()
    except Exception:
        documents = [turn.utterance for turn in context if turn.utterance]
        if background_text:
            documents.append(background_text)
        tfidf = TfidfEmbedder()
        tfidf.fit(documents)
        return tfidf


def _embed_texts(
    embed_fn: EmbedderFn,
    other_texts: Sequence[str],
    speaker_texts: Sequence[str],
    background_text: str | None,
) -> tuple[Array, Array, Array | None]:
    context_matrix = _encode(embed_fn, other_texts)
    speaker_matrix = _encode(embed_fn, speaker_texts)
    background_vector = _encode(embed_fn, [background_text]) if background_text else None

    dim = 0
    for mat in (context_matrix, speaker_matrix, background_vector):
        if mat.size:
            dim = mat.shape[1]
            break

    context_matrix = _ensure_shape(context_matrix, dim)
    speaker_matrix = _ensure_shape(speaker_matrix, dim)
    if background_vector is not None:
        background_vector = background_vector.reshape(1, -1)
    return context_matrix, speaker_matrix, background_vector


def _encode(embed_fn: EmbedderFn, texts: Sequence[str]) -> Array:
    if not texts:
        return np.zeros((0, 0), dtype=float)
    matrix = np.asarray(embed_fn(list(texts)), dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    return matrix


def _ensure_shape(matrix: Array, dim: int) -> Array:
    if matrix.size:
        return matrix
    if dim == 0:
        return matrix
    return np.zeros((0, dim), dtype=float)


def _speaker_representation(
    speaker_matrix: Array,
    background_vector: Array | None,
    history_len: int,
    window_size: int,
    use_augmentation: bool,
) -> tuple[Array | None, bool]:
    if use_augmentation and background_vector is not None:
        message_avg = (
            speaker_matrix.mean(axis=0) if speaker_matrix.size else np.zeros(background_vector.shape[1])
        )
        alpha = min(1.0, history_len / max(window_size, 1))
        augmented = alpha * message_avg + (1 - alpha) * background_vector.reshape(-1)
        return augmented.reshape(1, -1), True

    if speaker_matrix.size == 0:
        return None, False
    return speaker_matrix, False


def _cosine_similarity_matrix(a: Array, b: Array, eps: float = 1e-12) -> Array:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=float)
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + eps)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + eps)
    return np.clip(a_norm @ b_norm.T, -1.0, 1.0)


__all__ = [
    "average_embedding_similarity_score",
    "maximum_embedding_similarity_score",
    "TfidfEmbedder",
    "SentenceTransformerEmbedder",
]
