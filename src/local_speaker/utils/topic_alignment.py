"""Topic-based local speaker alignment (Sec. 4.1.5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

try:  # Optional neural topic modeling dependency
    from bertopic import BERTopic
except ImportError:  # pragma: no cover
    BERTopic = None  # type: ignore

from .common import names_equal
from .types import ContextWindow


MIN_BERTOPIC_DOCS = 10


@dataclass
class TopicModeler:
    """Flexible topic modeler supporting both LDA and BERTopic backends."""

    n_topics: int = 20
    max_features: int = 5000
    backend: str = "auto"  # auto | lda | bertopic
    embedding_model: str | object | None = None
    min_bertopic_docs: int = MIN_BERTOPIC_DOCS

    def __post_init__(self) -> None:
        self._impl, self._backend_used = self._create_impl(self.backend)

    def _create_impl(self, backend: str):
        backend_normalized = backend.lower()
        if backend_normalized == "bertopic" or (
            backend_normalized == "auto" and BERTopic is not None
        ):
            if BERTopic is None:
                backend_normalized = "lda"
            else:
                return _BERTTopicModeler(embedding_model=self.embedding_model), "bertopic"
        return _LDATopicModeler(n_topics=self.n_topics, max_features=self.max_features), "lda"

    @property
    def backend_used(self) -> str:
        return self._backend_used

    @property
    def is_fitted(self) -> bool:
        return self._impl.is_fitted

    def fit(self, corpus: Iterable[str]) -> None:
        documents = [doc for doc in corpus if doc]
        if not documents:
            return

        if self._backend_used == "bertopic" and len(documents) < self.min_bertopic_docs:
            self._impl, self._backend_used = self._create_impl("lda")

        try:
            self._impl.fit(documents)
        except (ValueError, TypeError):
            if self._backend_used == "bertopic":
                self._impl, self._backend_used = self._create_impl("lda")
                self._impl.fit(documents)
            else:
                raise

    def aggregate_topic_distribution(self, documents: Sequence[str]) -> np.ndarray | None:
        return self._impl.aggregate_topic_distribution(documents)


class _LDATopicModeler:
    def __init__(self, n_topics: int, max_features: int) -> None:
        self.n_topics = n_topics
        self.max_features = max_features
        self.vectorizer = CountVectorizer(max_features=self.max_features, stop_words=None)
        self.lda = LatentDirichletAllocation(n_components=self.n_topics, random_state=0)
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, corpus: Iterable[str]) -> None:
        documents = [doc for doc in corpus if doc]
        if not documents:
            return
        doc_term = self.vectorizer.fit_transform(documents)
        self.lda.fit(doc_term)
        self._is_fitted = True

    def aggregate_topic_distribution(self, documents: Sequence[str]) -> np.ndarray | None:
        if not self._is_fitted or not documents:
            return None
        doc_term = self.vectorizer.transform(documents)
        matrix = self.lda.transform(doc_term)
        if matrix.size == 0:
            return None
        dist = matrix.mean(axis=0)
        total = dist.sum()
        if total <= 0:
            return None
        return dist / total


class _BERTTopicModeler:
    def __init__(self, embedding_model: str | object | None = None) -> None:
        if BERTopic is None:
            raise ImportError("bertopic must be installed for neural topic modeling.")
        self.embedding_model = embedding_model or "all-MiniLM-L6-v2"
        self.model = BERTopic(embedding_model=self.embedding_model, calculate_probabilities=True)
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, corpus: Iterable[str]) -> None:
        documents = [doc for doc in corpus if doc]
        if not documents:
            return
        self.model.fit(documents)
        self._is_fitted = True

    def aggregate_topic_distribution(self, documents: Sequence[str]) -> np.ndarray | None:
        if not self._is_fitted or not documents:
            return None
        _, probabilities = self.model.transform(list(documents))
        if probabilities is None:
            return None
        matrix = np.asarray(probabilities, dtype=float)
        if matrix.size == 0:
            return None
        dist = matrix.mean(axis=0)
        total = dist.sum()
        if total <= 0:
            return None
        return dist / total


def topic_alignment_score(
    predicted_speaker: str,
    context: ContextWindow,
    model: TopicModeler | None = None,
    speaker_background_texts: Sequence[str] | None = None,
    min_context_messages: int = 1,
) -> float:
    """Compute Eq. (8) using the provided or auto-fitted topic model."""

    context_texts = [turn.utterance for turn in context if turn.utterance]
    if len(context_texts) < min_context_messages:
        return 0.0

    speaker_texts = [turn.utterance for turn in context if names_equal(turn.speaker, predicted_speaker)]
    additional = [text for text in (speaker_background_texts or []) if text]
    augmented_speaker_texts = speaker_texts + additional

    if not augmented_speaker_texts:
        return 0.0

    model = model or TopicModeler()
    if not model.is_fitted:
        corpus = context_texts + augmented_speaker_texts
        model.fit(corpus)

    context_dist = _safe_distribution(model, context_texts)
    speaker_dist = _safe_distribution(model, augmented_speaker_texts)

    if (context_dist is None or speaker_dist is None) and model.backend_used == "bertopic":
        fallback = TopicModeler(
            n_topics=model.n_topics,
            max_features=model.max_features,
            backend="lda",
        )
        fallback.fit(corpus)
        context_dist = _safe_distribution(fallback, context_texts)
        speaker_dist = _safe_distribution(fallback, augmented_speaker_texts)

    if context_dist is None or speaker_dist is None:
        return 0.0

    jsd = jensen_shannon_divergence(context_dist, speaker_dist)
    return float(1.0 - np.sqrt(jsd))


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = _normalize_distribution(p, eps)
    q = _normalize_distribution(q, eps)
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    ratio = np.where(p > 0, p * np.log((p + eps) / (q + eps)), 0.0)
    return float(np.sum(ratio))


def _normalize_distribution(dist: np.ndarray, eps: float) -> np.ndarray:
    total = dist.sum()
    if total <= 0:
        return np.full_like(dist, 1.0 / len(dist))
    norm = dist / (total + eps)
    return np.clip(norm, eps, 1.0)


def _safe_distribution(model: TopicModeler, docs: Sequence[str]) -> np.ndarray | None:
    try:
        return model.aggregate_topic_distribution(docs)
    except Exception:
        return None


__all__ = [
    "topic_alignment_score",
    "TopicModeler",
    "jensen_shannon_divergence",
]
