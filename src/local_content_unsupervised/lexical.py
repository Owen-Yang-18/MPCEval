"""Lexical novelty metrics for local unsupervised content flow (Sec. 4.2.2).

This is a simplified, faster implementation for MPDD (Traditional Chinese):
- Tokenize each message into words.
- Remove punctuation.
- Treat remaining unigrams as lexical units E(·).
- Use word embeddings to filter out semantic near-synonyms (Eq. 14).
- Compute per-pair IDF over {each context turn, u_p} (Eq. 16).

Flowing_weighted_SNR (Eq. 16):
    sum_{w in E_truly-novel(u_p, C^-k)} IDF(w) /
    sum_{w in E(u_p)} IDF(w)
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set

import numpy as np

from src.local_content_unsupervised.types import ContextWindow

Array = np.ndarray

DEFAULT_SIMILARITY_THRESHOLD = 0.7


@dataclass(frozen=True)
class LexicalFlowConfig:
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    tokenizer: str = "pkuseg"  # pkuseg | jieba | english (spaCy) | english_regex
    remove_stopwords_in_context: bool = True
    stopwords: Set[str] = field(
        default_factory=lambda: {
            # Intentionally minimal stopwords; only applied to context.
            "的",
            "了",
            "呢",
            "啊",
            "呀",
            "哦",
            "喔",
            "嗯",
            "唉",
            "哎",
            "嗎",
            "吗",
            "么",
            "麼",
            "吧",
        }
    )


_PUNCT_RE = re.compile(r"^[\W_]+$", flags=re.UNICODE)
_EN_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?|[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")


def _is_punct(token: str) -> bool:
    return bool(_PUNCT_RE.match(token))


class BaseTokenizer:
    backend: str

    def tokenize(self, text: str) -> List[str]:
        raise NotImplementedError


class ChineseTokenizer(BaseTokenizer):
    """Tokenizer wrapper with pluggable backends."""

    def __init__(self, backend: str) -> None:
        backend = backend.lower().strip()
        self.backend = backend
        self._segmenter = None

        if backend == "pkuseg":
            try:
                import pkuseg  # type: ignore
            except Exception as exc:
                raise ImportError(
                    "Tokenizer backend 'pkuseg' requested but pkuseg is not installed. "
                    "Install it via `pip install pkuseg`."
                ) from exc
            self._segmenter = pkuseg.pkuseg()
        elif backend == "jieba":
            try:
                import jieba  # type: ignore
            except Exception as exc:
                raise ImportError(
                    "Tokenizer backend 'jieba' requested but jieba is not installed. "
                    "Install it via `pip install jieba`."
                ) from exc
            self._segmenter = jieba
        else:
            raise ValueError("tokenizer must be 'pkuseg' or 'jieba'")

    def tokenize(self, text: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []

        if self.backend == "pkuseg":
            return [t for t in self._segmenter.cut(text) if t]  # type: ignore[union-attr]
        return [t for t in self._segmenter.lcut(text) if t]  # type: ignore[union-attr]


class EnglishRegexTokenizer(BaseTokenizer):
    """Regex-based English tokenizer with basic normalization."""

    def __init__(self, backend: str) -> None:
        backend = backend.lower().strip()
        if backend not in {"english_regex", "regex_en"}:
            raise ValueError("tokenizer must be 'english_regex' or 'regex_en'")
        self.backend = backend

    def tokenize(self, text: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        text = text.lower()
        return _EN_TOKEN_RE.findall(text)


class SpacyEnglishTokenizer(BaseTokenizer):
    """spaCy English tokenizer with lightweight pipeline."""

    def __init__(self, backend: str) -> None:
        backend = backend.lower().strip()
        if backend not in {"english", "en", "spacy", "spacy_en"}:
            raise ValueError("tokenizer must be 'english', 'en', 'spacy', or 'spacy_en'")
        self.backend = backend

        try:
            import spacy  # type: ignore
        except Exception as exc:
            raise ImportError(
                "Tokenizer backend 'english' requested but spaCy is not installed. "
                "Install it via `pip install spacy`."
            ) from exc

        try:
            self._nlp = spacy.load(
                "en_core_web_sm",
                disable=[
                    "tagger",
                    "parser",
                    "ner",
                    "lemmatizer",
                    "attribute_ruler",
                    "tok2vec",
                ],
            )
        except Exception:
            self._nlp = spacy.blank("en")

    def tokenize(self, text: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        doc = self._nlp(text)
        return [t.text.lower() for t in doc if t.text and not t.is_space]


@lru_cache(maxsize=4)
def get_tokenizer(backend: str) -> BaseTokenizer:
    """Return a cached tokenizer instance (pkuseg init is expensive)."""

    backend = backend.lower().strip()
    if backend in {"pkuseg", "jieba"}:
        return ChineseTokenizer(backend)
    if backend in {"english", "en", "spacy", "spacy_en"}:
        return SpacyEnglishTokenizer(backend)
    if backend in {"english_regex", "regex_en"}:
        return EnglishRegexTokenizer(backend)
    raise ValueError("tokenizer must be 'pkuseg', 'jieba', 'english', or 'english_regex'")


@lru_cache(maxsize=50_000)
def tokenize_text(backend: str, text: str) -> tuple[str, ...]:
    """Tokenize with a cached tokenizer and memoize tokenization results."""

    return tuple(get_tokenizer(backend).tokenize(text))


def extract_lexical_terms(
    text: str,
    *,
    tokenizer: BaseTokenizer,
    stopwords: Set[str] | None = None,
    remove_stopwords: bool = False,
) -> Set[str]:
    terms: set[str] = set()
    for tok in tokenize_text(tokenizer.backend, text):
        tok = tok.strip()
        if not tok:
            continue
        if remove_stopwords and stopwords and tok in stopwords:
            continue
        if _is_punct(tok):
            continue
        terms.add(tok)
    return terms


class WordEmbeddings:
    """Word embedding lookup loaded from a gensim KeyedVectors file."""

    def __init__(self, vectors_path: Path, *, limit: int | None = None) -> None:
        self.vectors_path = vectors_path
        self._kv = self._load_vectors(vectors_path, limit=limit)
        self.dim = int(self._kv.vector_size)

    @staticmethod
    def _load_vectors(path: Path, *, limit: int | None):
        try:
            from gensim.models import KeyedVectors  # type: ignore
        except Exception as exc:
            raise ImportError(
                "gensim is required to load word embeddings. Install via `pip install gensim`."
            ) from exc

        if path.suffix == ".kv":
            return KeyedVectors.load(str(path), mmap="r")
        if path.suffixes[-2:] == [".vec", ".gz"] or path.suffix == ".vec":
            # Slower (loads into RAM) but convenient for smoke tests.
            # Pass the filename directly so gensim/smart_open can handle .gz.
            return KeyedVectors.load_word2vec_format(str(path), binary=False, limit=limit)
        return KeyedVectors.load(str(path), mmap="r")

    def contains(self, token: str) -> bool:
        return token in self._kv

    @lru_cache(maxsize=200_000)
    def vector(self, token: str) -> Array | None:
        if token not in self._kv:
            return None
        vec = self._kv.get_vector(token)
        vec = np.asarray(vec, dtype=float)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return None
        return vec / norm


def _pair_idf(
    docs: Sequence[Set[str]],
) -> Dict[str, float]:
    """Per-pair IDF over the document set {context turns, u_p}."""

    total_docs = max(len(docs), 1)
    df: Counter[str] = Counter()
    for units in docs:
        df.update(units)
    idf: Dict[str, float] = {}
    for term, freq in df.items():
        idf[term] = math.log((total_docs + 1.0) / (freq + 1.0)) + 1.0
    return idf


def flowing_weighted_snr(
    context: ContextWindow,
    next_message: str,
    *,
    embeddings: WordEmbeddings,
    config: LexicalFlowConfig | None = None,
    idf: Mapping[str, float] | None = None,
    idf_scope: str = "pair",
) -> float:
    """Compute Flowing_weighted_SNR (Eq. 16) for a (context, next_message) pair.

    OOV tokens are skipped (excluded from E(u_p) and E(C)).
    """

    cfg = config or LexicalFlowConfig()
    tokenizer = get_tokenizer(cfg.tokenizer)

    context_docs: list[set[str]] = []
    context_vocab: set[str] = set()
    for turn in context:
        terms = extract_lexical_terms(
            turn.utterance,
            tokenizer=tokenizer,
            stopwords=cfg.stopwords,
            remove_stopwords=cfg.remove_stopwords_in_context,
        )
        terms = {t for t in terms if embeddings.contains(t)}
        if terms:
            context_docs.append(terms)
            context_vocab |= terms

    # Do NOT remove stopwords from u_p (per experiment design).
    predicted_terms = extract_lexical_terms(
        next_message,
        tokenizer=tokenizer,
        stopwords=cfg.stopwords,
        remove_stopwords=False,
    )
    predicted_terms = {t for t in predicted_terms if embeddings.contains(t)}
    if not predicted_terms:
        return 0.0

    # Eq. (14): truly novel terms are those with max similarity to context < tau.
    if not context_vocab:
        truly_novel = set(predicted_terms)
    else:
        pred_list = sorted(predicted_terms)
        ctx_list = sorted(context_vocab)

        pred_vecs = [embeddings.vector(t) for t in pred_list]
        ctx_vecs = [embeddings.vector(t) for t in ctx_list]
        pred_pairs = [(t, v) for t, v in zip(pred_list, pred_vecs) if v is not None]
        ctx_matrix = (
            np.stack([v for v in ctx_vecs if v is not None], axis=0)
            if any(v is not None for v in ctx_vecs)
            else None
        )

        if not pred_pairs or ctx_matrix is None or ctx_matrix.size == 0:
            # If we can't compare embeddings, treat all in-vocab predicted terms as novel.
            truly_novel = set(predicted_terms)
        else:
            pred_terms2 = [t for t, _ in pred_pairs]
            pred_matrix = np.stack([v for _, v in pred_pairs], axis=0)
            sims = pred_matrix @ ctx_matrix.T
            max_sim = sims.max(axis=1) if sims.size else np.zeros((len(pred_terms2),), dtype=float)
            truly_novel = {
                t for t, s in zip(pred_terms2, max_sim) if float(s) < cfg.similarity_threshold
            }

    # Eq. (16): per-pair IDF weighting over {context turns, u_p}.
    if idf_scope not in {"pair", "corpus"}:
        raise ValueError("idf_scope must be 'pair' or 'corpus'")
    if idf_scope == "corpus":
        idf_lookup = dict(idf or {})
    else:
        idf_lookup = _pair_idf([*context_docs, set(predicted_terms)])

    def weight(term: str) -> float:
        return float(idf_lookup.get(term, 1.0))

    denom = sum(weight(t) for t in predicted_terms)
    if denom <= 0:
        return 0.0
    numer = sum(weight(t) for t in truly_novel)
    return float(numer / denom)


def convert_vec_to_kv(vec_path: Path, kv_path: Path, *, limit: int | None = None) -> None:
    """Utility: convert word2vec text vectors (.vec or .vec.gz) to gensim .kv.

    This is intentionally simple and loads the whole file; for huge vectors you
    may want to do this on a larger node or with a smaller pretrained file.
    """

    try:
        from gensim.models import KeyedVectors  # type: ignore
    except Exception as exc:
        raise ImportError("gensim is required to convert embeddings.") from exc

    # Pass the filename directly so gensim/smart_open can handle .gz.
    kv = KeyedVectors.load_word2vec_format(str(vec_path), binary=False, limit=limit)
    kv.save(str(kv_path))
