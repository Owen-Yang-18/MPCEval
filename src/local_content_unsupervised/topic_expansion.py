"""Topic Expansion Score (TES) for Sec. 4.2.2 (Eq. 21).

Implements the Topic Expansion Score described after Eq. (20).

Given:
  - Topic distribution θ(X) over L topics from a topic model (BERTopic)
  - A context C with turns u_1..u_m and predicted next message u_p

Step 1: choose k* (reasonable context size) by expansion in steps of Δ:
  stop at k when JSD(θ(C^{-k}), θ(C^{-(k+Δ)})) > τ.

Step 2: compute Q_p = θ(C^{-ℓ} ⊕ u_p), with small ℓ to avoid dilution.
Let T_dom(P*, ρ) be the smallest set of topics covering mass ρ of P*.
Then:
  FlowingTES = Σ_{l ∉ T_dom(P*,ρ)} Q_{p,l}.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


def _concat_utterances(turns: Sequence[Mapping[str, Any]]) -> str:
    parts: list[str] = []
    for t in turns:
        utt = str(t.get("utterance", "")).strip()
        if utt:
            parts.append(utt)
    return "\n".join(parts).strip()


def _tdom_indices(p: np.ndarray, rho: float) -> set[int]:
    if p.size == 0:
        return set()
    if rho <= 0:
        return set()
    if rho >= 1:
        return set(range(int(p.size)))
    order = np.argsort(-p)  # descending
    s = 0.0
    keep: set[int] = set()
    for idx in order.tolist():
        keep.add(int(idx))
        s += float(p[int(idx)])
        if s >= rho:
            break
    return keep


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen–Shannon divergence with base 2, in [0, 1]."""

    if p.size == 0 or q.size == 0:
        return 0.0
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape for JSD")

    # Pure-numpy JSD (base 2) to avoid requiring scipy at import time.
    # Assumes p and q are already normalized distributions.
    p = p.astype(float, copy=False)
    q = q.astype(float, copy=False)
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        aa = a[mask]
        bb = b[mask]
        # log2(a/b) = ln(a/b) / ln(2)
        return float(np.sum(aa * (np.log(aa) - np.log(bb))) / math.log(2.0))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


@dataclass
class BERTopicTheta:
    """BERTopic wrapper that produces topic distributions θ(X)."""

    model_path: Path
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    device: str = "cpu"  # "cpu" or "cuda"
    min_topic_size: int = 10
    nr_topics: int | None = None

    def __post_init__(self) -> None:
        self._model = None
        self._topic_ids: list[int] = []

    def load_or_fit(self, docs: Sequence[str], *, refresh: bool = False) -> None:
        from bertopic import BERTopic  # type: ignore

        if self.model_path.exists() and not refresh:
            self._model = BERTopic.load(str(self.model_path))
            self._init_topic_ids()
            return

        # Embeddings backend (SentenceTransformer). If the model isn't present
        # locally, this may trigger a download.
        from sentence_transformers import SentenceTransformer  # type: ignore

        embedder = SentenceTransformer(self.embedding_model, device=self.device)
        model = BERTopic(
            embedding_model=embedder,
            calculate_probabilities=True,
            min_topic_size=int(self.min_topic_size),
            nr_topics=self.nr_topics,
        )
        model.fit(list(docs))
        self._model = model
        self._init_topic_ids()

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(str(self.model_path))

    def _init_topic_ids(self) -> None:
        if self._model is None:
            self._topic_ids = []
            return
        try:
            info = self._model.get_topic_info()
            topics = [int(x) for x in info["Topic"].tolist()]
        except Exception:
            topics = []
        # Keep topics excluding outlier (-1) by default; include it if the
        # returned probability vectors include it (handled at runtime).
        self._topic_ids = [t for t in topics if t != -1]

    def theta(self, text: str) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("BERTopic model not initialized. Call load_or_fit() first.")
        t = (text or "").strip()
        if not t:
            return np.zeros((0,), dtype=float)

        # Prefer transform() probabilities when available; they tend to be less
        # degenerate (approximate_distribution can return all-zeros for short texts).
        vec: np.ndarray | None = None
        try:
            _topics, probs = self._model.transform([t])
            if probs is not None:
                v = np.asarray(probs[0], dtype=float)
                if np.isfinite(v).all() and float(v.sum()) > 0:
                    vec = v
        except Exception:
            vec = None

        if vec is None:
            approx = getattr(self._model, "approximate_distribution", None)
            if callable(approx):
                dist = self._model.approximate_distribution([t])
                probs2 = dist[0] if isinstance(dist, tuple) else dist
                v2 = np.asarray(probs2[0], dtype=float)
                if np.isfinite(v2).all() and float(v2.sum()) > 0:
                    vec = v2

        if vec is None:
            return np.zeros((0,), dtype=float)

        # Normalize to a distribution (defensive).
        s = float(vec.sum())
        if s <= 0:
            return np.zeros_like(vec, dtype=float)
        return (vec / s).astype(float)


def choose_k_star(
    context_turns: Sequence[Mapping[str, Any]],
    *,
    theta_model: BERTopicTheta,
    delta: int = 2,
    tau: float = 0.1,
) -> int:
    """Choose k* by expanding k in steps of Δ until JSD change exceeds τ."""

    m = len(context_turns)
    if m <= 0:
        return 0
    if delta <= 0:
        raise ValueError("delta must be > 0")
    if tau < 0:
        raise ValueError("tau must be >= 0")

    k = min(delta, m)
    while k + delta <= m:
        doc_k = _concat_utterances(context_turns[-k:])
        doc_k2 = _concat_utterances(context_turns[-(k + delta) :])
        p = theta_model.theta(doc_k)
        q = theta_model.theta(doc_k2)
        # If topic vectors are degenerate, keep expanding.
        if p.size == 0 or q.size == 0 or p.shape != q.shape:
            k += delta
            continue
        if _jsd(p, q) > tau:
            break
        k += delta
    return int(k)


def topic_expansion_score(
    context_turns: Sequence[Mapping[str, Any]],
    predicted_message: str,
    *,
    theta_model: BERTopicTheta,
    ell: int = 2,
    delta: int = 2,
    rho: float = 0.8,
    tau: float = 0.1,
    up_repeat: int = 0,
    up_repeat_cap: int = 20,
) -> float:
    """Compute FlowingTES (Eq. 21) for a single (context, u_p) pair."""

    if ell < 0:
        raise ValueError("ell must be >= 0")
    if up_repeat < 0:
        raise ValueError("up_repeat must be >= 0 (0 means auto)")
    if up_repeat_cap <= 0:
        raise ValueError("up_repeat_cap must be > 0")
    if not context_turns:
        return 0.0

    k_star = choose_k_star(context_turns, theta_model=theta_model, delta=delta, tau=tau)
    if k_star <= 0:
        return 0.0

    p_star_doc = _concat_utterances(context_turns[-k_star:])
    p_star = theta_model.theta(p_star_doc)
    if p_star.size == 0:
        return 0.0

    tdom = _tdom_indices(p_star, float(rho))

    ell_turns = context_turns[-ell:] if ell > 0 else []
    # Anti-dilution: upweight u_p by repeating it (monotonic in length).
    up = (predicted_message or "").strip()
    ell_text = _concat_utterances(ell_turns)
    if up_repeat == 0:
        # Auto: choose repeats so u_p is not washed out by C^{-ell} length.
        denom = max(len(up), 1)
        ratio = len(ell_text) / float(denom)
        repeat_eff = max(1, min(int(math.ceil(ratio)), int(up_repeat_cap)))
    else:
        repeat_eff = int(up_repeat)
    up_aug = "\n".join([up] * int(repeat_eff)) if up else ""
    q_doc = "\n".join([ell_text, up_aug]).strip()
    q_p = theta_model.theta(q_doc)
    if q_p.size == 0 or q_p.shape != p_star.shape:
        return 0.0

    inactive = [i for i in range(int(q_p.size)) if i not in tdom]
    if not inactive:
        return 0.0
    return float(q_p[inactive].sum())


def load_tes_context_cache(path: Path) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    if not path.exists():
        return {}, {}
    with path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)
    if isinstance(raw, dict) and "contexts" in raw and isinstance(raw["contexts"], dict):
        meta = raw.get("meta", {}) if isinstance(raw.get("meta", {}), dict) else {}
        contexts_raw = raw["contexts"]
    else:
        meta = {}
        contexts_raw = raw
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(contexts_raw, dict):
        for did, v in contexts_raw.items():
            if isinstance(v, dict):
                out[str(did)] = dict(v)
    return out, dict(meta)


def save_tes_context_cache(path: Path, *, cache: Dict[str, Dict[str, Any]], meta: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": dict(meta), "contexts": cache}
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False)
