"""Centroid-based global speaker-content consistency metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class SpeakerScores:
    count: int
    single_avg: float
    single_max: float
    multi_avg: float
    multi_max: float
    k_selected: int


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return matrix / norms


def _single_centroid_scores(embeddings: np.ndarray) -> tuple[float, float]:
    if embeddings.shape[0] == 0:
        return 0.0, 0.0
    centroid = embeddings.mean(axis=0, keepdims=True)
    centroid = _normalize_rows(centroid)
    sims = embeddings @ centroid.T
    sims = sims.squeeze(axis=1)
    return float(sims.mean()), float(sims.max())


def _select_gmm_k(
    embeddings: np.ndarray,
    max_k: int,
    min_cluster_size: int,
    random_state: int,
    min_k_large: int,
    large_threshold: int,
    min_messages_per_cluster: int,
) -> tuple[int, np.ndarray]:
    m = embeddings.shape[0]
    if m < min_cluster_size or m <= 1:
        return 1, embeddings.mean(axis=0, keepdims=True)

    try:
        from sklearn.mixture import GaussianMixture
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("scikit-learn is required for GMM clustering") from exc

    k_max = min(
        max_k,
        int(m ** 0.5) or 1,
        max(1, m // max(min_messages_per_cluster, 1)),
    )
    k_min = 1
    if m >= large_threshold:
        k_min = min_k_large
    k_min = min(k_min, k_max)
    best_k = 1
    best_bic = None
    best_means = embeddings.mean(axis=0, keepdims=True)
    for k in range(k_min, k_max + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            reg_covar=1e-6,
            random_state=random_state,
        )
        gmm.fit(embeddings)
        bic = gmm.bic(embeddings)
        if best_bic is None or bic < best_bic:
            best_bic = bic
            best_k = k
            best_means = gmm.means_
    return best_k, best_means


def _multi_centroid_scores(
    embeddings: np.ndarray,
    centroids: np.ndarray,
) -> tuple[float, float]:
    if embeddings.shape[0] == 0:
        return 0.0, 0.0
    centroids = _normalize_rows(centroids)
    sims = embeddings @ centroids.T
    best = sims.max(axis=1)
    return float(best.mean()), float(best.max())


def compute_speaker_scores(
    speaker_embeddings: Dict[str, np.ndarray],
    max_k: int,
    min_cluster_size: int,
    random_state: int,
    min_k_large: int,
    large_threshold: int,
    min_messages_per_cluster: int,
) -> Dict[str, SpeakerScores]:
    results: Dict[str, SpeakerScores] = {}
    for speaker, embeds in speaker_embeddings.items():
        single_avg, single_max = _single_centroid_scores(embeds)
        k_selected, centroids = _select_gmm_k(
            embeds,
            max_k=max_k,
            min_cluster_size=min_cluster_size,
            random_state=random_state,
            min_k_large=min_k_large,
            large_threshold=large_threshold,
            min_messages_per_cluster=min_messages_per_cluster,
        )
        multi_avg, multi_max = _multi_centroid_scores(embeds, centroids)
        results[speaker] = SpeakerScores(
            count=embeds.shape[0],
            single_avg=single_avg,
            single_max=single_max,
            multi_avg=multi_avg,
            multi_max=multi_max,
            k_selected=k_selected,
        )
    return results


def weighted_aggregate(
    scores: Dict[str, SpeakerScores],
) -> Dict[str, Dict[str, float]]:
    total = sum(value.count for value in scores.values())
    if total == 0:
        return {
            "single": {"avg": 0.0, "max": 0.0},
            "multi": {"avg": 0.0, "max": 0.0},
        }

    single_avg = sum(value.single_avg * value.count for value in scores.values()) / total
    single_max = sum(value.single_max * value.count for value in scores.values()) / total
    multi_avg = sum(value.multi_avg * value.count for value in scores.values()) / total
    multi_max = sum(value.multi_max * value.count for value in scores.values()) / total
    return {
        "single": {"avg": float(single_avg), "max": float(single_max)},
        "multi": {"avg": float(multi_avg), "max": float(multi_max)},
    }


def to_numpy_embeddings(embeddings: torch.Tensor) -> np.ndarray:
    matrix = embeddings.detach().cpu().numpy()
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    return _normalize_rows(matrix.astype(np.float64))
