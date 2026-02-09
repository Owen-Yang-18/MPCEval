"""Global speaker diversity and dynamics metrics."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Mapping, Sequence

import numpy as np

try:  # Optional dependency; GPU path uses torch when available.
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore


def normalized_speaker_entropy(speaker_counts: Mapping[str, int]) -> float:
    total = sum(speaker_counts.values())
    num_speakers = len(speaker_counts)
    if total <= 0 or num_speakers <= 1:
        return 0.0
    entropy = 0.0
    for count in speaker_counts.values():
        if count <= 0:
            continue
        p_i = count / total
        entropy -= p_i * math.log(p_i, 2)
    return float(entropy / math.log(num_speakers, 2))


def gini_coefficient(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=float)
    if np.all(arr == 0):
        return 0.0
    diff_sum = np.abs(arr[:, None] - arr[None, :]).sum()
    return float(diff_sum / (2 * len(arr) * arr.sum()))


def local_novelty_radius(
    similarity: np.ndarray | "torch.Tensor",
    max_neighbors: int,
    percentile: float,
) -> tuple[np.ndarray, float]:
    n = similarity.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=float), 0.0

    if torch is not None and isinstance(similarity, torch.Tensor):
        sims = torch.nan_to_num(
            similarity.clone(), nan=0.0, posinf=1.0, neginf=-1.0
        )
        sims.fill_diagonal_(-float("inf"))
        nearest = sims.max(dim=1).values
        tau = torch.quantile(nearest, percentile / 100.0).item()

        k = min(max_neighbors, n - 1)
        top_vals = sims.topk(k=k, dim=1).values
        mask = top_vals >= tau
        counts = mask.sum(dim=1)
        sums = (top_vals * mask).sum(dim=1)
        fallback = torch.where(
            torch.isfinite(top_vals[:, 0]),
            top_vals[:, 0],
            torch.zeros_like(top_vals[:, 0]),
        )
        mean_sim = torch.where(counts > 0, sums / counts.clamp(min=1), fallback)
        novelty = 1.0 - mean_sim
        return novelty.cpu().numpy(), float(tau)

    sims = np.nan_to_num(np.array(similarity, copy=True), nan=0.0, posinf=1.0, neginf=-1.0)
    np.fill_diagonal(sims, -np.inf)
    nearest = np.max(sims, axis=1)
    tau = float(np.quantile(nearest, percentile / 100.0))
    k = min(max_neighbors, n - 1)
    top_vals = np.partition(sims, -k, axis=1)[:, -k:]
    mask = top_vals >= tau
    counts = mask.sum(axis=1)
    sums = (top_vals * mask).sum(axis=1)
    mean_sim = np.where(counts > 0, sums / np.clip(counts, 1, None), nearest)
    novelty = 1.0 - mean_sim
    return novelty.astype(float), tau


def kernel_logdet_score(similarity: np.ndarray | "torch.Tensor", eps: float) -> float:
    n = similarity.shape[0]
    if n <= 1:
        return 0.0
    if torch is not None and isinstance(similarity, torch.Tensor):
        kernel = similarity + torch.eye(n, device=similarity.device) * eps
        sign, logdet = torch.linalg.slogdet(kernel)
        if sign <= 0:
            return 0.0
        return float(logdet.item())
    kernel = np.array(similarity, copy=True)
    kernel = kernel + np.eye(n) * eps
    sign, logdet = np.linalg.slogdet(kernel)
    if sign <= 0:
        return 0.0
    return float(logdet)


def graph_mst_score(similarity: np.ndarray) -> float:
    n = similarity.shape[0]
    if n <= 1:
        return 0.0
    try:
        from scipy.sparse.csgraph import minimum_spanning_tree
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("scipy is required for graph-mst spread") from exc
    dist = 1.0 - similarity
    dist = np.clip(dist, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    dist = np.minimum(dist, dist.T)
    mst = minimum_spanning_tree(dist)
    total = float(mst.sum())
    return total / (n - 1)


def compute_global_spread(
    similarity_t: np.ndarray | "torch.Tensor",
    similarity_np: np.ndarray,
    method: str,
    eps: float,
) -> float:
    if method == "kernel-logdet":
        return kernel_logdet_score(similarity_t, eps=eps)
    if method == "graph-mst":
        return graph_mst_score(similarity_np)
    raise ValueError(f"Unknown global spread method: {method}")


def compute_global_deltas(
    similarity_t: np.ndarray | "torch.Tensor",
    similarity_np: np.ndarray,
    speaker_ids: Sequence[str],
    method: str,
    eps: float,
) -> tuple[Dict[str, float], float]:
    full_score = compute_global_spread(
        similarity_t, similarity_np, method=method, eps=eps
    )
    deltas: Dict[str, float] = {}
    speakers = sorted(set(speaker_ids))
    for speaker in speakers:
        mask_np = np.array([sid != speaker for sid in speaker_ids], dtype=bool)
        if mask_np.sum() <= 1:
            reduced = 0.0
        else:
            if method == "kernel-logdet":
                if torch is not None and isinstance(similarity_t, torch.Tensor):
                    mask_t = torch.as_tensor(mask_np, device=similarity_t.device)
                    sub_sim = similarity_t[mask_t][:, mask_t]
                else:
                    sub_sim = similarity_np[np.ix_(mask_np, mask_np)]
                reduced = kernel_logdet_score(sub_sim, eps=eps)
            else:
                sub_sim = similarity_np[np.ix_(mask_np, mask_np)]
                reduced = graph_mst_score(sub_sim)
        delta = full_score - reduced
        deltas[speaker] = max(0.0, float(delta))
    return deltas, full_score


def compute_contributions(
    similarity_t: np.ndarray | "torch.Tensor",
    similarity_np: np.ndarray,
    speaker_ids: Sequence[str],
    max_neighbors: int,
    percentile: float,
    global_method: str,
    lambda_weight: float,
    eps: float,
) -> Dict[str, object]:
    novelty_scores, tau = local_novelty_radius(
        similarity_t, max_neighbors=max_neighbors, percentile=percentile
    )
    local_totals: Dict[str, float] = defaultdict(float)
    for speaker_id, novelty in zip(speaker_ids, novelty_scores):
        local_totals[speaker_id] += float(novelty)

    global_deltas, global_full = compute_global_deltas(
        similarity_t,
        similarity_np,
        speaker_ids,
        method=global_method,
        eps=eps,
    )

    local_sum = sum(local_totals.values())
    global_sum = sum(global_deltas.values())
    local_norm = {
        speaker: (value / local_sum if local_sum > 0 else 0.0)
        for speaker, value in local_totals.items()
    }
    global_norm = {
        speaker: (value / global_sum if global_sum > 0 else 0.0)
        for speaker, value in global_deltas.items()
    }

    combined: Dict[str, float] = {}
    speakers = sorted(set(speaker_ids))
    for speaker in speakers:
        if local_sum > 0 and global_sum > 0:
            combined[speaker] = (
                lambda_weight * local_norm.get(speaker, 0.0)
                + (1.0 - lambda_weight) * global_norm.get(speaker, 0.0)
            )
        elif local_sum > 0:
            combined[speaker] = local_norm.get(speaker, 0.0)
        elif global_sum > 0:
            combined[speaker] = global_norm.get(speaker, 0.0)
        else:
            combined[speaker] = 0.0

    sgini = gini_coefficient(list(combined.values()))
    sgini_local = gini_coefficient(list(local_totals.values()))
    sgini_global = gini_coefficient(list(global_deltas.values()))

    return {
        "local_novelty_totals": dict(local_totals),
        "global_spread_deltas": global_deltas,
        "combined_contribution": combined,
        "sgini": sgini,
        "sgini_local": sgini_local,
        "sgini_global": sgini_global,
        "global_spread_full": global_full,
        "radius_tau": tau,
    }
