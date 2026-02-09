"""Global unsupervised content metrics (Section 5.2.2)."""

from __future__ import annotations

import math

import torch


def progression_distance(embeddings: torch.Tensor) -> float | None:
    if embeddings.shape[0] < 2:
        return None
    start = embeddings[0]
    end = embeddings[-1]
    distance = torch.norm(end - start, p=2).item()
    return distance / math.log10(embeddings.shape[0] + 1)


def raw_displacement(embeddings: torch.Tensor) -> float | None:
    if embeddings.shape[0] < 2:
        return None
    start = embeddings[0]
    end = embeddings[-1]
    return float(torch.norm(end - start, p=2).item())


def harmonic_mean_progression(
    embeddings: torch.Tensor, epsilon: float
) -> float | None:
    if embeddings.shape[0] < 2:
        return None
    diffs = embeddings[1:] - embeddings[:-1]
    step_dist = torch.norm(diffs, p=2, dim=1)
    denom = step_dist + epsilon
    if denom.numel() == 0:
        return None
    return float(denom.numel() / torch.sum(1.0 / denom).item())


def mean_step_distance(embeddings: torch.Tensor) -> float | None:
    if embeddings.shape[0] < 2:
        return None
    diffs = embeddings[1:] - embeddings[:-1]
    step_dist = torch.norm(diffs, p=2, dim=1)
    return float(step_dist.mean().item())
