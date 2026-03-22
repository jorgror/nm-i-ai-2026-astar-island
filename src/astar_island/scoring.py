"""Official score implementation for Astar Island."""

from __future__ import annotations

import math

from .constants import NUM_CLASSES
from .models import Tensor3D


def cell_entropy(distribution: list[float]) -> float:
    _validate_cell_distribution(distribution)
    entropy = 0.0
    for p_i in distribution:
        if p_i > 0.0:
            entropy -= p_i * math.log(p_i)
    return entropy


def cell_kl_divergence(ground_truth: list[float], prediction: list[float]) -> float:
    _validate_cell_distribution(ground_truth)
    _validate_cell_distribution(prediction)

    kl = 0.0
    for p_i, q_i in zip(ground_truth, prediction):
        if p_i <= 0.0:
            continue
        if q_i <= 0.0:
            return math.inf
        kl += p_i * math.log(p_i / q_i)
    return kl


def weighted_kl(ground_truth: Tensor3D, prediction: Tensor3D) -> float:
    _validate_matching_shapes(ground_truth, prediction)

    weighted_kl_sum = 0.0
    entropy_sum = 0.0

    for gt_row, pred_row in zip(ground_truth, prediction):
        for gt_cell, pred_cell in zip(gt_row, pred_row):
            ent = cell_entropy(gt_cell)
            entropy_sum += ent
            if ent == 0.0:
                continue

            kl = cell_kl_divergence(gt_cell, pred_cell)
            if math.isinf(kl):
                return math.inf
            weighted_kl_sum += ent * kl

    if entropy_sum == 0.0:
        return 0.0
    return weighted_kl_sum / entropy_sum


def score_from_weighted_kl(weighted_kl_value: float) -> float:
    if math.isinf(weighted_kl_value):
        return 0.0
    score = 100.0 * math.exp(-3.0 * weighted_kl_value)
    return max(0.0, min(100.0, score))


def score_seed(ground_truth: Tensor3D, prediction: Tensor3D) -> tuple[float, float]:
    """Return (weighted_kl, score)."""
    wkl = weighted_kl(ground_truth, prediction)
    return wkl, score_from_weighted_kl(wkl)


def score_round(seed_scores: list[float], expected_seeds: int = 5) -> float:
    if expected_seeds <= 0:
        raise ValueError("expected_seeds must be > 0")
    if len(seed_scores) > expected_seeds:
        raise ValueError("seed_scores longer than expected_seeds")

    total = sum(seed_scores)
    missing = expected_seeds - len(seed_scores)
    if missing > 0:
        total += 0.0 * missing
    return total / float(expected_seeds)


def _validate_matching_shapes(a: Tensor3D, b: Tensor3D) -> None:
    if len(a) != len(b):
        raise ValueError(f"Height mismatch: {len(a)} vs {len(b)}")

    for y, (row_a, row_b) in enumerate(zip(a, b)):
        if len(row_a) != len(row_b):
            raise ValueError(f"Width mismatch at row {y}: {len(row_a)} vs {len(row_b)}")
        for x, (cell_a, cell_b) in enumerate(zip(row_a, row_b)):
            if len(cell_a) != len(cell_b):
                raise ValueError(
                    f"Class dimension mismatch at ({y},{x}): {len(cell_a)} vs {len(cell_b)}"
                )


def _validate_cell_distribution(distribution: list[float]) -> None:
    if len(distribution) != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} class probabilities, got {len(distribution)}")

    prob_sum = 0.0
    for value in distribution:
        if value < 0.0:
            raise ValueError("Negative probability encountered")
        prob_sum += value

    if abs(prob_sum - 1.0) > 1e-2:
        raise ValueError(f"Cell probabilities sum to {prob_sum:.8f}, expected 1.0")
