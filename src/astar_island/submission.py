"""Submission validation and serialization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from .constants import NUM_CLASSES
from .models import Tensor3D, ValidationResult


def validate_prediction_tensor(
    prediction: Tensor3D,
    *,
    expected_width: int,
    expected_height: int,
    sum_tolerance: float = 0.01,
) -> ValidationResult:
    errors: list[str] = []

    if len(prediction) != expected_height:
        errors.append(f"Expected {expected_height} rows, got {len(prediction)}")
        return ValidationResult(ok=False, errors=errors)

    for y, row in enumerate(prediction):
        if len(row) != expected_width:
            errors.append(f"Row {y}: expected {expected_width} cols, got {len(row)}")
            continue

        for x, probs in enumerate(row):
            if len(probs) != NUM_CLASSES:
                errors.append(f"Cell ({y},{x}): expected {NUM_CLASSES} probs, got {len(probs)}")
                continue

            sum_probs = 0.0
            for value in probs:
                if value < 0.0:
                    errors.append(f"Cell ({y},{x}): negative probability")
                sum_probs += value

            if abs(sum_probs - 1.0) > sum_tolerance:
                errors.append(
                    f"Cell ({y},{x}): probs sum to {sum_probs:.6f}, expected 1.0"
                )

    return ValidationResult(ok=(len(errors) == 0), errors=errors)


def floor_and_normalize(prediction: Tensor3D, floor: float = 0.01) -> Tensor3D:
    if floor < 0.0:
        raise ValueError("floor must be non-negative")

    normalized: Tensor3D = []
    for row in prediction:
        out_row: list[list[float]] = []
        for probs in row:
            floored = [max(float(p), floor) for p in probs]
            total = sum(floored)
            if total == 0.0:
                out_row.append([1.0 / NUM_CLASSES] * NUM_CLASSES)
            else:
                out_row.append([p / total for p in floored])
        normalized.append(out_row)

    return normalized


def serialize_submission(round_id: str, seed_index: int, prediction: Tensor3D) -> dict[str, object]:
    return {
        "round_id": str(round_id),
        "seed_index": int(seed_index),
        "prediction": prediction,
    }


@dataclass(slots=True)
class SeedSubmissionPlan:
    seed_index: int
    payload: dict[str, object]
    used_fallback: bool
    source: str
    errors: list[str]


def build_safe_round_submission(
    *,
    round_id: str,
    seeds_count: int,
    map_width: int,
    map_height: int,
    predictions_by_seed: Mapping[int, Tensor3D | None],
    fallback_prediction_for_seed: Callable[[int], Tensor3D],
    probability_floor: float = 0.01,
    sum_tolerance: float = 0.01,
) -> list[SeedSubmissionPlan]:
    """Build a safe per-seed submission plan with fallback and validation."""
    plans: list[SeedSubmissionPlan] = []

    for seed_index in range(max(0, int(seeds_count))):
        raw_prediction = predictions_by_seed.get(seed_index)
        source = "model"
        used_fallback = False
        errors: list[str] = []

        normalized_prediction: Tensor3D | None = None
        if raw_prediction is not None:
            normalized_prediction, raw_errors = _normalize_and_validate(
                raw_prediction,
                expected_width=map_width,
                expected_height=map_height,
                probability_floor=probability_floor,
                sum_tolerance=sum_tolerance,
            )
            if raw_errors:
                errors.extend(raw_errors)

        if normalized_prediction is None:
            source = "fallback"
            used_fallback = True
            fallback_prediction = fallback_prediction_for_seed(seed_index)
            normalized_prediction, fallback_errors = _normalize_and_validate(
                fallback_prediction,
                expected_width=map_width,
                expected_height=map_height,
                probability_floor=probability_floor,
                sum_tolerance=sum_tolerance,
            )
            if fallback_errors:
                first_error = fallback_errors[0]
                raise ValueError(
                    f"Fallback prediction invalid for seed={seed_index}: {first_error}"
                )

        plans.append(
            SeedSubmissionPlan(
                seed_index=seed_index,
                payload=serialize_submission(
                    round_id=round_id,
                    seed_index=seed_index,
                    prediction=normalized_prediction,
                ),
                used_fallback=used_fallback,
                source=source,
                errors=errors,
            )
        )

    return plans


def missing_seed_indices(*, submitted_seed_indices: list[int], seeds_count: int) -> list[int]:
    submitted = {
        int(seed_index)
        for seed_index in submitted_seed_indices
        if 0 <= int(seed_index) < int(seeds_count)
    }
    return [seed_index for seed_index in range(max(0, int(seeds_count))) if seed_index not in submitted]


def _normalize_and_validate(
    prediction: Tensor3D,
    *,
    expected_width: int,
    expected_height: int,
    probability_floor: float,
    sum_tolerance: float,
) -> tuple[Tensor3D | None, list[str]]:
    try:
        normalized = floor_and_normalize(prediction, floor=probability_floor)
    except (TypeError, ValueError) as exc:
        return None, [str(exc)]

    validation = validate_prediction_tensor(
        normalized,
        expected_width=expected_width,
        expected_height=expected_height,
        sum_tolerance=sum_tolerance,
    )
    if not validation.ok:
        return None, validation.errors
    return normalized, []
