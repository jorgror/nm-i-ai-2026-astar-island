"""Submission validation and serialization utilities."""

from __future__ import annotations

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
