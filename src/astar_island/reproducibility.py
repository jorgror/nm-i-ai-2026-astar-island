"""Helpers for reproducibility metadata in evaluation artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .round_data import RoundRecord


def build_round_dataset_fingerprint(
    *,
    rounds: list[RoundRecord],
    logs_root: str | Path,
) -> dict[str, object]:
    """Build a stable fingerprint for a concrete round evaluation set."""
    ordered = sorted(
        rounds,
        key=lambda row: (
            row.round_number is None,
            int(row.round_number or 0),
            row.round_id,
        ),
    )
    rows = [
        (
            str(record.round_id),
            int(record.round_number) if record.round_number is not None else None,
            str(record.status) if record.status is not None else None,
            int(record.map_width),
            int(record.map_height),
            int(record.seeds_count),
        )
        for record in ordered
    ]
    canonical = "\n".join(
        f"{row_id}|{round_number}|{status}|{width}|{height}|{seeds}"
        for row_id, round_number, status, width, height, seeds in rows
    )
    fingerprint_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    return {
        "fingerprint_version": 1,
        "logs_root": str(Path(logs_root).resolve()),
        "round_count": len(rows),
        "round_ids": [row_id for row_id, *_ in rows],
        "round_numbers": [round_number for _, round_number, *_ in rows],
        "statuses": sorted(
            {
                status
                for _, _, status, _, _, _ in rows
                if status is not None
            }
        ),
        "sha256": fingerprint_sha256,
    }
