"""Parsing helpers for official Astar Island JSON payloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import AnalysisSeedData, RoundDetail, SeedInitialState, Settlement


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_round_detail(payload: dict[str, Any]) -> RoundDetail:
    round_id = str(payload["id"])
    map_width = int(payload["map_width"])
    map_height = int(payload["map_height"])
    seeds_count = int(payload.get("seeds_count", len(payload.get("initial_states", []))))

    initial_states_payload = payload.get("initial_states", [])
    initial_states: list[SeedInitialState] = []

    for seed_payload in initial_states_payload:
        grid = _parse_grid(seed_payload["grid"], expected_w=map_width, expected_h=map_height)
        settlements_payload = seed_payload.get("settlements", [])
        settlements = [
            Settlement(
                x=int(s["x"]),
                y=int(s["y"]),
                has_port=bool(s.get("has_port", False)),
                alive=bool(s.get("alive", True)),
            )
            for s in settlements_payload
        ]
        initial_states.append(SeedInitialState(grid=grid, settlements=settlements))

    return RoundDetail(
        round_id=round_id,
        round_number=_parse_optional_int(payload.get("round_number")),
        status=_parse_optional_str(payload.get("status")),
        map_width=map_width,
        map_height=map_height,
        seeds_count=seeds_count,
        initial_states=initial_states,
        raw=payload,
    )


def parse_analysis_seed(payload: dict[str, Any]) -> AnalysisSeedData:
    width = int(payload["width"])
    height = int(payload["height"])

    raw_prediction = payload.get("prediction")
    prediction = None
    if raw_prediction is not None:
        prediction = _parse_tensor(raw_prediction, expected_w=width, expected_h=height)
    ground_truth = _parse_tensor(payload["ground_truth"], expected_w=width, expected_h=height)

    initial_grid = None
    if payload.get("initial_grid") is not None:
        initial_grid = _parse_grid(payload["initial_grid"], expected_w=width, expected_h=height)

    raw_score = payload.get("score")
    score = float(raw_score) if raw_score is not None else None

    return AnalysisSeedData(
        prediction=prediction,
        ground_truth=ground_truth,
        score=score,
        width=width,
        height=height,
        initial_grid=initial_grid,
        raw=payload,
    )


def parse_round_score(payload: Any, round_id: str | None = None) -> float | None:
    """Extract round_score from either a row object or a list of /my-rounds rows."""
    if isinstance(payload, dict):
        val = payload.get("round_score")
        return float(val) if val is not None else None

    if isinstance(payload, list):
        if round_id is not None:
            for row in payload:
                if str(row.get("id")) == round_id:
                    val = row.get("round_score")
                    return float(val) if val is not None else None
        for row in payload:
            if row.get("round_score") is not None:
                return float(row["round_score"])

    return None


def _parse_grid(grid: Any, expected_w: int, expected_h: int) -> list[list[int]]:
    if not isinstance(grid, list):
        raise ValueError("Grid must be a list of rows")
    if len(grid) != expected_h:
        raise ValueError(f"Grid has {len(grid)} rows, expected {expected_h}")

    parsed: list[list[int]] = []
    for y, row in enumerate(grid):
        if not isinstance(row, list):
            raise ValueError(f"Grid row {y} is not a list")
        if len(row) != expected_w:
            raise ValueError(f"Grid row {y} has {len(row)} cols, expected {expected_w}")
        parsed.append([int(v) for v in row])
    return parsed


def _parse_tensor(tensor: Any, expected_w: int, expected_h: int) -> list[list[list[float]]]:
    if not isinstance(tensor, list):
        raise ValueError("Tensor must be a list of rows")
    if len(tensor) != expected_h:
        raise ValueError(f"Tensor has {len(tensor)} rows, expected {expected_h}")

    parsed: list[list[list[float]]] = []
    for y, row in enumerate(tensor):
        if not isinstance(row, list):
            raise ValueError(f"Tensor row {y} is not a list")
        if len(row) != expected_w:
            raise ValueError(f"Tensor row {y} has {len(row)} cols, expected {expected_w}")

        parsed_row: list[list[float]] = []
        for x, cell in enumerate(row):
            if not isinstance(cell, list):
                raise ValueError(f"Tensor cell ({y},{x}) is not a list")
            parsed_row.append([float(v) for v in cell])
        parsed.append(parsed_row)

    return parsed


def _parse_optional_int(val: Any) -> int | None:
    if val is None:
        return None
    return int(val)


def _parse_optional_str(val: Any) -> str | None:
    if val is None:
        return None
    return str(val)
