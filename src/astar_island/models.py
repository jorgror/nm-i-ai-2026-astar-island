"""Canonical data models for round and analysis payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

Grid2D = list[list[int]]
Tensor3D = list[list[list[float]]]


@dataclass(slots=True)
class Settlement:
    x: int
    y: int
    has_port: bool
    alive: bool


@dataclass(slots=True)
class SeedInitialState:
    grid: Grid2D
    settlements: list[Settlement]


@dataclass(slots=True)
class RoundDetail:
    round_id: str
    round_number: int | None
    status: str | None
    map_width: int
    map_height: int
    seeds_count: int
    initial_states: list[SeedInitialState]
    raw: dict[str, Any]


@dataclass(slots=True)
class AnalysisSeedData:
    prediction: Tensor3D | None
    ground_truth: Tensor3D
    score: float | None
    width: int
    height: int
    initial_grid: Grid2D | None
    raw: dict[str, Any]


@dataclass(slots=True)
class SeedScoreResult:
    seed_index: int
    weighted_kl: float
    score: float
    official_score: float | None
    score_diff: float | None
    source_path: str | None
    missing_submission: bool


@dataclass(slots=True)
class RoundScoreResult:
    round_id: str
    per_seed: list[SeedScoreResult]
    round_score: float
    official_round_score: float | None
    round_score_diff: float | None


@dataclass(slots=True)
class ValidationResult:
    ok: bool
    errors: list[str]
