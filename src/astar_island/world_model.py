"""Step-14 world-model utilities built on top of Baseline B."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .baseline_b import (
    BaselineBConfig,
    LogisticRegressionModel,
    build_seed_feature_grid,
    predict_with_model,
    train_multinomial_logistic_regression,
)
from .models import SeedInitialState, Tensor3D
from .round_data import RoundRecord, load_round_dataset


@dataclass(slots=True)
class TrainedBaselineBWorldModel:
    model: LogisticRegressionModel
    config: BaselineBConfig
    rounds_used: int
    samples_used: int


class BaselineBWorldModelPredictor:
    """Callable seed predictor compatible with RoundLatentConditionalModel."""

    def __init__(
        self,
        *,
        model: LogisticRegressionModel,
        probability_floor: float = 1e-4,
    ) -> None:
        self.model = model
        self.probability_floor = float(probability_floor)
        self._feature_cache: dict[int, list[list[list[float]]]] = {}

    def __call__(self, seed_initial_state: SeedInitialState, seed_index: int) -> Tensor3D:
        del seed_index  # Seed index is unused because Baseline B is seed-state driven.
        cache_key = id(seed_initial_state)
        features_grid = self._feature_cache.get(cache_key)
        if features_grid is None:
            features_grid = build_seed_feature_grid(seed_initial_state)
            self._feature_cache[cache_key] = features_grid
        return predict_with_model(
            self.model,
            features_grid=features_grid,
            probability_floor=self.probability_floor,
        )


def train_baseline_b_world_model_from_logs(
    *,
    logs_root: str | Path,
    config: BaselineBConfig | None = None,
    strict: bool = False,
    exclude_round_id: str | None = None,
) -> TrainedBaselineBWorldModel:
    rounds = load_round_dataset(
        logs_root,
        include_replays=False,
        strict=strict,
    )
    return train_baseline_b_world_model_from_rounds(
        rounds=rounds,
        config=config,
        exclude_round_id=exclude_round_id,
    )


def train_baseline_b_world_model_from_rounds(
    *,
    rounds: Iterable[RoundRecord],
    config: BaselineBConfig | None = None,
    exclude_round_id: str | None = None,
) -> TrainedBaselineBWorldModel:
    cfg = config or BaselineBConfig()
    rng = random.Random(cfg.random_seed)

    features: list[list[float]] = []
    targets: list[list[float]] = []
    rounds_used = 0

    for round_record in rounds:
        if exclude_round_id is not None and str(round_record.round_id) == str(exclude_round_id):
            continue
        round_used = False
        for seed in round_record.seeds:
            if seed.initial_state is None or seed.analysis is None:
                continue
            round_used = True
            feature_grid = build_seed_feature_grid(seed.initial_state)
            coords = _sample_coords(
                width=seed.analysis.width,
                height=seed.analysis.height,
                max_cells=cfg.max_cells_per_seed,
                rng=rng,
            )
            ground_truth = seed.analysis.ground_truth
            for x, y in coords:
                features.append(feature_grid[y][x])
                targets.append(ground_truth[y][x])
        if round_used:
            rounds_used += 1

    if not features:
        raise ValueError("No training samples available for Baseline B world model.")

    model = train_multinomial_logistic_regression(
        features=features,
        targets=targets,
        config=cfg,
        rng=rng,
    )
    return TrainedBaselineBWorldModel(
        model=model,
        config=cfg,
        rounds_used=rounds_used,
        samples_used=len(features),
    )


def _sample_coords(
    *,
    width: int,
    height: int,
    max_cells: int | None,
    rng: random.Random,
) -> list[tuple[int, int]]:
    coords = [(x, y) for y in range(height) for x in range(width)]
    if max_cells is None or max_cells <= 0 or max_cells >= len(coords):
        return coords
    return rng.sample(coords, max_cells)

