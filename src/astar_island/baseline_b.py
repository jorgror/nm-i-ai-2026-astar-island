"""Feature-based non-neural baseline (multinomial logistic regression)."""

from __future__ import annotations

import csv
import json
import math
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .models import SeedInitialState, Tensor3D
from .priors import baseline_prior_from_initial_grid
from .round_data import LeaveOneRoundOutSplit, load_leave_one_round_out
from .scoring import score_round, score_seed
from .submission import floor_and_normalize

CLASS_EMPTY = 0
CLASS_SETTLEMENT = 1
CLASS_PORT = 2
CLASS_RUIN = 3
CLASS_FOREST = 4
CLASS_MOUNTAIN = 5

OCEAN_TERRAIN = 10

FEATURE_NAMES = [
    "bias",
    "x_norm",
    "y_norm",
    "edge_norm",
    "is_class0",
    "is_settlement",
    "is_port",
    "is_ruin",
    "is_forest",
    "is_mountain",
    "coastal",
    "near_settlement",
    "near_ocean",
    "settlement_density",
    "forest_adj",
    "mountain_adj",
]


@dataclass(slots=True)
class BaselineBConfig:
    learning_rate: float = 0.05
    epochs: int = 5
    l2: float = 1e-4
    samples_per_epoch: int = 25000
    max_cells_per_seed: int | None = 1000
    probability_floor: float = 1e-4
    random_seed: int = 7


@dataclass(slots=True)
class LogisticRegressionModel:
    feature_names: list[str]
    weights: list[list[float]]
    biases: list[float]

    @property
    def num_classes(self) -> int:
        return len(self.weights)

    @property
    def num_features(self) -> int:
        return len(self.feature_names)


@dataclass(slots=True)
class SeedBaselineResult:
    round_id: str
    round_number: int | None
    seed_index: int
    weighted_kl_baseline_b: float
    score_baseline_b: float
    weighted_kl_prior_a: float
    score_prior_a: float
    score_gain_vs_prior_a: float


@dataclass(slots=True)
class RoundBaselineResult:
    round_id: str
    round_number: int | None
    seeds_evaluated: int
    round_score_baseline_b: float
    round_score_prior_a: float
    score_gain_vs_prior_a: float


@dataclass(slots=True)
class BaselineBReport:
    config: BaselineBConfig
    seed_results: list[SeedBaselineResult]
    round_results: list[RoundBaselineResult]
    mean_seed_score_baseline_b: float
    mean_seed_score_prior_a: float
    mean_seed_gain_vs_prior_a: float
    mean_round_score_baseline_b: float
    mean_round_score_prior_a: float
    mean_round_gain_vs_prior_a: float

    def write(self, output_dir: str | Path) -> dict[str, str]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        seed_csv = out_dir / "loo_seed_results.csv"
        round_csv = out_dir / "loo_round_results.csv"
        summary_json = out_dir / "summary.json"

        self._write_seed_csv(seed_csv)
        self._write_round_csv(round_csv)
        summary_json.write_text(
            json.dumps(
                {
                    "config": asdict(self.config),
                    "mean_seed_score_baseline_b": self.mean_seed_score_baseline_b,
                    "mean_seed_score_prior_a": self.mean_seed_score_prior_a,
                    "mean_seed_gain_vs_prior_a": self.mean_seed_gain_vs_prior_a,
                    "mean_round_score_baseline_b": self.mean_round_score_baseline_b,
                    "mean_round_score_prior_a": self.mean_round_score_prior_a,
                    "mean_round_gain_vs_prior_a": self.mean_round_gain_vs_prior_a,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return {
            "seed_csv": str(seed_csv),
            "round_csv": str(round_csv),
            "summary_json": str(summary_json),
        }

    def _write_seed_csv(self, path: Path) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "round_id",
                    "round_number",
                    "seed_index",
                    "weighted_kl_baseline_b",
                    "score_baseline_b",
                    "weighted_kl_prior_a",
                    "score_prior_a",
                    "score_gain_vs_prior_a",
                ],
            )
            writer.writeheader()
            for row in self.seed_results:
                writer.writerow(asdict(row))

    def _write_round_csv(self, path: Path) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "round_id",
                    "round_number",
                    "seeds_evaluated",
                    "round_score_baseline_b",
                    "round_score_prior_a",
                    "score_gain_vs_prior_a",
                ],
            )
            writer.writeheader()
            for row in self.round_results:
                writer.writerow(asdict(row))


def evaluate_baseline_b_leave_one_round_out(
    *,
    logs_root: str | Path,
    config: BaselineBConfig | None = None,
    strict: bool = True,
) -> BaselineBReport:
    cfg = config or BaselineBConfig()
    splits = load_leave_one_round_out(logs_root, include_replays=False, strict=strict)

    rng = random.Random(cfg.random_seed)
    feature_cache: dict[tuple[str, int], list[list[list[float]]]] = {}
    seed_results: list[SeedBaselineResult] = []
    round_results: list[RoundBaselineResult] = []

    for split in splits:
        model = _train_for_split(split, cfg, feature_cache, rng)
        round_b_scores: list[float] = []
        round_a_scores: list[float] = []
        seeds_evaluated = 0

        validation_round = split.validation_round
        for seed in validation_round.seeds:
            if seed.analysis is None or seed.initial_state is None:
                if strict:
                    raise ValueError(
                        f"Missing analysis/initial_state for round={validation_round.round_id} "
                        f"seed={seed.seed_index}"
                    )
                continue

            features = _seed_features_cached(
                feature_cache=feature_cache,
                round_id=validation_round.round_id,
                seed_index=seed.seed_index,
                state=seed.initial_state,
            )
            pred_b = predict_with_model(
                model,
                features_grid=features,
                probability_floor=cfg.probability_floor,
            )
            wkl_b, score_b = score_seed(seed.analysis.ground_truth, pred_b)

            pred_a = baseline_prior_from_initial_grid(
                seed.initial_state.grid,
                settlements=seed.initial_state.settlements,
            )
            wkl_a, score_a = score_seed(seed.analysis.ground_truth, pred_a)

            round_b_scores.append(score_b)
            round_a_scores.append(score_a)
            seeds_evaluated += 1

            seed_results.append(
                SeedBaselineResult(
                    round_id=validation_round.round_id,
                    round_number=validation_round.round_number,
                    seed_index=seed.seed_index,
                    weighted_kl_baseline_b=wkl_b,
                    score_baseline_b=score_b,
                    weighted_kl_prior_a=wkl_a,
                    score_prior_a=score_a,
                    score_gain_vs_prior_a=score_b - score_a,
                )
            )

        if seeds_evaluated == 0:
            continue

        round_score_b = score_round(round_b_scores, expected_seeds=seeds_evaluated)
        round_score_a = score_round(round_a_scores, expected_seeds=seeds_evaluated)
        round_results.append(
            RoundBaselineResult(
                round_id=validation_round.round_id,
                round_number=validation_round.round_number,
                seeds_evaluated=seeds_evaluated,
                round_score_baseline_b=round_score_b,
                round_score_prior_a=round_score_a,
                score_gain_vs_prior_a=round_score_b - round_score_a,
            )
        )

    mean_seed_b = _mean([row.score_baseline_b for row in seed_results])
    mean_seed_a = _mean([row.score_prior_a for row in seed_results])
    mean_round_b = _mean([row.round_score_baseline_b for row in round_results])
    mean_round_a = _mean([row.round_score_prior_a for row in round_results])

    return BaselineBReport(
        config=cfg,
        seed_results=seed_results,
        round_results=round_results,
        mean_seed_score_baseline_b=mean_seed_b,
        mean_seed_score_prior_a=mean_seed_a,
        mean_seed_gain_vs_prior_a=mean_seed_b - mean_seed_a,
        mean_round_score_baseline_b=mean_round_b,
        mean_round_score_prior_a=mean_round_a,
        mean_round_gain_vs_prior_a=mean_round_b - mean_round_a,
    )


def _train_for_split(
    split: LeaveOneRoundOutSplit,
    cfg: BaselineBConfig,
    feature_cache: dict[tuple[str, int], list[list[list[float]]]],
    rng: random.Random,
) -> LogisticRegressionModel:
    features: list[list[float]] = []
    targets: list[list[float]] = []

    for round_record in split.training_rounds:
        for seed in round_record.seeds:
            if seed.analysis is None or seed.initial_state is None:
                continue
            feat_grid = _seed_features_cached(
                feature_cache=feature_cache,
                round_id=round_record.round_id,
                seed_index=seed.seed_index,
                state=seed.initial_state,
            )
            coords = _sample_coords(
                width=seed.analysis.width,
                height=seed.analysis.height,
                max_cells=cfg.max_cells_per_seed,
                rng=rng,
            )
            gt = seed.analysis.ground_truth
            for x, y in coords:
                features.append(feat_grid[y][x])
                targets.append(gt[y][x])

    return train_multinomial_logistic_regression(
        features=features,
        targets=targets,
        config=cfg,
        rng=rng,
    )


def train_multinomial_logistic_regression(
    *,
    features: list[list[float]],
    targets: list[list[float]],
    config: BaselineBConfig,
    rng: random.Random,
) -> LogisticRegressionModel:
    if not features or not targets:
        raise ValueError("Need non-empty features and targets")
    if len(features) != len(targets):
        raise ValueError("features/targets length mismatch")

    num_features = len(features[0])
    num_classes = len(targets[0])
    if num_classes <= 1:
        raise ValueError("Expected at least 2 classes")

    weights = [[0.0 for _ in range(num_features)] for _ in range(num_classes)]
    biases = [0.0 for _ in range(num_classes)]
    indices = list(range(len(features)))
    updates_per_epoch = min(config.samples_per_epoch, len(indices))

    for epoch in range(config.epochs):
        rng.shuffle(indices)
        lr = config.learning_rate * (0.85**epoch)
        for idx in indices[:updates_per_epoch]:
            x_vec = features[idx]
            y_true = targets[idx]

            logits = [
                _dot(weights[class_idx], x_vec) + biases[class_idx]
                for class_idx in range(num_classes)
            ]
            probs = _softmax(logits)

            for class_idx in range(num_classes):
                diff = probs[class_idx] - y_true[class_idx]
                if diff == 0.0:
                    continue
                row = weights[class_idx]
                for feat_idx, value in enumerate(x_vec):
                    row[feat_idx] -= lr * ((diff * value) + (config.l2 * row[feat_idx]))
                biases[class_idx] -= lr * diff

    return LogisticRegressionModel(
        feature_names=list(FEATURE_NAMES),
        weights=weights,
        biases=biases,
    )


def predict_with_model(
    model: LogisticRegressionModel,
    *,
    features_grid: list[list[list[float]]],
    probability_floor: float = 1e-4,
) -> Tensor3D:
    output: Tensor3D = []
    for row in features_grid:
        out_row: list[list[float]] = []
        for x_vec in row:
            logits = [
                _dot(model.weights[class_idx], x_vec) + model.biases[class_idx]
                for class_idx in range(model.num_classes)
            ]
            out_row.append(_softmax(logits))
        output.append(out_row)

    if probability_floor > 0.0:
        return floor_and_normalize(output, floor=probability_floor)
    return output


def build_seed_feature_grid(state: SeedInitialState) -> list[list[list[float]]]:
    grid = state.grid
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0

    settlements = _collect_settlement_points(state)
    oceans = _collect_points_by_terrain(grid, terrain=OCEAN_TERRAIN)

    dist_settlement = _distance_map(width=width, height=height, points=settlements)
    dist_ocean = _distance_map(width=width, height=height, points=oceans)

    max_x = max(1, width - 1)
    max_y = max(1, height - 1)
    max_edge = max(1, min(width, height) // 2)

    features: list[list[list[float]]] = []
    for y in range(height):
        out_row: list[list[float]] = []
        for x in range(width):
            terrain = int(grid[y][x])
            is_class0 = 1.0 if terrain in (0, 10, 11) else 0.0
            is_settlement = 1.0 if terrain == 1 else 0.0
            is_port = 1.0 if terrain == 2 else 0.0
            is_ruin = 1.0 if terrain == 3 else 0.0
            is_forest = 1.0 if terrain == 4 else 0.0
            is_mountain = 1.0 if terrain == 5 else 0.0

            edge_dist = min(x, y, width - 1 - x, height - 1 - y)
            edge_norm = edge_dist / float(max_edge)
            edge_norm = max(0.0, min(1.0, edge_norm))

            d_settle = min(dist_settlement[y][x], 12)
            d_ocean = min(dist_ocean[y][x], 12)

            near_settlement = math.exp(-d_settle / 3.0)
            near_ocean = math.exp(-d_ocean / 3.0)

            out_row.append(
                [
                    1.0,
                    x / float(max_x),
                    y / float(max_y),
                    edge_norm,
                    is_class0,
                    is_settlement,
                    is_port,
                    is_ruin,
                    is_forest,
                    is_mountain,
                    1.0 if _is_coastal(grid, x=x, y=y) else 0.0,
                    near_settlement,
                    near_ocean,
                    _settlement_density(settlements, x=x, y=y, radius=3),
                    _adjacency_ratio(grid, x=x, y=y, terrain=4),
                    _adjacency_ratio(grid, x=x, y=y, terrain=5),
                ]
            )
        features.append(out_row)

    return features


def _seed_features_cached(
    *,
    feature_cache: dict[tuple[str, int], list[list[list[float]]]],
    round_id: str,
    seed_index: int,
    state: SeedInitialState,
) -> list[list[list[float]]]:
    key = (round_id, seed_index)
    cached = feature_cache.get(key)
    if cached is not None:
        return cached
    computed = build_seed_feature_grid(state)
    feature_cache[key] = computed
    return computed


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


def _distance_map(*, width: int, height: int, points: list[tuple[int, int]]) -> list[list[int]]:
    inf = 10**9
    dist = [[inf for _ in range(width)] for _ in range(height)]
    if not points:
        return dist

    q: deque[tuple[int, int]] = deque()
    for x, y in points:
        if 0 <= x < width and 0 <= y < height and dist[y][x] != 0:
            dist[y][x] = 0
            q.append((x, y))

    while q:
        x, y = q.popleft()
        base = dist[y][x]
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            nx = x + dx
            ny = y + dy
            if ny < 0 or ny >= height or nx < 0 or nx >= width:
                continue
            candidate = base + 1
            if candidate < dist[ny][nx]:
                dist[ny][nx] = candidate
                q.append((nx, ny))
    return dist


def _collect_settlement_points(state: SeedInitialState) -> list[tuple[int, int]]:
    points: set[tuple[int, int]] = set()
    for y, row in enumerate(state.grid):
        for x, value in enumerate(row):
            terrain = int(value)
            if terrain in (1, 2):
                points.add((x, y))
    for settlement in state.settlements:
        if settlement.alive:
            points.add((int(settlement.x), int(settlement.y)))
    return sorted(points)


def _collect_points_by_terrain(grid: list[list[int]], *, terrain: int) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for y, row in enumerate(grid):
        for x, value in enumerate(row):
            if int(value) == terrain:
                points.append((x, y))
    return points


def _settlement_density(
    settlements: list[tuple[int, int]],
    *,
    x: int,
    y: int,
    radius: int,
) -> float:
    if radius <= 0 or not settlements:
        return 0.0
    count = 0
    for sx, sy in settlements:
        if abs(sx - x) + abs(sy - y) <= radius:
            count += 1
    # Soft cap to keep feature in [0,1] with useful dynamic range.
    return min(1.0, count / 6.0)


def _adjacency_ratio(grid: list[list[int]], *, x: int, y: int, terrain: int) -> float:
    matches = 0
    total = 0
    for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
        nx = x + dx
        ny = y + dy
        if ny < 0 or ny >= len(grid):
            continue
        if nx < 0 or nx >= len(grid[ny]):
            continue
        total += 1
        if int(grid[ny][nx]) == terrain:
            matches += 1
    if total == 0:
        return 0.0
    return matches / float(total)


def _is_coastal(grid: list[list[int]], *, x: int, y: int) -> bool:
    for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
        nx = x + dx
        ny = y + dy
        if ny < 0 or ny >= len(grid):
            continue
        if nx < 0 or nx >= len(grid[ny]):
            continue
        if int(grid[ny][nx]) == OCEAN_TERRAIN:
            return True
    return False


def _softmax(logits: list[float]) -> list[float]:
    max_logit = max(logits)
    exps = [math.exp(value - max_logit) for value in logits]
    total = sum(exps)
    if total <= 0.0:
        return [1.0 / len(logits)] * len(logits)
    return [value / total for value in exps]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))
