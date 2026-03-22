"""Small spatial baseline model (local patch softmax) with LOO evaluation."""

from __future__ import annotations

import csv
import json
import math
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

from .models import SeedInitialState, Tensor3D
from .priors import baseline_prior_from_initial_grid
from .round_data import LeaveOneRoundOutSplit, load_leave_one_round_out
from .scoring import cell_entropy, score_from_weighted_kl, score_round, score_seed
from .submission import floor_and_normalize

NUM_CLASSES = 6
OCEAN_TERRAIN = 10
PAD_CLASS = 0

GRID_TO_CLASS = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    10: 0,
    11: 0,
}


@dataclass(slots=True)
class BaselineCConfig:
    patch_radius: int = 1
    learning_rate: float = 0.04
    epochs: int = 3
    l2: float = 1e-4
    samples_per_epoch: int = 12000
    max_cells_per_seed: int | None = 900
    entropy_weight_power: float = 1.0
    min_entropy_weight: float = 0.02
    probability_floor: float = 1e-4
    random_seed: int = 7


@dataclass(slots=True)
class TrainingEpochMetric:
    epoch: int
    learning_rate: float
    weighted_kl: float
    weighted_cross_entropy: float
    round_score: float
    sample_count: int


@dataclass(slots=True)
class SpatialSoftmaxModel:
    feature_names: list[str]
    weights: list[list[float]]
    biases: list[float]
    patch_radius: int

    @property
    def num_classes(self) -> int:
        return len(self.weights)

    @property
    def num_features(self) -> int:
        return len(self.feature_names)


@dataclass(slots=True)
class SeedSpatialResult:
    round_id: str
    round_number: int | None
    seed_index: int
    weighted_kl_baseline_c: float
    score_baseline_c: float
    weighted_kl_prior_a: float
    score_prior_a: float
    score_gain_vs_prior_a: float


@dataclass(slots=True)
class RoundSpatialResult:
    round_id: str
    round_number: int | None
    seeds_evaluated: int
    round_score_baseline_c: float
    round_score_prior_a: float
    score_gain_vs_prior_a: float


@dataclass(slots=True)
class BaselineCReport:
    config: BaselineCConfig
    seed_results: list[SeedSpatialResult]
    round_results: list[RoundSpatialResult]
    mean_seed_score_baseline_c: float
    mean_seed_score_prior_a: float
    mean_seed_gain_vs_prior_a: float
    mean_round_score_baseline_c: float
    mean_round_score_prior_a: float
    mean_round_gain_vs_prior_a: float
    mean_training_weighted_kl_by_epoch: list[float]
    mean_training_round_score_by_epoch: list[float]

    def write(self, output_dir: str | Path) -> dict[str, str]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        seed_csv = out_dir / "loo_seed_results.csv"
        round_csv = out_dir / "loo_round_results.csv"
        summary_json = out_dir / "summary.json"

        _write_seed_csv(seed_csv, self.seed_results)
        _write_round_csv(round_csv, self.round_results)
        summary_json.write_text(
            json.dumps(
                {
                    "config": asdict(self.config),
                    "mean_seed_score_baseline_c": self.mean_seed_score_baseline_c,
                    "mean_seed_score_prior_a": self.mean_seed_score_prior_a,
                    "mean_seed_gain_vs_prior_a": self.mean_seed_gain_vs_prior_a,
                    "mean_round_score_baseline_c": self.mean_round_score_baseline_c,
                    "mean_round_score_prior_a": self.mean_round_score_prior_a,
                    "mean_round_gain_vs_prior_a": self.mean_round_gain_vs_prior_a,
                    "mean_training_weighted_kl_by_epoch": self.mean_training_weighted_kl_by_epoch,
                    "mean_training_round_score_by_epoch": self.mean_training_round_score_by_epoch,
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


def evaluate_baseline_c_leave_one_round_out(
    *,
    logs_root: str | Path,
    config: BaselineCConfig | None = None,
    strict: bool = True,
) -> BaselineCReport:
    cfg = config or BaselineCConfig()
    splits = load_leave_one_round_out(logs_root, include_replays=False, strict=strict)
    rng = random.Random(cfg.random_seed)

    feature_cache: dict[tuple[str, int], list[list[list[float]]]] = {}
    seed_results: list[SeedSpatialResult] = []
    round_results: list[RoundSpatialResult] = []
    split_training_history: list[list[TrainingEpochMetric]] = []

    for split in splits:
        model, training_history = _train_for_split(split, cfg, feature_cache, rng)
        split_training_history.append(training_history)
        validation_round = split.validation_round

        c_scores: list[float] = []
        a_scores: list[float] = []
        seeds_evaluated = 0

        for seed in validation_round.seeds:
            if seed.initial_state is None or seed.analysis is None:
                if strict:
                    raise ValueError(
                        f"Missing initial_state/analysis for round={validation_round.round_id} "
                        f"seed={seed.seed_index}"
                    )
                continue

            features_grid = _seed_features_cached(
                feature_cache=feature_cache,
                round_id=validation_round.round_id,
                seed_index=seed.seed_index,
                state=seed.initial_state,
                patch_radius=cfg.patch_radius,
            )
            pred_c = predict_with_spatial_model(
                model,
                features_grid=features_grid,
                probability_floor=cfg.probability_floor,
            )
            wkl_c, score_c = score_seed(seed.analysis.ground_truth, pred_c)

            pred_a = baseline_prior_from_initial_grid(
                seed.initial_state.grid,
                settlements=seed.initial_state.settlements,
            )
            wkl_a, score_a = score_seed(seed.analysis.ground_truth, pred_a)

            c_scores.append(score_c)
            a_scores.append(score_a)
            seeds_evaluated += 1
            seed_results.append(
                SeedSpatialResult(
                    round_id=validation_round.round_id,
                    round_number=validation_round.round_number,
                    seed_index=seed.seed_index,
                    weighted_kl_baseline_c=wkl_c,
                    score_baseline_c=score_c,
                    weighted_kl_prior_a=wkl_a,
                    score_prior_a=score_a,
                    score_gain_vs_prior_a=score_c - score_a,
                )
            )

        if seeds_evaluated == 0:
            continue

        round_score_c = score_round(c_scores, expected_seeds=seeds_evaluated)
        round_score_a = score_round(a_scores, expected_seeds=seeds_evaluated)
        round_results.append(
            RoundSpatialResult(
                round_id=validation_round.round_id,
                round_number=validation_round.round_number,
                seeds_evaluated=seeds_evaluated,
                round_score_baseline_c=round_score_c,
                round_score_prior_a=round_score_a,
                score_gain_vs_prior_a=round_score_c - round_score_a,
            )
        )

    mean_seed_c = _mean([row.score_baseline_c for row in seed_results])
    mean_seed_a = _mean([row.score_prior_a for row in seed_results])
    mean_round_c = _mean([row.round_score_baseline_c for row in round_results])
    mean_round_a = _mean([row.round_score_prior_a for row in round_results])
    mean_training_weighted_kl_by_epoch = _aggregate_epoch_metric(
        split_training_history, metric_name="weighted_kl"
    )
    mean_training_round_score_by_epoch = _aggregate_epoch_metric(
        split_training_history, metric_name="round_score"
    )

    return BaselineCReport(
        config=cfg,
        seed_results=seed_results,
        round_results=round_results,
        mean_seed_score_baseline_c=mean_seed_c,
        mean_seed_score_prior_a=mean_seed_a,
        mean_seed_gain_vs_prior_a=mean_seed_c - mean_seed_a,
        mean_round_score_baseline_c=mean_round_c,
        mean_round_score_prior_a=mean_round_a,
        mean_round_gain_vs_prior_a=mean_round_c - mean_round_a,
        mean_training_weighted_kl_by_epoch=mean_training_weighted_kl_by_epoch,
        mean_training_round_score_by_epoch=mean_training_round_score_by_epoch,
    )


def _train_for_split(
    split: LeaveOneRoundOutSplit,
    cfg: BaselineCConfig,
    feature_cache: dict[tuple[str, int], list[list[list[float]]]],
    rng: random.Random,
) -> tuple[SpatialSoftmaxModel, list[TrainingEpochMetric]]:
    features: list[list[float]] = []
    targets: list[list[float]] = []

    for round_record in split.training_rounds:
        for seed in round_record.seeds:
            if seed.initial_state is None or seed.analysis is None:
                continue

            features_grid = _seed_features_cached(
                feature_cache=feature_cache,
                round_id=round_record.round_id,
                seed_index=seed.seed_index,
                state=seed.initial_state,
                patch_radius=cfg.patch_radius,
            )
            coords = _sample_coords(
                width=seed.analysis.width,
                height=seed.analysis.height,
                max_cells=cfg.max_cells_per_seed,
                rng=rng,
            )
            for x, y in coords:
                features.append(features_grid[y][x])
                targets.append(seed.analysis.ground_truth[y][x])

    return train_spatial_softmax_model(
        features=features,
        targets=targets,
        config=cfg,
        rng=rng,
        return_history=True,
    )


def train_spatial_softmax_model(
    *,
    features: list[list[float]],
    targets: list[list[float]],
    config: BaselineCConfig,
    rng: random.Random,
    return_history: bool = False,
) -> tuple[SpatialSoftmaxModel, list[TrainingEpochMetric]] | SpatialSoftmaxModel:
    if not features or not targets:
        raise ValueError("Need non-empty features and targets")
    if len(features) != len(targets):
        raise ValueError("features/targets length mismatch")
    if config.patch_radius < 1:
        raise ValueError("patch_radius must be >= 1")

    num_features = len(features[0])
    num_classes = len(targets[0])
    weights = [[0.0 for _ in range(num_features)] for _ in range(num_classes)]
    biases = [0.0 for _ in range(num_classes)]
    indices = list(range(len(features)))
    updates_per_epoch = min(config.samples_per_epoch, len(indices))
    sample_weights = _entropy_weights_for_targets(
        targets,
        power=config.entropy_weight_power,
        min_weight=config.min_entropy_weight,
    )
    history: list[TrainingEpochMetric] = []

    for epoch in range(config.epochs):
        rng.shuffle(indices)
        lr = config.learning_rate * (0.85**epoch)
        epoch_indices = indices[:updates_per_epoch]
        for idx in epoch_indices:
            x_vec = features[idx]
            y_true = targets[idx]
            ent_weight = sample_weights[idx]

            logits = [
                _dot(weights[class_idx], x_vec) + biases[class_idx]
                for class_idx in range(num_classes)
            ]
            probs = _softmax(logits)

            for class_idx in range(num_classes):
                diff = ent_weight * (probs[class_idx] - y_true[class_idx])
                if diff == 0.0:
                    continue
                row = weights[class_idx]
                for feat_idx, value in enumerate(x_vec):
                    row[feat_idx] -= lr * ((diff * value) + (config.l2 * row[feat_idx]))
                biases[class_idx] -= lr * diff

        history.append(
            _summarize_epoch(
                epoch=epoch + 1,
                learning_rate=lr,
                indices=epoch_indices,
                features=features,
                targets=targets,
                sample_weights=sample_weights,
                weights=weights,
                biases=biases,
                probability_floor=config.probability_floor,
            )
        )

    model = SpatialSoftmaxModel(
        feature_names=spatial_feature_names(config.patch_radius),
        weights=weights,
        biases=biases,
        patch_radius=config.patch_radius,
    )
    if return_history:
        return model, history
    return model


def predict_with_spatial_model(
    model: SpatialSoftmaxModel,
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


def spatial_feature_names(patch_radius: int) -> list[str]:
    if patch_radius < 1:
        raise ValueError("patch_radius must be >= 1")
    names = [
        "bias",
        "x_norm",
        "y_norm",
        "edge_norm",
        "coastal",
        "near_settlement",
        "near_ocean",
        "settlement_density",
    ]
    for dy in range(-patch_radius, patch_radius + 1):
        for dx in range(-patch_radius, patch_radius + 1):
            for cls in range(NUM_CLASSES):
                names.append(f"patch_dy{dy}_dx{dx}_class{cls}")
    return names


def build_seed_spatial_feature_grid(
    state: SeedInitialState,
    *,
    patch_radius: int = 1,
) -> list[list[list[float]]]:
    if patch_radius < 1:
        raise ValueError("patch_radius must be >= 1")
    grid = state.grid
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0

    class_grid = [[GRID_TO_CLASS.get(int(value), 0) for value in row] for row in grid]
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
            edge_dist = min(x, y, width - 1 - x, height - 1 - y)
            edge_norm = min(1.0, max(0.0, edge_dist / float(max_edge)))
            d_settle = min(dist_settlement[y][x], 12)
            d_ocean = min(dist_ocean[y][x], 12)

            vec = [
                1.0,
                x / float(max_x),
                y / float(max_y),
                edge_norm,
                1.0 if _is_coastal(grid, x=x, y=y) else 0.0,
                math.exp(-d_settle / 3.0),
                math.exp(-d_ocean / 3.0),
                _settlement_density(settlements, x=x, y=y, radius=3),
            ]

            for dy in range(-patch_radius, patch_radius + 1):
                ny = y + dy
                for dx in range(-patch_radius, patch_radius + 1):
                    nx = x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        cls = class_grid[ny][nx]
                    else:
                        cls = PAD_CLASS
                    one_hot = [0.0] * NUM_CLASSES
                    one_hot[cls] = 1.0
                    vec.extend(one_hot)
            out_row.append(vec)
        features.append(out_row)
    return features


def _seed_features_cached(
    *,
    feature_cache: dict[tuple[str, int], list[list[list[float]]]],
    round_id: str,
    seed_index: int,
    state: SeedInitialState,
    patch_radius: int,
) -> list[list[list[float]]]:
    key = (round_id, seed_index)
    cached = feature_cache.get(key)
    if cached is not None:
        return cached
    computed = build_seed_spatial_feature_grid(state, patch_radius=patch_radius)
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


def _collect_settlement_points(state: SeedInitialState) -> list[tuple[int, int]]:
    points: set[tuple[int, int]] = set()
    for y, row in enumerate(state.grid):
        for x, value in enumerate(row):
            if int(value) in (1, 2):
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
    return min(1.0, count / 6.0)


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


def _write_seed_csv(path: Path, rows: list[SeedSpatialResult]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "round_id",
                "round_number",
                "seed_index",
                "weighted_kl_baseline_c",
                "score_baseline_c",
                "weighted_kl_prior_a",
                "score_prior_a",
                "score_gain_vs_prior_a",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _write_round_csv(path: Path, rows: list[RoundSpatialResult]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "round_id",
                "round_number",
                "seeds_evaluated",
                "round_score_baseline_c",
                "round_score_prior_a",
                "score_gain_vs_prior_a",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _softmax(logits: list[float]) -> list[float]:
    max_logit = max(logits)
    exps = [math.exp(value - max_logit) for value in logits]
    total = sum(exps)
    if total <= 0.0:
        return [1.0 / len(logits)] * len(logits)
    return [value / total for value in exps]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _summarize_epoch(
    *,
    epoch: int,
    learning_rate: float,
    indices: list[int],
    features: list[list[float]],
    targets: list[list[float]],
    sample_weights: list[float],
    weights: list[list[float]],
    biases: list[float],
    probability_floor: float,
) -> TrainingEpochMetric:
    weighted_kl_sum = 0.0
    weighted_cross_entropy_sum = 0.0
    weight_sum = 0.0

    for idx in indices:
        x_vec = features[idx]
        y_true = targets[idx]
        ent_weight = sample_weights[idx]
        logits = [
            _dot(weights[class_idx], x_vec) + biases[class_idx]
            for class_idx in range(len(weights))
        ]
        probs = _softmax(logits)
        probs = _floor_distribution(probs, floor=probability_floor)
        kl, cross_entropy = _kl_and_cross_entropy(y_true, probs)

        weighted_kl_sum += ent_weight * kl
        weighted_cross_entropy_sum += ent_weight * cross_entropy
        weight_sum += ent_weight

    if weight_sum <= 0.0:
        weighted_kl_value = 0.0
        weighted_cross_entropy_value = 0.0
    else:
        weighted_kl_value = weighted_kl_sum / weight_sum
        weighted_cross_entropy_value = weighted_cross_entropy_sum / weight_sum

    return TrainingEpochMetric(
        epoch=epoch,
        learning_rate=learning_rate,
        weighted_kl=weighted_kl_value,
        weighted_cross_entropy=weighted_cross_entropy_value,
        round_score=score_from_weighted_kl(weighted_kl_value),
        sample_count=len(indices),
    )


def _entropy_weights_for_targets(
    targets: list[list[float]],
    *,
    power: float,
    min_weight: float,
) -> list[float]:
    if power <= 0.0:
        raise ValueError("entropy_weight_power must be > 0")
    if min_weight < 0.0:
        raise ValueError("min_entropy_weight must be >= 0")

    weights: list[float] = []
    for target in targets:
        entropy = cell_entropy(target)
        scaled = entropy**power
        weights.append(max(min_weight, scaled))
    return weights


def _floor_distribution(probs: list[float], *, floor: float) -> list[float]:
    if floor <= 0.0:
        return probs
    floored = [max(value, floor) for value in probs]
    total = sum(floored)
    if total <= 0.0:
        return [1.0 / len(probs)] * len(probs)
    return [value / total for value in floored]


def _kl_and_cross_entropy(y_true: list[float], probs: list[float]) -> tuple[float, float]:
    kl = 0.0
    cross_entropy = 0.0
    for truth, pred in zip(y_true, probs):
        if truth <= 0.0:
            continue
        cross_entropy -= truth * math.log(pred)
        kl += truth * math.log(truth / pred)
    return kl, cross_entropy


def _aggregate_epoch_metric(
    histories: list[list[TrainingEpochMetric]],
    *,
    metric_name: str,
) -> list[float]:
    if not histories:
        return []
    max_epochs = max((len(history) for history in histories), default=0)
    output: list[float] = []
    for epoch_idx in range(max_epochs):
        values: list[float] = []
        for history in histories:
            if epoch_idx >= len(history):
                continue
            values.append(float(getattr(history[epoch_idx], metric_name)))
        output.append(_mean(values))
    return output


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))
