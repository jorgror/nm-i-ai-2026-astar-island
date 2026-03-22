"""Round fingerprint extraction and lightweight clustering utilities."""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .round_data import RoundRecord, RoundSeedRecord

Coord = tuple[int, int]

DEFAULT_FEATURES = [
    "settlement_survival_rate",
    "settlement_growth_rate",
    "port_retention_rate",
    "port_creation_rate",
    "ruin_frequency",
    "forest_reclaim_rate",
    "average_spread_radius",
    "takeover_conflict_rate",
    "owner_change_rate",
    "owner_consolidation",
    "dominant_owner_share",
    "survivor_population_gap",
    "survivor_food_gap",
    "survivor_wealth_gap",
    "survivor_defense_gap",
]

COMPACT_FEATURES = [
    "settlement_survival_rate",
    "settlement_growth_rate",
    "port_retention_rate",
    "port_creation_rate",
    "ruin_frequency",
    "average_spread_radius",
    "takeover_conflict_rate",
    "owner_change_rate",
    "owner_consolidation",
    "dominant_owner_share",
]

DYNAMICS_FEATURES = [
    "settlement_survival_rate",
    "settlement_growth_rate",
    "port_retention_rate",
    "port_creation_rate",
    "ruin_frequency",
    "average_spread_radius",
    "takeover_conflict_rate",
    "owner_change_rate",
    "owner_consolidation",
    "dominant_owner_share",
    "survivor_population_gap",
]


@dataclass(slots=True)
class SeedFingerprint:
    initial_settlements: float
    final_settlements: float
    settlement_survival_rate: float
    settlement_growth_rate: float
    port_retention_rate: float
    port_creation_rate: float
    ruin_frequency: float
    forest_reclaim_rate: float
    average_spread_radius: float
    takeover_conflict_rate: float
    owner_change_rate: float
    owner_consolidation: float
    dominant_owner_share: float
    survivor_population_gap: float
    survivor_food_gap: float
    survivor_wealth_gap: float
    survivor_defense_gap: float


@dataclass(slots=True)
class RoundFingerprint:
    round_id: str
    round_number: int | None
    seeds_used: int
    initial_settlements: float
    final_settlements: float
    settlement_survival_rate: float
    settlement_growth_rate: float
    port_retention_rate: float
    port_creation_rate: float
    ruin_frequency: float
    forest_reclaim_rate: float
    average_spread_radius: float
    takeover_conflict_rate: float
    owner_change_rate: float
    owner_consolidation: float
    dominant_owner_share: float
    survivor_population_gap: float
    survivor_food_gap: float
    survivor_wealth_gap: float
    survivor_defense_gap: float

    def feature_vector(self, feature_names: list[str]) -> list[float]:
        return [float(getattr(self, name)) for name in feature_names]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ClusterPoint:
    round_id: str
    cluster_id: int
    pc1: float
    pc2: float


@dataclass(slots=True)
class ClusteringResult:
    k: int
    feature_names: list[str]
    means: list[float]
    stds: list[float]
    assignments: list[ClusterPoint]
    inertia: float
    centroids_normalized: list[list[float]]
    centroids_raw: list[list[float]]
    elbow: list[tuple[int, float]]


@dataclass(slots=True)
class ArchetypeReport:
    fingerprints: list[RoundFingerprint]
    clustering: ClusteringResult

    def write(self, output_dir: str | Path) -> dict[str, str]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        fingerprints_json = out_dir / "round_fingerprints.json"
        clusters_json = out_dir / "clustering_summary.json"
        table_csv = out_dir / "round_fingerprint_table.csv"
        scatter_svg = out_dir / "cluster_scatter.svg"
        elbow_svg = out_dir / "cluster_elbow.svg"

        _write_json(
            fingerprints_json,
            {"round_fingerprints": [fp.as_dict() for fp in self.fingerprints]},
        )

        _write_json(
            clusters_json,
            {
                "k": self.clustering.k,
                "feature_names": self.clustering.feature_names,
                "means": self.clustering.means,
                "stds": self.clustering.stds,
                "inertia": self.clustering.inertia,
                "centroids_normalized": self.clustering.centroids_normalized,
                "centroids_raw": self.clustering.centroids_raw,
                "elbow": [
                    {"k": k, "inertia": inertia}
                    for k, inertia in self.clustering.elbow
                ],
                "assignments": [
                    {
                        "round_id": point.round_id,
                        "cluster_id": point.cluster_id,
                        "pc1": point.pc1,
                        "pc2": point.pc2,
                    }
                    for point in self.clustering.assignments
                ],
            },
        )

        _write_table_csv(
            table_csv,
            fingerprints=self.fingerprints,
            assignments=self.clustering.assignments,
            feature_names=self.clustering.feature_names,
        )

        _write_scatter_svg(
            scatter_svg,
            points=self.clustering.assignments,
            labels=_round_labels(self.fingerprints),
        )
        _write_elbow_svg(
            elbow_svg,
            elbow=self.clustering.elbow,
        )

        return {
            "fingerprints_json": str(fingerprints_json),
            "clusters_json": str(clusters_json),
            "table_csv": str(table_csv),
            "scatter_svg": str(scatter_svg),
            "elbow_svg": str(elbow_svg),
        }


def build_archetype_report(
    rounds: Iterable[RoundRecord],
    *,
    k: int = 4,
    max_k: int = 8,
    random_seed: int = 7,
    feature_names: list[str] | None = None,
) -> ArchetypeReport:
    fingerprints = compute_round_fingerprints(rounds)
    clustering = cluster_round_fingerprints(
        fingerprints,
        k=k,
        max_k=max_k,
        random_seed=random_seed,
        feature_names=feature_names or DEFAULT_FEATURES,
    )
    return ArchetypeReport(fingerprints=fingerprints, clustering=clustering)


def compute_round_fingerprints(rounds: Iterable[RoundRecord]) -> list[RoundFingerprint]:
    records: list[RoundFingerprint] = []
    for round_record in rounds:
        seed_metrics: list[SeedFingerprint] = []
        for seed in round_record.seeds:
            metrics = _compute_seed_fingerprint(seed)
            if metrics is not None:
                seed_metrics.append(metrics)

        if not seed_metrics:
            continue

        records.append(
            RoundFingerprint(
                round_id=round_record.round_id,
                round_number=round_record.round_number,
                seeds_used=len(seed_metrics),
                initial_settlements=_mean([m.initial_settlements for m in seed_metrics]),
                final_settlements=_mean([m.final_settlements for m in seed_metrics]),
                settlement_survival_rate=_mean(
                    [m.settlement_survival_rate for m in seed_metrics]
                ),
                settlement_growth_rate=_mean(
                    [m.settlement_growth_rate for m in seed_metrics]
                ),
                port_retention_rate=_mean([m.port_retention_rate for m in seed_metrics]),
                port_creation_rate=_mean([m.port_creation_rate for m in seed_metrics]),
                ruin_frequency=_mean([m.ruin_frequency for m in seed_metrics]),
                forest_reclaim_rate=_mean([m.forest_reclaim_rate for m in seed_metrics]),
                average_spread_radius=_mean(
                    [m.average_spread_radius for m in seed_metrics]
                ),
                takeover_conflict_rate=_mean(
                    [m.takeover_conflict_rate for m in seed_metrics]
                ),
                owner_change_rate=_mean([m.owner_change_rate for m in seed_metrics]),
                owner_consolidation=_mean([m.owner_consolidation for m in seed_metrics]),
                dominant_owner_share=_mean([m.dominant_owner_share for m in seed_metrics]),
                survivor_population_gap=_mean(
                    [m.survivor_population_gap for m in seed_metrics]
                ),
                survivor_food_gap=_mean([m.survivor_food_gap for m in seed_metrics]),
                survivor_wealth_gap=_mean([m.survivor_wealth_gap for m in seed_metrics]),
                survivor_defense_gap=_mean(
                    [m.survivor_defense_gap for m in seed_metrics]
                ),
            )
        )

    records.sort(key=lambda item: item.round_number or 0)
    return records


def cluster_round_fingerprints(
    fingerprints: list[RoundFingerprint],
    *,
    k: int = 4,
    max_k: int = 8,
    random_seed: int = 7,
    feature_names: list[str] | None = None,
) -> ClusteringResult:
    if not fingerprints:
        raise ValueError("No fingerprints available to cluster")

    feature_names = feature_names or DEFAULT_FEATURES
    matrix = [fp.feature_vector(feature_names) for fp in fingerprints]
    normalized, means, stds = _zscore(matrix)

    points_n = len(normalized)
    if points_n == 1:
        assignments = [
            ClusterPoint(
                round_id=fingerprints[0].round_id,
                cluster_id=0,
                pc1=0.0,
                pc2=0.0,
            )
        ]
        return ClusteringResult(
            k=1,
            feature_names=feature_names,
            means=means,
            stds=stds,
            assignments=assignments,
            inertia=0.0,
            centroids_normalized=[normalized[0]],
            centroids_raw=[matrix[0]],
            elbow=[(1, 0.0)],
        )

    k_eff = max(2, min(k, points_n))
    max_k_eff = max(2, min(max_k, points_n))
    if max_k_eff < k_eff:
        max_k_eff = k_eff

    rng = random.Random(random_seed)
    cluster_ids, centroids_norm, inertia = _kmeans(normalized, k_eff, rng)

    pc1, pc2 = _pca_2d(normalized, random_seed=random_seed)
    assignments = [
        ClusterPoint(
            round_id=fingerprints[idx].round_id,
            cluster_id=cluster_ids[idx],
            pc1=pc1[idx],
            pc2=pc2[idx],
        )
        for idx in range(points_n)
    ]

    elbow: list[tuple[int, float]] = []
    for candidate_k in range(2, max_k_eff + 1):
        _, _, candidate_inertia = _kmeans(
            normalized,
            candidate_k,
            random.Random(random_seed + candidate_k),
        )
        elbow.append((candidate_k, candidate_inertia))

    centroids_raw = [
        [
            (centroids_norm[c_idx][f_idx] * stds[f_idx]) + means[f_idx]
            for f_idx in range(len(feature_names))
        ]
        for c_idx in range(k_eff)
    ]

    return ClusteringResult(
        k=k_eff,
        feature_names=feature_names,
        means=means,
        stds=stds,
        assignments=assignments,
        inertia=inertia,
        centroids_normalized=centroids_norm,
        centroids_raw=centroids_raw,
        elbow=elbow,
    )


def _compute_seed_fingerprint(seed: RoundSeedRecord) -> SeedFingerprint | None:
    replay = seed.replay
    if not isinstance(replay, dict):
        return None

    initial_frame = _frame_by_step(replay, pick="min")
    final_frame = _frame_by_step(replay, pick="max")
    if initial_frame is None or final_frame is None:
        return None

    initial_grid = None
    if seed.initial_state is not None:
        initial_grid = seed.initial_state.grid
    elif isinstance(initial_frame.get("grid"), list):
        initial_grid = initial_frame.get("grid")

    final_grid = final_frame.get("grid")
    if not isinstance(final_grid, list) or not final_grid:
        return None

    initial_settlements = _settlement_map(initial_frame.get("settlements", []))
    final_settlements = _settlement_map(final_frame.get("settlements", []))

    init_coords = set(initial_settlements.keys())
    final_coords = set(final_settlements.keys())
    survivors = init_coords & final_coords
    lost = init_coords - final_coords
    new = final_coords - init_coords

    initial_count = len(init_coords)
    final_count = len(final_coords)

    survival_rate = _safe_ratio(len(survivors), initial_count)
    growth_rate = _safe_ratio(final_count - initial_count, initial_count)

    init_port_coords = {
        coord
        for coord, settlement in initial_settlements.items()
        if settlement["has_port"]
    }
    final_port_coords = {
        coord for coord, settlement in final_settlements.items() if settlement["has_port"]
    }

    port_retention = _safe_ratio(
        len(init_port_coords & final_port_coords),
        len(init_port_coords),
    )
    port_creation = _safe_ratio(
        len(final_port_coords - init_port_coords),
        final_count,
    )

    owner_changes = 0
    for coord in survivors:
        owner_i = initial_settlements[coord]["owner_id"]
        owner_f = final_settlements[coord]["owner_id"]
        if owner_i is not None and owner_f is not None and owner_i != owner_f:
            owner_changes += 1
    owner_change_rate = _safe_ratio(owner_changes, len(survivors))
    takeover_conflict_rate = _safe_ratio(len(lost) + owner_changes, initial_count)

    initial_owners = {
        settlement["owner_id"]
        for settlement in initial_settlements.values()
        if settlement["owner_id"] is not None
    }
    final_owner_counts: dict[int, int] = {}
    for settlement in final_settlements.values():
        owner = settlement["owner_id"]
        if owner is None:
            continue
        final_owner_counts[owner] = final_owner_counts.get(owner, 0) + 1

    if not initial_owners:
        owner_consolidation = 0.0
    else:
        owner_consolidation = 1.0 - _safe_ratio(len(final_owner_counts), len(initial_owners))
        owner_consolidation = max(0.0, min(1.0, owner_consolidation))

    dominant_owner_share = 0.0
    if final_count > 0 and final_owner_counts:
        dominant_owner_share = max(final_owner_counts.values()) / float(final_count)

    average_spread = _average_spread_radius(
        initial_coords=init_coords,
        final_coords=final_coords,
    )

    ruin_frequency = _terrain_frequency(final_grid, terrain_code=3)
    forest_reclaim_rate = _forest_reclaim_rate(initial_grid, final_grid)

    survivor_population_gap = _group_stat_gap(
        final_settlements,
        group_a=survivors,
        group_b=new,
        field="population",
    )
    survivor_food_gap = _group_stat_gap(
        final_settlements,
        group_a=survivors,
        group_b=new,
        field="food",
    )
    survivor_wealth_gap = _group_stat_gap(
        final_settlements,
        group_a=survivors,
        group_b=new,
        field="wealth",
    )
    survivor_defense_gap = _group_stat_gap(
        final_settlements,
        group_a=survivors,
        group_b=new,
        field="defense",
    )

    return SeedFingerprint(
        initial_settlements=float(initial_count),
        final_settlements=float(final_count),
        settlement_survival_rate=survival_rate,
        settlement_growth_rate=growth_rate,
        port_retention_rate=port_retention,
        port_creation_rate=port_creation,
        ruin_frequency=ruin_frequency,
        forest_reclaim_rate=forest_reclaim_rate,
        average_spread_radius=average_spread,
        takeover_conflict_rate=takeover_conflict_rate,
        owner_change_rate=owner_change_rate,
        owner_consolidation=owner_consolidation,
        dominant_owner_share=dominant_owner_share,
        survivor_population_gap=survivor_population_gap,
        survivor_food_gap=survivor_food_gap,
        survivor_wealth_gap=survivor_wealth_gap,
        survivor_defense_gap=survivor_defense_gap,
    )


def _frame_by_step(replay_payload: dict[str, Any], *, pick: str) -> dict[str, Any] | None:
    frames = replay_payload.get("frames")
    if not isinstance(frames, list) or not frames:
        return None

    frame_dicts = [frame for frame in frames if isinstance(frame, dict)]
    if not frame_dicts:
        return None

    if pick == "min":
        return min(frame_dicts, key=lambda frame: int(frame.get("step", 0)))
    if pick == "max":
        return max(frame_dicts, key=lambda frame: int(frame.get("step", 0)))
    raise ValueError(f"Unsupported pick mode: {pick}")


def _settlement_map(settlements_payload: Any) -> dict[Coord, dict[str, Any]]:
    out: dict[Coord, dict[str, Any]] = {}
    if not isinstance(settlements_payload, list):
        return out

    for raw in settlements_payload:
        if not isinstance(raw, dict):
            continue
        x = int(raw.get("x", -1))
        y = int(raw.get("y", -1))
        if x < 0 or y < 0:
            continue
        coord = (x, y)
        out[coord] = {
            "owner_id": int(raw["owner_id"]) if raw.get("owner_id") is not None else None,
            "has_port": bool(raw.get("has_port", False)),
            "population": _to_float(raw.get("population")),
            "food": _to_float(raw.get("food")),
            "wealth": _to_float(raw.get("wealth")),
            "defense": _to_float(raw.get("defense")),
        }
    return out


def _group_stat_gap(
    settlements: dict[Coord, dict[str, Any]],
    *,
    group_a: set[Coord],
    group_b: set[Coord],
    field: str,
) -> float:
    vals_a = [
        settlements[coord][field]
        for coord in group_a
        if coord in settlements and settlements[coord][field] is not None
    ]
    vals_b = [
        settlements[coord][field]
        for coord in group_b
        if coord in settlements and settlements[coord][field] is not None
    ]
    if not vals_a or not vals_b:
        return 0.0
    return _mean(vals_a) - _mean(vals_b)


def _average_spread_radius(*, initial_coords: set[Coord], final_coords: set[Coord]) -> float:
    if not initial_coords or not final_coords:
        return 0.0
    distances: list[int] = []
    for fx, fy in final_coords:
        distances.append(
            min(abs(fx - ix) + abs(fy - iy) for ix, iy in initial_coords)
        )
    return _mean([float(d) for d in distances])


def _terrain_frequency(grid: Any, *, terrain_code: int) -> float:
    if not isinstance(grid, list) or not grid:
        return 0.0
    total = 0
    matches = 0
    for row in grid:
        if not isinstance(row, list):
            continue
        for value in row:
            total += 1
            if int(value) == terrain_code:
                matches += 1
    return _safe_ratio(matches, total)


def _forest_reclaim_rate(initial_grid: Any, final_grid: Any) -> float:
    if not isinstance(initial_grid, list) or not isinstance(final_grid, list):
        return 0.0
    if len(initial_grid) != len(final_grid) or not initial_grid:
        return 0.0

    initial_ruins = 0
    reclaimed = 0
    for y, init_row in enumerate(initial_grid):
        fin_row = final_grid[y]
        if not isinstance(init_row, list) or not isinstance(fin_row, list):
            continue
        width = min(len(init_row), len(fin_row))
        for x in range(width):
            if int(init_row[x]) == 3:
                initial_ruins += 1
                if int(fin_row[x]) == 4:
                    reclaimed += 1
    return _safe_ratio(reclaimed, initial_ruins)


def _zscore(matrix: list[list[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    if not matrix:
        return [], [], []
    cols = len(matrix[0])
    means = [0.0] * cols
    stds = [1.0] * cols
    for col in range(cols):
        values = [row[col] for row in matrix]
        means[col] = _mean(values)
        var = _mean([(value - means[col]) ** 2 for value in values])
        std = math.sqrt(var)
        stds[col] = std if std > 1e-12 else 1.0
    normalized = [
        [(row[col] - means[col]) / stds[col] for col in range(cols)]
        for row in matrix
    ]
    return normalized, means, stds


def _kmeans(
    matrix: list[list[float]],
    k: int,
    rng: random.Random,
    *,
    max_iters: int = 100,
) -> tuple[list[int], list[list[float]], float]:
    points_n = len(matrix)
    dims = len(matrix[0]) if points_n > 0 else 0
    if k <= 0 or k > points_n:
        raise ValueError(f"Invalid k={k} for points_n={points_n}")

    centroid_indices = rng.sample(range(points_n), k)
    centroids = [matrix[idx][:] for idx in centroid_indices]
    assignments = [-1] * points_n

    for _ in range(max_iters):
        changed = False
        for idx, point in enumerate(matrix):
            best_cluster = 0
            best_dist = float("inf")
            for cluster_idx, centroid in enumerate(centroids):
                dist = _sq_distance(point, centroid)
                if dist < best_dist:
                    best_dist = dist
                    best_cluster = cluster_idx
            if assignments[idx] != best_cluster:
                assignments[idx] = best_cluster
                changed = True

        sums = [[0.0] * dims for _ in range(k)]
        counts = [0] * k
        for idx, point in enumerate(matrix):
            cluster_idx = assignments[idx]
            counts[cluster_idx] += 1
            for dim in range(dims):
                sums[cluster_idx][dim] += point[dim]

        for cluster_idx in range(k):
            if counts[cluster_idx] == 0:
                replacement = matrix[rng.randrange(points_n)]
                centroids[cluster_idx] = replacement[:]
            else:
                centroids[cluster_idx] = [
                    sums[cluster_idx][dim] / float(counts[cluster_idx])
                    for dim in range(dims)
                ]
        if not changed:
            break

    inertia = 0.0
    for idx, point in enumerate(matrix):
        inertia += _sq_distance(point, centroids[assignments[idx]])
    return assignments, centroids, inertia


def _pca_2d(
    matrix: list[list[float]],
    *,
    random_seed: int = 7,
) -> tuple[list[float], list[float]]:
    n = len(matrix)
    d = len(matrix[0]) if n > 0 else 0
    if n == 0:
        return [], []
    if d == 0:
        return [0.0] * n, [0.0] * n

    covariance = _covariance_matrix(matrix)
    rng = random.Random(random_seed)
    v1, lambda1 = _power_iteration(covariance, rng)
    if v1 is None:
        return [0.0] * n, [0.0] * n

    cov2 = [
        [
            covariance[i][j] - (lambda1 * v1[i] * v1[j])
            for j in range(d)
        ]
        for i in range(d)
    ]
    v2, _ = _power_iteration(cov2, random.Random(random_seed + 1))
    if v2 is None:
        v2 = [0.0] * d

    pc1 = [_dot(row, v1) for row in matrix]
    pc2 = [_dot(row, v2) for row in matrix]
    return pc1, pc2


def _covariance_matrix(matrix: list[list[float]]) -> list[list[float]]:
    n = len(matrix)
    d = len(matrix[0]) if n > 0 else 0
    if n <= 1:
        return [[0.0 for _ in range(d)] for _ in range(d)]
    cov = [[0.0 for _ in range(d)] for _ in range(d)]
    scale = 1.0 / float(n - 1)
    for row in matrix:
        for i in range(d):
            ri = row[i]
            for j in range(d):
                cov[i][j] += ri * row[j]
    for i in range(d):
        for j in range(d):
            cov[i][j] *= scale
    return cov


def _power_iteration(
    matrix: list[list[float]],
    rng: random.Random,
    *,
    iters: int = 80,
) -> tuple[list[float] | None, float]:
    dim = len(matrix)
    if dim == 0:
        return None, 0.0
    vec = [rng.random() - 0.5 for _ in range(dim)]
    norm = _norm(vec)
    if norm <= 1e-12:
        return None, 0.0
    vec = [value / norm for value in vec]

    for _ in range(iters):
        next_vec = [
            sum(matrix[row][col] * vec[col] for col in range(dim))
            for row in range(dim)
        ]
        next_norm = _norm(next_vec)
        if next_norm <= 1e-12:
            return None, 0.0
        vec = [value / next_norm for value in next_vec]

    eigenvalue = 0.0
    mv = [
        sum(matrix[row][col] * vec[col] for col in range(dim))
        for row in range(dim)
    ]
    eigenvalue = _dot(vec, mv)
    return vec, eigenvalue


def _round_labels(fingerprints: list[RoundFingerprint]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for fp in fingerprints:
        round_no = fp.round_number if fp.round_number is not None else "?"
        labels[fp.round_id] = f"R{round_no}"
    return labels


def _write_scatter_svg(
    path: Path,
    *,
    points: list[ClusterPoint],
    labels: dict[str, str],
) -> None:
    width = 980
    height = 620
    pad_x = 70
    pad_y = 50

    xs = [point.pc1 for point in points]
    ys = [point.pc2 for point in points]

    min_x, max_x = _minmax(xs)
    min_y, max_y = _minmax(ys)
    if math.isclose(min_x, max_x):
        max_x = min_x + 1.0
    if math.isclose(min_y, max_y):
        max_y = min_y + 1.0

    def to_x(value: float) -> float:
        return pad_x + ((value - min_x) / (max_x - min_x)) * (width - 2 * pad_x)

    def to_y(value: float) -> float:
        # Invert y for screen coordinates.
        return height - pad_y - ((value - min_y) / (max_y - min_y)) * (height - 2 * pad_y)

    palette = [
        "#d1495b",
        "#2e86ab",
        "#3caea3",
        "#f6d55c",
        "#6d597a",
        "#f18f01",
        "#1b9aaa",
        "#9c6644",
    ]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff" />',
        f'<line x1="{pad_x}" y1="{height - pad_y}" x2="{width - pad_x}" y2="{height - pad_y}" '
        'stroke="#444" stroke-width="1.5" />',
        f'<line x1="{pad_x}" y1="{pad_y}" x2="{pad_x}" y2="{height - pad_y}" '
        'stroke="#444" stroke-width="1.5" />',
        f'<text x="{width / 2}" y="28" text-anchor="middle" fill="#222" '
        'font-family="Arial, sans-serif" font-size="20">Round Archetype Clusters (PCA)</text>',
        f'<text x="{width / 2}" y="{height - 12}" text-anchor="middle" fill="#333" '
        'font-family="Arial, sans-serif" font-size="14">PC1</text>',
        f'<text x="22" y="{height / 2}" transform="rotate(-90 22 {height / 2})" text-anchor="middle" '
        'fill="#333" font-family="Arial, sans-serif" font-size="14">PC2</text>',
    ]

    for point in points:
        x = to_x(point.pc1)
        y = to_y(point.pc2)
        color = palette[point.cluster_id % len(palette)]
        label = labels.get(point.round_id, point.round_id[:6])
        parts.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6.2" fill="{color}" stroke="#222" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{x + 8:.2f}" y="{y - 8:.2f}" fill="#222" '
            'font-family="Arial, sans-serif" font-size="11">'
            f"{label}</text>"
        )

    legend_x = width - pad_x - 140
    legend_y = pad_y + 10
    cluster_ids = sorted({point.cluster_id for point in points})
    for idx, cluster_id in enumerate(cluster_ids):
        y = legend_y + idx * 20
        color = palette[cluster_id % len(palette)]
        parts.append(
            f'<rect x="{legend_x}" y="{y}" width="12" height="12" fill="{color}" stroke="#222" stroke-width="0.8" />'
        )
        parts.append(
            f'<text x="{legend_x + 18}" y="{y + 10}" fill="#222" '
            'font-family="Arial, sans-serif" font-size="12">'
            f"Cluster {cluster_id}</text>"
        )

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_elbow_svg(path: Path, *, elbow: list[tuple[int, float]]) -> None:
    width = 820
    height = 520
    pad_x = 70
    pad_y = 55

    if not elbow:
        path.write_text("", encoding="utf-8")
        return

    xs = [float(k) for k, _ in elbow]
    ys = [inertia for _, inertia in elbow]
    min_x, max_x = _minmax(xs)
    min_y, max_y = _minmax(ys)
    if math.isclose(min_x, max_x):
        max_x = min_x + 1.0
    if math.isclose(min_y, max_y):
        max_y = min_y + 1.0

    def to_x(value: float) -> float:
        return pad_x + ((value - min_x) / (max_x - min_x)) * (width - 2 * pad_x)

    def to_y(value: float) -> float:
        return height - pad_y - ((value - min_y) / (max_y - min_y)) * (height - 2 * pad_y)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff" />',
        f'<line x1="{pad_x}" y1="{height - pad_y}" x2="{width - pad_x}" y2="{height - pad_y}" stroke="#444" stroke-width="1.5" />',
        f'<line x1="{pad_x}" y1="{pad_y}" x2="{pad_x}" y2="{height - pad_y}" stroke="#444" stroke-width="1.5" />',
        f'<text x="{width / 2}" y="30" text-anchor="middle" fill="#222" font-family="Arial, sans-serif" font-size="20">K-Means Elbow Plot</text>',
        f'<text x="{width / 2}" y="{height - 14}" text-anchor="middle" fill="#333" font-family="Arial, sans-serif" font-size="14">k (clusters)</text>',
        f'<text x="20" y="{height / 2}" transform="rotate(-90 20 {height / 2})" text-anchor="middle" fill="#333" font-family="Arial, sans-serif" font-size="14">Inertia</text>',
    ]

    line_points = " ".join(
        f"{to_x(float(k)):.2f},{to_y(inertia):.2f}"
        for k, inertia in elbow
    )
    parts.append(
        f'<polyline points="{line_points}" fill="none" stroke="#2e86ab" stroke-width="2.5" />'
    )
    for k, inertia in elbow:
        x = to_x(float(k))
        y = to_y(inertia)
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="#d1495b" />')
        parts.append(
            f'<text x="{x:.2f}" y="{height - pad_y + 18}" text-anchor="middle" fill="#333" '
            'font-family="Arial, sans-serif" font-size="11">'
            f"{k}</text>"
        )

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_table_csv(
    path: Path,
    *,
    fingerprints: list[RoundFingerprint],
    assignments: list[ClusterPoint],
    feature_names: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_round = {item.round_id: item for item in assignments}
    headers = [
        "round_id",
        "round_number",
        "seeds_used",
        "cluster_id",
        "pc1",
        "pc2",
        "initial_settlements",
        "final_settlements",
        *feature_names,
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for fp in fingerprints:
            assignment = by_round.get(fp.round_id)
            row: dict[str, Any] = {
                "round_id": fp.round_id,
                "round_number": fp.round_number,
                "seeds_used": fp.seeds_used,
                "cluster_id": assignment.cluster_id if assignment is not None else "",
                "pc1": f"{assignment.pc1:.6f}" if assignment is not None else "",
                "pc2": f"{assignment.pc2:.6f}" if assignment is not None else "",
                "initial_settlements": f"{fp.initial_settlements:.6f}",
                "final_settlements": f"{fp.final_settlements:.6f}",
            }
            for feature_name in feature_names:
                row[feature_name] = f"{float(getattr(fp, feature_name)):.6f}"
            writer.writerow(row)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _minmax(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    return min(values), max(values)


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _sq_distance(a: list[float], b: list[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))
