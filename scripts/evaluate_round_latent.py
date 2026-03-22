#!/usr/bin/env python3
"""Evaluate step-9 round-latent model against prior baselines."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from astar_island.offline_emulator import OfflineRoundState, run_offline_round
from astar_island.priors import baseline_prior_from_initial_grid
from astar_island.round_data import load_round_dataset
from astar_island.round_latent import RoundLatentConditionalModel


@dataclass(slots=True)
class SeedEvalRow:
    round_id: str
    round_number: int | None
    seed_index: int
    score_no_query_prior: float
    score_query_prior: float
    score_query_latent: float
    delta_latent_vs_query_prior: float
    delta_latent_vs_no_query_prior: float
    weighted_kl_no_query_prior: float
    weighted_kl_query_prior: float
    weighted_kl_query_latent: float


@dataclass(slots=True)
class RoundEvalRow:
    round_id: str
    round_number: int | None
    queries_used_query_prior: int
    queries_used_query_latent: int
    round_score_no_query_prior: float
    round_score_query_prior: float
    round_score_query_latent: float
    delta_latent_vs_query_prior: float
    delta_latent_vs_no_query_prior: float


class NoQueryPolicy:
    def next_query(self, state: OfflineRoundState) -> None:  # noqa: ARG002
        return None


class PriorOnlyModel:
    def predict(self, round_state, seed_initial_state, seed_index):  # noqa: ANN001, ARG002
        if seed_initial_state is None:
            width = int(round_state.map_width)
            height = int(round_state.map_height)
            uniform = [1.0 / 6.0 for _ in range(6)]
            return [[uniform[:] for _ in range(width)] for _ in range(height)]
        return baseline_prior_from_initial_grid(
            seed_initial_state.grid,
            settlements=seed_initial_state.settlements,
        )


class DeterministicThreePhasePolicy:
    """Step-10-like deterministic scheduler used for fair model comparison."""

    def __init__(self, *, query_budget: int = 50, default_window: int = 15) -> None:
        self.query_budget = query_budget
        self.default_window = default_window
        self._plan: list[dict[str, int]] = []
        self._prepared = False

    def next_query(self, state: OfflineRoundState) -> dict[str, int] | None:
        if not self._prepared:
            self._prepare_plan(state)
            self._prepared = True
        idx = int(state.queries_used)
        if idx < 0 or idx >= len(self._plan):
            return None
        return self._plan[idx]

    def _prepare_plan(self, state: OfflineRoundState) -> None:
        if state.seeds_count <= 0:
            self._plan = []
            return

        window = min(
            max(5, int(self.default_window)),
            int(state.map_width),
            int(state.map_height),
        )
        if window < 5:
            self._plan = []
            return

        per_seed_windows: dict[int, list[tuple[int, int]]] = {}
        for seed_index, initial in enumerate(state.initial_states):
            if initial is None:
                per_seed_windows[seed_index] = [(0, 0)]
                continue
            importance = _importance_map_for_seed(initial.grid, initial.settlements)
            top_windows = _top_windows(importance, window=window, count=8, min_center_distance=6.0)
            center = _center_window(
                width=state.map_width,
                height=state.map_height,
                window=window,
            )
            settlement = _settlement_window(
                settlements=initial.settlements,
                width=state.map_width,
                height=state.map_height,
                window=window,
            )
            ordered = [top_windows[0] if top_windows else center, settlement, center]
            ordered.extend(top_windows[1:])
            per_seed_windows[seed_index] = _dedupe_windows(ordered)

        # Phase 1 (15): broad calibration, 3 probes per seed.
        phase1: list[dict[str, int]] = []
        for seed in range(state.seeds_count):
            windows = per_seed_windows.get(seed, [(0, 0)])
            picks = [windows[0], windows[1] if len(windows) > 1 else windows[0], windows[2] if len(windows) > 2 else windows[0]]
            for x, y in picks:
                phase1.append(_query(seed, x, y, window))

        # Phase 2 (20): repeated probes on informative motifs, two windows per seed repeated twice.
        phase2: list[dict[str, int]] = []
        for seed in range(state.seeds_count):
            windows = per_seed_windows.get(seed, [(0, 0)])
            first = windows[0]
            second = windows[1] if len(windows) > 1 else first
            phase2.append(_query(seed, first[0], first[1], window))
            phase2.append(_query(seed, second[0], second[1], window))
            phase2.append(_query(seed, first[0], first[1], window))
            phase2.append(_query(seed, second[0], second[1], window))

        # Phase 3 (15): exploit uncertainty, take next three windows per seed.
        phase3: list[dict[str, int]] = []
        for seed in range(state.seeds_count):
            windows = per_seed_windows.get(seed, [(0, 0)])
            for idx in range(2, 5):
                pick = windows[idx] if idx < len(windows) else windows[-1]
                phase3.append(_query(seed, pick[0], pick[1], window))

        self._plan = (phase1 + phase2 + phase3)[: self.query_budget]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-root",
        default=str(REPO_ROOT / "logs"),
        help="Logs root with historical rounds.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs" / "step9_round_latent_eval"),
        help="Output folder for step-9 evaluation artifacts.",
    )
    parser.add_argument(
        "--query-budget",
        type=int,
        default=50,
        help="Max queries per round.",
    )
    parser.add_argument(
        "--probability-floor",
        type=float,
        default=1e-4,
        help="Probability floor before normalization in emulator validation path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require all rounds/seeds/replays to be present.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logs_root = Path(args.logs_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rounds = load_round_dataset(logs_root, include_replays=True, strict=args.strict)
    rounds_sorted = sorted(
        rounds,
        key=lambda row: (
            row.round_number is None,
            int(row.round_number or 0),
            row.round_id,
        ),
    )

    seed_rows: list[SeedEvalRow] = []
    round_rows: list[RoundEvalRow] = []

    for record in rounds_sorted:
        no_query = run_offline_round(
            policy=NoQueryPolicy(),
            model=PriorOnlyModel(),
            round_id=record.round_id,
            logs_root=str(logs_root),
            query_budget=0,
            include_replays=True,
            strict=args.strict,
            allow_missing_replays=False,
            probability_floor=args.probability_floor,
        )
        query_prior = run_offline_round(
            policy=DeterministicThreePhasePolicy(query_budget=args.query_budget),
            model=PriorOnlyModel(),
            round_id=record.round_id,
            logs_root=str(logs_root),
            query_budget=args.query_budget,
            include_replays=True,
            strict=args.strict,
            allow_missing_replays=False,
            probability_floor=args.probability_floor,
        )
        query_latent = run_offline_round(
            policy=DeterministicThreePhasePolicy(query_budget=args.query_budget),
            model=RoundLatentConditionalModel(),
            round_id=record.round_id,
            logs_root=str(logs_root),
            query_budget=args.query_budget,
            include_replays=True,
            strict=args.strict,
            allow_missing_replays=False,
            probability_floor=args.probability_floor,
        )

        no_query_by_seed = {row.seed_index: row for row in no_query.per_seed}
        query_prior_by_seed = {row.seed_index: row for row in query_prior.per_seed}
        query_latent_by_seed = {row.seed_index: row for row in query_latent.per_seed}
        seed_indices = sorted(
            set(no_query_by_seed) | set(query_prior_by_seed) | set(query_latent_by_seed)
        )
        for seed_index in seed_indices:
            a = no_query_by_seed[seed_index]
            b = query_prior_by_seed[seed_index]
            c = query_latent_by_seed[seed_index]
            seed_rows.append(
                SeedEvalRow(
                    round_id=record.round_id,
                    round_number=record.round_number,
                    seed_index=seed_index,
                    score_no_query_prior=a.score,
                    score_query_prior=b.score,
                    score_query_latent=c.score,
                    delta_latent_vs_query_prior=c.score - b.score,
                    delta_latent_vs_no_query_prior=c.score - a.score,
                    weighted_kl_no_query_prior=a.weighted_kl,
                    weighted_kl_query_prior=b.weighted_kl,
                    weighted_kl_query_latent=c.weighted_kl,
                )
            )

        round_rows.append(
            RoundEvalRow(
                round_id=record.round_id,
                round_number=record.round_number,
                queries_used_query_prior=query_prior.queries_used,
                queries_used_query_latent=query_latent.queries_used,
                round_score_no_query_prior=no_query.round_score,
                round_score_query_prior=query_prior.round_score,
                round_score_query_latent=query_latent.round_score,
                delta_latent_vs_query_prior=query_latent.round_score - query_prior.round_score,
                delta_latent_vs_no_query_prior=query_latent.round_score - no_query.round_score,
            )
        )

    seed_csv = output_dir / "seed_results.csv"
    round_csv = output_dir / "round_results.csv"
    summary_json = output_dir / "summary.json"
    run_summary_json = output_dir / "run_summary.json"

    _write_csv(seed_csv, seed_rows)
    _write_csv(round_csv, round_rows)
    summary = _build_summary(seed_rows, round_rows, query_budget=args.query_budget)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    run_summary = {
        "rounds_evaluated": len(round_rows),
        "seeds_evaluated": len(seed_rows),
        "query_budget": args.query_budget,
        "mean_round_score_no_query_prior": summary["mean_round_score_no_query_prior"],
        "mean_round_score_query_prior": summary["mean_round_score_query_prior"],
        "mean_round_score_query_latent": summary["mean_round_score_query_latent"],
        "mean_round_delta_latent_vs_query_prior": summary["mean_round_delta_latent_vs_query_prior"],
        "mean_round_delta_latent_vs_no_query_prior": summary["mean_round_delta_latent_vs_no_query_prior"],
        "output_files": {
            "seed_csv": str(seed_csv),
            "round_csv": str(round_csv),
            "summary_json": str(summary_json),
        },
    }
    run_summary_json.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    print("Step 9 evaluation complete.")
    print(f"rounds_evaluated={len(round_rows)}")
    print(f"seeds_evaluated={len(seed_rows)}")
    print(f"mean_round_score_no_query_prior={summary['mean_round_score_no_query_prior']:.6f}")
    print(f"mean_round_score_query_prior={summary['mean_round_score_query_prior']:.6f}")
    print(f"mean_round_score_query_latent={summary['mean_round_score_query_latent']:.6f}")
    print(
        "mean_round_delta_latent_vs_query_prior="
        f"{summary['mean_round_delta_latent_vs_query_prior']:+.6f}"
    )
    print(
        "mean_round_delta_latent_vs_no_query_prior="
        f"{summary['mean_round_delta_latent_vs_no_query_prior']:+.6f}"
    )
    print("output_files:")
    print(f"  seed_csv: {seed_csv}")
    print(f"  round_csv: {round_csv}")
    print(f"  summary_json: {summary_json}")
    print(f"  run_summary_json: {run_summary_json}")
    return 0


def _write_csv(path: Path, rows: list[object]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _build_summary(
    seed_rows: list[SeedEvalRow],
    round_rows: list[RoundEvalRow],
    *,
    query_budget: int,
) -> dict[str, object]:
    mean_seed_no_query_prior = _mean([row.score_no_query_prior for row in seed_rows])
    mean_seed_query_prior = _mean([row.score_query_prior for row in seed_rows])
    mean_seed_query_latent = _mean([row.score_query_latent for row in seed_rows])

    mean_round_no_query_prior = _mean([row.round_score_no_query_prior for row in round_rows])
    mean_round_query_prior = _mean([row.round_score_query_prior for row in round_rows])
    mean_round_query_latent = _mean([row.round_score_query_latent for row in round_rows])
    round_delta_vs_query = [row.delta_latent_vs_query_prior for row in round_rows]
    round_delta_vs_no_query = [row.delta_latent_vs_no_query_prior for row in round_rows]

    return {
        "query_budget": query_budget,
        "rounds_evaluated": len(round_rows),
        "seeds_evaluated": len(seed_rows),
        "mean_seed_score_no_query_prior": mean_seed_no_query_prior,
        "mean_seed_score_query_prior": mean_seed_query_prior,
        "mean_seed_score_query_latent": mean_seed_query_latent,
        "mean_seed_delta_latent_vs_query_prior": mean_seed_query_latent - mean_seed_query_prior,
        "mean_seed_delta_latent_vs_no_query_prior": mean_seed_query_latent - mean_seed_no_query_prior,
        "mean_round_score_no_query_prior": mean_round_no_query_prior,
        "mean_round_score_query_prior": mean_round_query_prior,
        "mean_round_score_query_latent": mean_round_query_latent,
        "mean_round_delta_latent_vs_query_prior": mean_round_query_latent - mean_round_query_prior,
        "mean_round_delta_latent_vs_no_query_prior": mean_round_query_latent - mean_round_no_query_prior,
        "positive_round_delta_count": sum(1 for value in round_delta_vs_query if value > 0.0),
        "negative_round_delta_count": sum(1 for value in round_delta_vs_query if value < 0.0),
        "flat_round_delta_count": sum(1 for value in round_delta_vs_query if value == 0.0),
        "median_round_delta_latent_vs_query_prior": _median(round_delta_vs_query),
        "median_round_delta_latent_vs_no_query_prior": _median(round_delta_vs_no_query),
        "mean_queries_used_query_prior": _mean([row.queries_used_query_prior for row in round_rows]),
        "mean_queries_used_query_latent": _mean([row.queries_used_query_latent for row in round_rows]),
    }


def _importance_map_for_seed(
    initial_grid: list[list[int]],
    settlements,
) -> list[list[float]]:  # noqa: ANN001
    prior = baseline_prior_from_initial_grid(initial_grid, settlements=settlements)
    out: list[list[float]] = []
    for y, row in enumerate(prior):
        values: list[float] = []
        for x, probs in enumerate(row):
            dynamic_mass = probs[1] + probs[2] + probs[3]
            entropy = 0.0
            for p in probs:
                if p > 0.0:
                    entropy -= p * math.log(p)
            ent_norm = entropy / math.log(6.0)
            coast_bonus = 0.0
            if _is_coastal(initial_grid, x=x, y=y):
                coast_bonus = 0.12
            score = 0.72 * dynamic_mass + 0.20 * ent_norm + coast_bonus
            values.append(max(0.0, min(1.0, score)))
        out.append(values)
    return out


def _top_windows(
    importance: list[list[float]],
    *,
    window: int,
    count: int,
    min_center_distance: float,
) -> list[tuple[int, int]]:
    height = len(importance)
    width = len(importance[0]) if height > 0 else 0
    if width <= 0 or height <= 0:
        return [(0, 0)]

    window = min(window, width, height)
    max_x = width - window
    max_y = height - window
    integral = _integral_image(importance)

    candidates: list[tuple[float, int, int]] = []
    for y in range(max_y + 1):
        for x in range(max_x + 1):
            score = _window_sum(integral, x=x, y=y, w=window, h=window)
            candidates.append((score, x, y))
    candidates.sort(reverse=True, key=lambda row: row[0])

    selected: list[tuple[int, int]] = []
    selected_centers: list[tuple[float, float]] = []
    for _, x, y in candidates:
        cx = x + (window / 2.0)
        cy = y + (window / 2.0)
        too_close = any(
            math.dist((cx, cy), (sx, sy)) < min_center_distance
            for sx, sy in selected_centers
        )
        if too_close:
            continue
        selected.append((x, y))
        selected_centers.append((cx, cy))
        if len(selected) >= count:
            break

    if not selected:
        selected.append(_center_window(width=width, height=height, window=window))
    return selected


def _integral_image(values: list[list[float]]) -> list[list[float]]:
    height = len(values)
    width = len(values[0]) if height > 0 else 0
    integral = [[0.0 for _ in range(width + 1)] for _ in range(height + 1)]
    for y in range(height):
        row_sum = 0.0
        for x in range(width):
            row_sum += values[y][x]
            integral[y + 1][x + 1] = integral[y][x + 1] + row_sum
    return integral


def _window_sum(integral: list[list[float]], *, x: int, y: int, w: int, h: int) -> float:
    y2 = y + h
    x2 = x + w
    return (
        integral[y2][x2]
        - integral[y][x2]
        - integral[y2][x]
        + integral[y][x]
    )


def _center_window(*, width: int, height: int, window: int) -> tuple[int, int]:
    max_x = max(0, width - window)
    max_y = max(0, height - window)
    return (max_x // 2, max_y // 2)


def _settlement_window(
    *,
    settlements,
    width: int,
    height: int,
    window: int,
) -> tuple[int, int]:  # noqa: ANN001
    alive = [settlement for settlement in settlements if bool(settlement.alive)]
    pick = alive[0] if alive else (settlements[0] if settlements else None)
    if pick is None:
        return _center_window(width=width, height=height, window=window)
    sx = int(pick.x)
    sy = int(pick.y)
    return _clamp_window_top_left(
        x=sx - (window // 2),
        y=sy - (window // 2),
        width=width,
        height=height,
        window=window,
    )


def _dedupe_windows(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for point in points:
        if point in seen:
            continue
        seen.add(point)
        out.append(point)
    return out


def _query(seed_index: int, x: int, y: int, window: int) -> dict[str, int]:
    return {
        "seed_index": seed_index,
        "viewport_x": x,
        "viewport_y": y,
        "viewport_w": window,
        "viewport_h": window,
    }


def _is_coastal(grid: list[list[int]], *, x: int, y: int) -> bool:
    for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
        nx = x + dx
        ny = y + dy
        if ny < 0 or ny >= len(grid):
            continue
        if nx < 0 or nx >= len(grid[ny]):
            continue
        if int(grid[ny][nx]) == 10:
            return True
    return False


def _clamp_window_top_left(
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    window: int,
) -> tuple[int, int]:
    max_x = max(0, width - window)
    max_y = max(0, height - window)
    return (min(max(x, 0), max_x), min(max(y, 0), max_y))


def _mean(values: list[float | int]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


if __name__ == "__main__":
    raise SystemExit(main())
