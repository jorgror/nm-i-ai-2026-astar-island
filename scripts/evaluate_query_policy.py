#!/usr/bin/env python3
"""Backtest deterministic three-phase query policy against a simple fixed baseline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from astar_island.offline_emulator import OfflineRoundState, ViewportQuery, run_offline_round
from astar_island.priors import baseline_prior_from_initial_grid
from astar_island.query_policy import (
    DeterministicThreePhasePolicyConfig,
    DeterministicThreePhaseQueryPolicy,
)
from astar_island.round_data import load_round_dataset
from astar_island.round_latent import RoundLatentConditionalModel


@dataclass(slots=True)
class RoundPolicyRow:
    round_id: str
    round_number: int | None
    queries_used_three_phase: int
    queries_used_center: int
    round_score_three_phase: float
    round_score_center: float
    delta_three_phase_vs_center: float


@dataclass(slots=True)
class SeedPolicyRow:
    round_id: str
    round_number: int | None
    seed_index: int
    score_three_phase: float
    score_center: float
    delta_three_phase_vs_center: float
    weighted_kl_three_phase: float
    weighted_kl_center: float


class CenterSweepPolicy:
    """Simple fixed policy: round-robin seeds, always center viewport."""

    def __init__(self, *, query_budget: int = 50, default_window: int = 15) -> None:
        self.query_budget = query_budget
        self.default_window = default_window

    def next_query(self, state: OfflineRoundState) -> ViewportQuery | None:
        idx = int(state.queries_used)
        if idx >= self.query_budget or state.seeds_count <= 0:
            return None
        window = min(
            max(5, int(self.default_window)),
            int(state.map_width),
            int(state.map_height),
        )
        if window < 5:
            return None
        max_x = max(0, state.map_width - window)
        max_y = max(0, state.map_height - window)
        return ViewportQuery(
            seed_index=idx % state.seeds_count,
            viewport_x=max_x // 2,
            viewport_y=max_y // 2,
            viewport_w=window,
            viewport_h=window,
        )


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-root",
        default=str(REPO_ROOT / "logs"),
        help="Logs root with historical rounds.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs" / "step10_query_policy_eval"),
        help="Output folder for policy backtest artifacts.",
    )
    parser.add_argument(
        "--query-budget",
        type=int,
        default=50,
        help="Max queries per round for both policies.",
    )
    parser.add_argument(
        "--model",
        choices=("latent", "prior"),
        default="latent",
        help="Model used while comparing policies.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require all rounds/seeds/replays to be present.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rounds = load_round_dataset(args.logs_root, include_replays=True, strict=args.strict)
    rounds_sorted = sorted(
        rounds,
        key=lambda row: (
            row.round_number is None,
            int(row.round_number or 0),
            row.round_id,
        ),
    )

    model = RoundLatentConditionalModel() if args.model == "latent" else PriorOnlyModel()
    round_rows: list[RoundPolicyRow] = []
    seed_rows: list[SeedPolicyRow] = []

    for record in rounds_sorted:
        three_phase = run_offline_round(
            policy=DeterministicThreePhaseQueryPolicy(
                config=DeterministicThreePhasePolicyConfig(query_budget=args.query_budget)
            ),
            model=model,
            round_id=record.round_id,
            logs_root=args.logs_root,
            query_budget=args.query_budget,
            include_replays=True,
            strict=args.strict,
            allow_missing_replays=False,
        )
        center = run_offline_round(
            policy=CenterSweepPolicy(query_budget=args.query_budget),
            model=model,
            round_id=record.round_id,
            logs_root=args.logs_root,
            query_budget=args.query_budget,
            include_replays=True,
            strict=args.strict,
            allow_missing_replays=False,
        )

        three_by_seed = {row.seed_index: row for row in three_phase.per_seed}
        center_by_seed = {row.seed_index: row for row in center.per_seed}
        for seed_index in sorted(set(three_by_seed) | set(center_by_seed)):
            a = three_by_seed[seed_index]
            b = center_by_seed[seed_index]
            seed_rows.append(
                SeedPolicyRow(
                    round_id=record.round_id,
                    round_number=record.round_number,
                    seed_index=seed_index,
                    score_three_phase=a.score,
                    score_center=b.score,
                    delta_three_phase_vs_center=a.score - b.score,
                    weighted_kl_three_phase=a.weighted_kl,
                    weighted_kl_center=b.weighted_kl,
                )
            )

        round_rows.append(
            RoundPolicyRow(
                round_id=record.round_id,
                round_number=record.round_number,
                queries_used_three_phase=three_phase.queries_used,
                queries_used_center=center.queries_used,
                round_score_three_phase=three_phase.round_score,
                round_score_center=center.round_score,
                delta_three_phase_vs_center=three_phase.round_score - center.round_score,
            )
        )

    seed_csv = output_dir / "seed_results.csv"
    round_csv = output_dir / "round_results.csv"
    summary_json = output_dir / "summary.json"
    run_summary_json = output_dir / "run_summary.json"

    _write_csv(seed_csv, seed_rows)
    _write_csv(round_csv, round_rows)
    summary = _build_summary(round_rows=round_rows, seed_rows=seed_rows, model=args.model)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    run_summary = {
        "model": args.model,
        "rounds_evaluated": len(round_rows),
        "seeds_evaluated": len(seed_rows),
        "query_budget": args.query_budget,
        "mean_round_score_three_phase": summary["mean_round_score_three_phase"],
        "mean_round_score_center": summary["mean_round_score_center"],
        "mean_round_delta_three_phase_vs_center": summary["mean_round_delta_three_phase_vs_center"],
        "output_files": {
            "seed_csv": str(seed_csv),
            "round_csv": str(round_csv),
            "summary_json": str(summary_json),
        },
    }
    run_summary_json.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    print("Step 10 policy backtest complete.")
    print(f"model={args.model}")
    print(f"rounds_evaluated={len(round_rows)}")
    print(f"seeds_evaluated={len(seed_rows)}")
    print(f"mean_round_score_three_phase={summary['mean_round_score_three_phase']:.6f}")
    print(f"mean_round_score_center={summary['mean_round_score_center']:.6f}")
    print(
        "mean_round_delta_three_phase_vs_center="
        f"{summary['mean_round_delta_three_phase_vs_center']:+.6f}"
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
    *,
    round_rows: list[RoundPolicyRow],
    seed_rows: list[SeedPolicyRow],
    model: str,
) -> dict[str, object]:
    round_deltas = [row.delta_three_phase_vs_center for row in round_rows]
    return {
        "model": model,
        "rounds_evaluated": len(round_rows),
        "seeds_evaluated": len(seed_rows),
        "mean_seed_score_three_phase": _mean([row.score_three_phase for row in seed_rows]),
        "mean_seed_score_center": _mean([row.score_center for row in seed_rows]),
        "mean_seed_delta_three_phase_vs_center": _mean(
            [row.delta_three_phase_vs_center for row in seed_rows]
        ),
        "mean_round_score_three_phase": _mean([row.round_score_three_phase for row in round_rows]),
        "mean_round_score_center": _mean([row.round_score_center for row in round_rows]),
        "mean_round_delta_three_phase_vs_center": _mean(round_deltas),
        "median_round_delta_three_phase_vs_center": _median(round_deltas),
        "positive_round_delta_count": sum(1 for value in round_deltas if value > 0.0),
        "negative_round_delta_count": sum(1 for value in round_deltas if value < 0.0),
        "flat_round_delta_count": sum(1 for value in round_deltas if value == 0.0),
        "mean_queries_used_three_phase": _mean([row.queries_used_three_phase for row in round_rows]),
        "mean_queries_used_center": _mean([row.queries_used_center for row in round_rows]),
    }


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
