#!/usr/bin/env python3
"""Evaluate step-11 empirical blending on top of the round-latent model."""

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

from astar_island.offline_emulator import run_offline_round
from astar_island.query_policy import (
    DeterministicThreePhasePolicyConfig,
    DeterministicThreePhaseQueryPolicy,
)
from astar_island.round_data import load_round_dataset
from astar_island.round_latent import RoundLatentConditionalModel, RoundLatentConfig


@dataclass(slots=True)
class SeedBlendRow:
    round_id: str
    round_number: int | None
    seed_index: int
    score_no_blend: float
    score_blend: float
    delta_blend_vs_no_blend: float
    weighted_kl_no_blend: float
    weighted_kl_blend: float


@dataclass(slots=True)
class RoundBlendRow:
    round_id: str
    round_number: int | None
    queries_used_no_blend: int
    queries_used_blend: int
    round_score_no_blend: float
    round_score_blend: float
    delta_blend_vs_no_blend: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-root",
        default=str(REPO_ROOT / "logs"),
        help="Logs root with historical rounds.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs" / "step11_blending_eval"),
        help="Output folder for step-11 artifacts.",
    )
    parser.add_argument(
        "--query-budget",
        type=int,
        default=50,
        help="Max queries per round.",
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

    seed_rows: list[SeedBlendRow] = []
    round_rows: list[RoundBlendRow] = []
    for record in rounds_sorted:
        no_blend = run_offline_round(
            policy=DeterministicThreePhaseQueryPolicy(
                config=DeterministicThreePhasePolicyConfig(query_budget=args.query_budget)
            ),
            model=RoundLatentConditionalModel(
                config=RoundLatentConfig(enable_observation_blend=False)
            ),
            round_id=record.round_id,
            logs_root=args.logs_root,
            query_budget=args.query_budget,
            include_replays=True,
            strict=args.strict,
            allow_missing_replays=False,
        )
        blend = run_offline_round(
            policy=DeterministicThreePhaseQueryPolicy(
                config=DeterministicThreePhasePolicyConfig(query_budget=args.query_budget)
            ),
            model=RoundLatentConditionalModel(
                config=RoundLatentConfig(enable_observation_blend=True)
            ),
            round_id=record.round_id,
            logs_root=args.logs_root,
            query_budget=args.query_budget,
            include_replays=True,
            strict=args.strict,
            allow_missing_replays=False,
        )

        no_blend_by_seed = {row.seed_index: row for row in no_blend.per_seed}
        blend_by_seed = {row.seed_index: row for row in blend.per_seed}
        for seed_index in sorted(set(no_blend_by_seed) | set(blend_by_seed)):
            a = no_blend_by_seed[seed_index]
            b = blend_by_seed[seed_index]
            seed_rows.append(
                SeedBlendRow(
                    round_id=record.round_id,
                    round_number=record.round_number,
                    seed_index=seed_index,
                    score_no_blend=a.score,
                    score_blend=b.score,
                    delta_blend_vs_no_blend=b.score - a.score,
                    weighted_kl_no_blend=a.weighted_kl,
                    weighted_kl_blend=b.weighted_kl,
                )
            )

        round_rows.append(
            RoundBlendRow(
                round_id=record.round_id,
                round_number=record.round_number,
                queries_used_no_blend=no_blend.queries_used,
                queries_used_blend=blend.queries_used,
                round_score_no_blend=no_blend.round_score,
                round_score_blend=blend.round_score,
                delta_blend_vs_no_blend=blend.round_score - no_blend.round_score,
            )
        )

    seed_csv = output_dir / "seed_results.csv"
    round_csv = output_dir / "round_results.csv"
    summary_json = output_dir / "summary.json"
    run_summary_json = output_dir / "run_summary.json"

    _write_csv(seed_csv, seed_rows)
    _write_csv(round_csv, round_rows)
    summary = _build_summary(seed_rows=seed_rows, round_rows=round_rows)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    run_summary = {
        "rounds_evaluated": len(round_rows),
        "seeds_evaluated": len(seed_rows),
        "query_budget": args.query_budget,
        "mean_round_score_no_blend": summary["mean_round_score_no_blend"],
        "mean_round_score_blend": summary["mean_round_score_blend"],
        "mean_round_delta_blend_vs_no_blend": summary["mean_round_delta_blend_vs_no_blend"],
        "output_files": {
            "seed_csv": str(seed_csv),
            "round_csv": str(round_csv),
            "summary_json": str(summary_json),
        },
    }
    run_summary_json.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    print("Step 11 blending evaluation complete.")
    print(f"rounds_evaluated={len(round_rows)}")
    print(f"seeds_evaluated={len(seed_rows)}")
    print(f"mean_round_score_no_blend={summary['mean_round_score_no_blend']:.6f}")
    print(f"mean_round_score_blend={summary['mean_round_score_blend']:.6f}")
    print(
        "mean_round_delta_blend_vs_no_blend="
        f"{summary['mean_round_delta_blend_vs_no_blend']:+.6f}"
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
    seed_rows: list[SeedBlendRow],
    round_rows: list[RoundBlendRow],
) -> dict[str, object]:
    round_deltas = [row.delta_blend_vs_no_blend for row in round_rows]
    return {
        "rounds_evaluated": len(round_rows),
        "seeds_evaluated": len(seed_rows),
        "mean_seed_score_no_blend": _mean([row.score_no_blend for row in seed_rows]),
        "mean_seed_score_blend": _mean([row.score_blend for row in seed_rows]),
        "mean_seed_delta_blend_vs_no_blend": _mean(
            [row.delta_blend_vs_no_blend for row in seed_rows]
        ),
        "mean_round_score_no_blend": _mean([row.round_score_no_blend for row in round_rows]),
        "mean_round_score_blend": _mean([row.round_score_blend for row in round_rows]),
        "mean_round_delta_blend_vs_no_blend": _mean(round_deltas),
        "median_round_delta_blend_vs_no_blend": _median(round_deltas),
        "positive_round_delta_count": sum(1 for value in round_deltas if value > 0.0),
        "negative_round_delta_count": sum(1 for value in round_deltas if value < 0.0),
        "flat_round_delta_count": sum(1 for value in round_deltas if value == 0.0),
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
