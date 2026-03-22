#!/usr/bin/env python3
"""Evaluate step-9 round-latent model against prior baselines."""

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

from astar_island.offline_emulator import OfflineRoundState, run_offline_round
from astar_island.priors import baseline_prior_from_initial_grid
from astar_island.query_policy import (
    DeterministicThreePhasePolicyConfig,
    DeterministicThreePhaseQueryPolicy,
)
from astar_island.reproducibility import build_round_dataset_fingerprint
from astar_island.round_data import load_round_dataset
from astar_island.round_latent import RoundLatentConditionalModel, RoundLatentConfig


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
    dataset_fingerprint = build_round_dataset_fingerprint(
        rounds=rounds_sorted,
        logs_root=logs_root,
    )

    seed_rows: list[SeedEvalRow] = []
    round_rows: list[RoundEvalRow] = []

    for record in rounds_sorted:
        no_query = run_offline_round(
            policy=NoQueryPolicy(),
            model=PriorOnlyModel(),
            round_id=record.round_id,
            round_record=record,
            logs_root=str(logs_root),
            query_budget=0,
            include_replays=True,
            strict=args.strict,
            allow_missing_replays=False,
            probability_floor=args.probability_floor,
        )
        query_prior = run_offline_round(
            policy=DeterministicThreePhaseQueryPolicy(
                config=DeterministicThreePhasePolicyConfig(query_budget=args.query_budget)
            ),
            model=PriorOnlyModel(),
            round_id=record.round_id,
            round_record=record,
            logs_root=str(logs_root),
            query_budget=args.query_budget,
            include_replays=True,
            strict=args.strict,
            allow_missing_replays=False,
            probability_floor=args.probability_floor,
        )
        query_latent = run_offline_round(
            policy=DeterministicThreePhaseQueryPolicy(
                config=DeterministicThreePhasePolicyConfig(query_budget=args.query_budget)
            ),
            model=RoundLatentConditionalModel(
                config=RoundLatentConfig(enable_observation_blend=False)
            ),
            round_id=record.round_id,
            round_record=record,
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
    summary["dataset_fingerprint"] = dataset_fingerprint
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    run_summary = {
        "rounds_evaluated": len(round_rows),
        "seeds_evaluated": len(seed_rows),
        "query_budget": args.query_budget,
        "dataset_fingerprint": dataset_fingerprint,
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
