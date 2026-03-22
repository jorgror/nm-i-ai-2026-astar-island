#!/usr/bin/env python3
"""Step 14 evaluation: learned Baseline-B world model as latent base predictor."""

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

from astar_island.baseline_b import BaselineBConfig
from astar_island.offline_emulator import run_offline_round
from astar_island.query_policy import (
    DeterministicThreePhasePolicyConfig,
    DeterministicThreePhaseQueryPolicy,
)
from astar_island.reproducibility import build_round_dataset_fingerprint
from astar_island.round_data import load_leave_one_round_out, load_round_dataset
from astar_island.round_latent import RoundLatentConditionalModel, RoundLatentConfig
from astar_island.world_model import (
    BaselineBWorldModelPredictor,
    train_baseline_b_world_model_from_rounds,
)


@dataclass(slots=True)
class SeedStep14Row:
    round_id: str
    round_number: int | None
    seed_index: int
    score_prior_latent_blend: float
    score_world_latent_blend: float
    delta_world_vs_prior: float
    weighted_kl_prior_latent_blend: float
    weighted_kl_world_latent_blend: float


@dataclass(slots=True)
class RoundStep14Row:
    round_id: str
    round_number: int | None
    queries_used_prior_latent_blend: int
    queries_used_world_latent_blend: int
    round_score_prior_latent_blend: float
    round_score_world_latent_blend: float
    delta_world_vs_prior: float
    mean_seed_weighted_kl_prior_latent_blend: float
    mean_seed_weighted_kl_world_latent_blend: float
    world_model_rounds_used: int
    world_model_samples_used: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-root", default=str(REPO_ROOT / "logs"))
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs" / "step14_world_model_eval"),
    )
    parser.add_argument("--query-budget", type=int, default=50)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--probability-floor", type=float, default=1e-4)

    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--samples-per-epoch", type=int, default=20000)
    parser.add_argument("--max-cells-per-seed", type=int, default=1000)
    parser.add_argument("--entropy-weight-power", type=float, default=0.8)
    parser.add_argument("--min-entropy-weight", type=float, default=0.02)
    parser.add_argument("--random-seed", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rounds = load_round_dataset(
        args.logs_root,
        include_replays=True,
        strict=args.strict,
    )
    rounds_sorted = sorted(
        all_rounds,
        key=lambda row: (
            row.round_number is None,
            int(row.round_number or 0),
            row.round_id,
        ),
    )
    dataset_fingerprint = build_round_dataset_fingerprint(
        rounds=rounds_sorted,
        logs_root=args.logs_root,
    )

    splits = load_leave_one_round_out(
        args.logs_root,
        include_replays=True,
        strict=args.strict,
    )
    splits = sorted(
        splits,
        key=lambda split: (
            split.validation_round.round_number is None,
            int(split.validation_round.round_number or 0),
            split.validation_round.round_id,
        ),
    )

    b_cfg = BaselineBConfig(
        learning_rate=float(args.learning_rate),
        epochs=int(args.epochs),
        l2=float(args.l2),
        samples_per_epoch=int(args.samples_per_epoch),
        max_cells_per_seed=int(args.max_cells_per_seed),
        entropy_weight_power=float(args.entropy_weight_power),
        min_entropy_weight=float(args.min_entropy_weight),
        probability_floor=float(args.probability_floor),
        random_seed=int(args.random_seed),
    )

    seed_rows: list[SeedStep14Row] = []
    round_rows: list[RoundStep14Row] = []

    for split in splits:
        holdout = split.validation_round

        prior_result = run_offline_round(
            policy=DeterministicThreePhaseQueryPolicy(
                config=DeterministicThreePhasePolicyConfig(query_budget=args.query_budget)
            ),
            model=RoundLatentConditionalModel(
                config=RoundLatentConfig(enable_observation_blend=True)
            ),
            round_id=holdout.round_id,
            round_record=holdout,
            logs_root=args.logs_root,
            query_budget=args.query_budget,
            include_replays=True,
            strict=args.strict,
            allow_missing_replays=False,
            probability_floor=args.probability_floor,
        )

        trained_world = train_baseline_b_world_model_from_rounds(
            rounds=split.training_rounds,
            config=b_cfg,
        )
        world_predictor = BaselineBWorldModelPredictor(
            model=trained_world.model,
            probability_floor=b_cfg.probability_floor,
        )
        world_result = run_offline_round(
            policy=DeterministicThreePhaseQueryPolicy(
                config=DeterministicThreePhasePolicyConfig(query_budget=args.query_budget)
            ),
            model=RoundLatentConditionalModel(
                config=RoundLatentConfig(enable_observation_blend=True),
                base_predictor=world_predictor,
            ),
            round_id=holdout.round_id,
            round_record=holdout,
            logs_root=args.logs_root,
            query_budget=args.query_budget,
            include_replays=True,
            strict=args.strict,
            allow_missing_replays=False,
            probability_floor=args.probability_floor,
        )

        prior_by_seed = {row.seed_index: row for row in prior_result.per_seed}
        world_by_seed = {row.seed_index: row for row in world_result.per_seed}
        for seed_index in sorted(set(prior_by_seed) | set(world_by_seed)):
            prior_seed = prior_by_seed[seed_index]
            world_seed = world_by_seed[seed_index]
            seed_rows.append(
                SeedStep14Row(
                    round_id=holdout.round_id,
                    round_number=holdout.round_number,
                    seed_index=seed_index,
                    score_prior_latent_blend=prior_seed.score,
                    score_world_latent_blend=world_seed.score,
                    delta_world_vs_prior=world_seed.score - prior_seed.score,
                    weighted_kl_prior_latent_blend=prior_seed.weighted_kl,
                    weighted_kl_world_latent_blend=world_seed.weighted_kl,
                )
            )

        round_rows.append(
            RoundStep14Row(
                round_id=holdout.round_id,
                round_number=holdout.round_number,
                queries_used_prior_latent_blend=prior_result.queries_used,
                queries_used_world_latent_blend=world_result.queries_used,
                round_score_prior_latent_blend=prior_result.round_score,
                round_score_world_latent_blend=world_result.round_score,
                delta_world_vs_prior=world_result.round_score - prior_result.round_score,
                mean_seed_weighted_kl_prior_latent_blend=_mean(
                    [row.weighted_kl for row in prior_result.per_seed]
                ),
                mean_seed_weighted_kl_world_latent_blend=_mean(
                    [row.weighted_kl for row in world_result.per_seed]
                ),
                world_model_rounds_used=trained_world.rounds_used,
                world_model_samples_used=trained_world.samples_used,
            )
        )

    seed_csv = output_dir / "seed_results.csv"
    round_csv = output_dir / "round_results.csv"
    summary_json = output_dir / "summary.json"
    run_summary_json = output_dir / "run_summary.json"

    _write_csv(seed_csv, seed_rows)
    _write_csv(round_csv, round_rows)

    summary = {
        "query_budget": int(args.query_budget),
        "rounds_evaluated": len(round_rows),
        "seeds_evaluated": len(seed_rows),
        "dataset_fingerprint": dataset_fingerprint,
        "world_model_config": asdict(b_cfg),
        "mean_round_score_prior_latent_blend": _mean(
            [row.round_score_prior_latent_blend for row in round_rows]
        ),
        "mean_round_score_world_latent_blend": _mean(
            [row.round_score_world_latent_blend for row in round_rows]
        ),
        "mean_round_delta_world_vs_prior": _mean(
            [row.delta_world_vs_prior for row in round_rows]
        ),
        "median_round_delta_world_vs_prior": _median(
            [row.delta_world_vs_prior for row in round_rows]
        ),
        "positive_round_delta_count": sum(1 for row in round_rows if row.delta_world_vs_prior > 0.0),
        "negative_round_delta_count": sum(1 for row in round_rows if row.delta_world_vs_prior < 0.0),
        "flat_round_delta_count": sum(1 for row in round_rows if row.delta_world_vs_prior == 0.0),
        "mean_seed_score_prior_latent_blend": _mean(
            [row.score_prior_latent_blend for row in seed_rows]
        ),
        "mean_seed_score_world_latent_blend": _mean(
            [row.score_world_latent_blend for row in seed_rows]
        ),
        "mean_seed_delta_world_vs_prior": _mean(
            [row.delta_world_vs_prior for row in seed_rows]
        ),
        "mean_seed_weighted_kl_prior_latent_blend": _mean(
            [row.weighted_kl_prior_latent_blend for row in seed_rows]
        ),
        "mean_seed_weighted_kl_world_latent_blend": _mean(
            [row.weighted_kl_world_latent_blend for row in seed_rows]
        ),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    run_summary = {
        "query_budget": int(args.query_budget),
        "rounds_evaluated": len(round_rows),
        "seeds_evaluated": len(seed_rows),
        "dataset_fingerprint": dataset_fingerprint,
        "mean_round_score_prior_latent_blend": summary["mean_round_score_prior_latent_blend"],
        "mean_round_score_world_latent_blend": summary["mean_round_score_world_latent_blend"],
        "mean_round_delta_world_vs_prior": summary["mean_round_delta_world_vs_prior"],
        "output_files": {
            "seed_csv": str(seed_csv),
            "round_csv": str(round_csv),
            "summary_json": str(summary_json),
        },
    }
    run_summary_json.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    print("Step 14 world-model evaluation complete.")
    print(f"rounds_evaluated={len(round_rows)}")
    print(f"seeds_evaluated={len(seed_rows)}")
    print(
        "mean_round_score_prior_latent_blend="
        f"{summary['mean_round_score_prior_latent_blend']:.6f}"
    )
    print(
        "mean_round_score_world_latent_blend="
        f"{summary['mean_round_score_world_latent_blend']:.6f}"
    )
    print(
        "mean_round_delta_world_vs_prior="
        f"{summary['mean_round_delta_world_vs_prior']:+.6f}"
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
