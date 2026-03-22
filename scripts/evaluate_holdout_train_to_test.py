#!/usr/bin/env python3
"""Evaluate train<=N, test=M holdout scenario with default vs tuned Step-14 stacks."""

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
from astar_island.round_data import RoundRecord, load_round_dataset
from astar_island.round_latent import RoundLatentConditionalModel, RoundLatentConfig
from astar_island.world_model import (
    BaselineBWorldModelPredictor,
    train_baseline_b_world_model_from_rounds,
)


@dataclass(slots=True)
class ScenarioRow:
    scenario: str
    train_max_round_number: int
    test_round_number: int
    round_id: str
    queries_used: int
    round_score: float
    mean_seed_weighted_kl: float
    world_rounds_used: int
    world_samples_used: int


@dataclass(slots=True)
class SeedScenarioRow:
    scenario: str
    round_id: str
    seed_index: int
    score: float
    weighted_kl: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-root", default=str(REPO_ROOT / "logs"))
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs" / "holdout_train21_test22"),
    )
    parser.add_argument("--train-max-round-number", type=int, default=21)
    parser.add_argument("--test-round-number", type=int, default=22)
    parser.add_argument("--query-budget", type=int, default=50)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rounds = load_round_dataset(
        args.logs_root,
        include_replays=True,
        strict=args.strict,
    )
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
        logs_root=args.logs_root,
    )

    train_rounds = [
        row
        for row in rounds_sorted
        if row.round_number is not None
        and int(row.round_number) <= int(args.train_max_round_number)
    ]
    test_candidates = [
        row
        for row in rounds_sorted
        if row.round_number is not None
        and int(row.round_number) == int(args.test_round_number)
    ]
    if not test_candidates:
        raise ValueError(f"Missing test round_number={args.test_round_number}")
    test_round = test_candidates[-1]

    scenario_rows: list[ScenarioRow] = []
    seed_rows: list[SeedScenarioRow] = []

    # Scenario A: default Step-14 stack.
    default_world_cfg = BaselineBConfig()
    default_latent_cfg = RoundLatentConfig(enable_observation_blend=True)
    default_policy_cfg = DeterministicThreePhasePolicyConfig(query_budget=args.query_budget)
    default_row, default_seed_rows = _evaluate_world_scenario(
        scenario="step14_default_world_latent_blend",
        world_cfg=default_world_cfg,
        latent_cfg=default_latent_cfg,
        policy_cfg=default_policy_cfg,
        train_rounds=train_rounds,
        test_round=test_round,
        logs_root=args.logs_root,
        query_budget=args.query_budget,
        strict=args.strict,
    )
    scenario_rows.append(default_row)
    seed_rows.extend(default_seed_rows)

    # Scenario B: tuned holdout stack from targeted search.
    tuned_world_cfg = BaselineBConfig(
        learning_rate=0.07,
        epochs=4,
        l2=1e-4,
        samples_per_epoch=20000,
        max_cells_per_seed=1600,
        entropy_weight_power=0.8,
        min_entropy_weight=0.02,
        probability_floor=1e-4,
        random_seed=7,
    )
    tuned_latent_cfg = RoundLatentConfig(
        probability_floor=1e-4,
        enable_observation_blend=True,
        empirical_prior_strength=0.7,
        observation_confidence_scale=2.4,
        repeated_observation_bonus=0.1,
        max_observed_blend_weight=0.9,
        dynamic_blend_boost=0.3,
    )
    tuned_policy_cfg = DeterministicThreePhasePolicyConfig(
        query_budget=args.query_budget,
        default_window=15,
        phase1_target=10,
        phase2_target=25,
        phase3_target=15,
        top_windows_per_seed=6,
        min_center_distance=6.0,
    )
    tuned_row, tuned_seed_rows = _evaluate_world_scenario(
        scenario="step14_tuned_world_latent_blend",
        world_cfg=tuned_world_cfg,
        latent_cfg=tuned_latent_cfg,
        policy_cfg=tuned_policy_cfg,
        train_rounds=train_rounds,
        test_round=test_round,
        logs_root=args.logs_root,
        query_budget=args.query_budget,
        strict=args.strict,
    )
    scenario_rows.append(tuned_row)
    seed_rows.extend(tuned_seed_rows)

    # Scenario C: no world model (latent+blend over mechanics prior).
    prior_row, prior_seed_rows = _evaluate_prior_scenario(
        scenario="prior_latent_blend",
        latent_cfg=RoundLatentConfig(enable_observation_blend=True),
        policy_cfg=DeterministicThreePhasePolicyConfig(query_budget=args.query_budget),
        test_round=test_round,
        logs_root=args.logs_root,
        query_budget=args.query_budget,
        strict=args.strict,
    )
    scenario_rows.append(prior_row)
    seed_rows.extend(prior_seed_rows)

    by_name = {row.scenario: row for row in scenario_rows}
    summary = {
        "train_max_round_number": int(args.train_max_round_number),
        "test_round_number": int(args.test_round_number),
        "query_budget": int(args.query_budget),
        "dataset_fingerprint": dataset_fingerprint,
        "scenarios": [asdict(row) for row in scenario_rows],
        "deltas": {
            "tuned_vs_default": (
                by_name["step14_tuned_world_latent_blend"].round_score
                - by_name["step14_default_world_latent_blend"].round_score
            ),
            "tuned_vs_prior": (
                by_name["step14_tuned_world_latent_blend"].round_score
                - by_name["prior_latent_blend"].round_score
            ),
            "default_vs_prior": (
                by_name["step14_default_world_latent_blend"].round_score
                - by_name["prior_latent_blend"].round_score
            ),
        },
    }

    scenario_csv = output_dir / "scenario_summary.csv"
    seed_csv = output_dir / "seed_scenario_results.csv"
    summary_json = output_dir / "summary.json"
    run_summary_json = output_dir / "run_summary.json"

    _write_csv(scenario_csv, scenario_rows)
    _write_csv(seed_csv, seed_rows)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    run_summary_json.write_text(
        json.dumps(
            {
                "train_max_round_number": int(args.train_max_round_number),
                "test_round_number": int(args.test_round_number),
                "query_budget": int(args.query_budget),
                "dataset_fingerprint": dataset_fingerprint,
                "best_scenario": max(scenario_rows, key=lambda row: row.round_score).scenario,
                "best_round_score": max(row.round_score for row in scenario_rows),
                "output_files": {
                    "scenario_csv": str(scenario_csv),
                    "seed_csv": str(seed_csv),
                    "summary_json": str(summary_json),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Holdout evaluation complete.")
    print(f"train_max_round_number={args.train_max_round_number}")
    print(f"test_round_number={args.test_round_number}")
    for row in scenario_rows:
        print(
            f"scenario={row.scenario} round_score={row.round_score:.6f} "
            f"mean_seed_wkl={row.mean_seed_weighted_kl:.6f}"
        )
    print("deltas:")
    print(f"  tuned_vs_default={summary['deltas']['tuned_vs_default']:+.6f}")
    print(f"  tuned_vs_prior={summary['deltas']['tuned_vs_prior']:+.6f}")
    print(f"  default_vs_prior={summary['deltas']['default_vs_prior']:+.6f}")
    print("output_files:")
    print(f"  scenario_csv: {scenario_csv}")
    print(f"  seed_csv: {seed_csv}")
    print(f"  summary_json: {summary_json}")
    print(f"  run_summary_json: {run_summary_json}")
    return 0


def _evaluate_world_scenario(
    *,
    scenario: str,
    world_cfg: BaselineBConfig,
    latent_cfg: RoundLatentConfig,
    policy_cfg: DeterministicThreePhasePolicyConfig,
    train_rounds: list[RoundRecord],
    test_round: RoundRecord,
    logs_root: str,
    query_budget: int,
    strict: bool,
) -> tuple[ScenarioRow, list[SeedScenarioRow]]:
    trained = train_baseline_b_world_model_from_rounds(
        rounds=train_rounds,
        config=world_cfg,
    )
    predictor = BaselineBWorldModelPredictor(
        model=trained.model,
        probability_floor=world_cfg.probability_floor,
    )
    model = RoundLatentConditionalModel(
        config=latent_cfg,
        base_predictor=predictor,
    )
    policy = DeterministicThreePhaseQueryPolicy(config=policy_cfg)
    result = run_offline_round(
        policy=policy,
        model=model,
        round_id=test_round.round_id,
        round_record=test_round,
        logs_root=logs_root,
        query_budget=query_budget,
        include_replays=True,
        strict=strict,
        allow_missing_replays=False,
        probability_floor=world_cfg.probability_floor,
    )
    row = ScenarioRow(
        scenario=scenario,
        train_max_round_number=max(int(r.round_number or 0) for r in train_rounds),
        test_round_number=int(test_round.round_number or 0),
        round_id=test_round.round_id,
        queries_used=result.queries_used,
        round_score=result.round_score,
        mean_seed_weighted_kl=_mean([seed.weighted_kl for seed in result.per_seed]),
        world_rounds_used=trained.rounds_used,
        world_samples_used=trained.samples_used,
    )
    seed_rows = [
        SeedScenarioRow(
            scenario=scenario,
            round_id=test_round.round_id,
            seed_index=seed.seed_index,
            score=seed.score,
            weighted_kl=seed.weighted_kl,
        )
        for seed in result.per_seed
    ]
    return row, seed_rows


def _evaluate_prior_scenario(
    *,
    scenario: str,
    latent_cfg: RoundLatentConfig,
    policy_cfg: DeterministicThreePhasePolicyConfig,
    test_round: RoundRecord,
    logs_root: str,
    query_budget: int,
    strict: bool,
) -> tuple[ScenarioRow, list[SeedScenarioRow]]:
    model = RoundLatentConditionalModel(config=latent_cfg)
    policy = DeterministicThreePhaseQueryPolicy(config=policy_cfg)
    result = run_offline_round(
        policy=policy,
        model=model,
        round_id=test_round.round_id,
        round_record=test_round,
        logs_root=logs_root,
        query_budget=query_budget,
        include_replays=True,
        strict=strict,
        allow_missing_replays=False,
        probability_floor=latent_cfg.probability_floor,
    )
    row = ScenarioRow(
        scenario=scenario,
        train_max_round_number=0,
        test_round_number=int(test_round.round_number or 0),
        round_id=test_round.round_id,
        queries_used=result.queries_used,
        round_score=result.round_score,
        mean_seed_weighted_kl=_mean([seed.weighted_kl for seed in result.per_seed]),
        world_rounds_used=0,
        world_samples_used=0,
    )
    seed_rows = [
        SeedScenarioRow(
            scenario=scenario,
            round_id=test_round.round_id,
            seed_index=seed.seed_index,
            score=seed.score,
            weighted_kl=seed.weighted_kl,
        )
        for seed in result.per_seed
    ]
    return row, seed_rows


def _write_csv(path: Path, rows: list[object]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


if __name__ == "__main__":
    raise SystemExit(main())
