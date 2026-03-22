#!/usr/bin/env python3
"""Select stable world/latent/policy profiles via rolling holdout backtests."""

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
class CandidateProfile:
    profile_id: str
    world_id: str
    latent_id: str
    policy_id: str
    world_cfg: BaselineBConfig
    latent_cfg: RoundLatentConfig
    policy_cfg: DeterministicThreePhasePolicyConfig


@dataclass(slots=True)
class HoldoutRow:
    profile_id: str
    world_id: str
    latent_id: str
    policy_id: str
    train_max_round_number: int
    test_round_number: int
    round_id: str
    round_score: float
    mean_seed_weighted_kl: float


@dataclass(slots=True)
class ProfileSummaryRow:
    profile_id: str
    world_id: str
    latent_id: str
    policy_id: str
    holdouts_evaluated: int
    mean_round_score: float
    median_round_score: float
    mean_seed_weighted_kl: float
    std_round_score: float
    p10_round_score: float
    p25_round_score: float
    min_round_score: float
    max_round_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-root", default=str(REPO_ROOT / "logs"))
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs" / "rolling_profile_selection"),
    )
    parser.add_argument("--query-budget", type=int, default=50)
    parser.add_argument(
        "--min-test-round-number",
        type=int,
        default=12,
        help="Evaluate holdouts for round_number >= this value.",
    )
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
    rounds = [row for row in rounds if row.round_number is not None]
    rounds_sorted = sorted(rounds, key=lambda row: int(row.round_number or 0))
    if len(rounds_sorted) < 3:
        raise ValueError("Need at least 3 rounds for rolling profile selection.")

    dataset_fingerprint = build_round_dataset_fingerprint(
        rounds=rounds_sorted,
        logs_root=args.logs_root,
    )

    holdout_rounds = [
        row
        for row in rounds_sorted
        if int(row.round_number or 0) >= int(args.min_test_round_number)
    ]
    if not holdout_rounds:
        raise ValueError(
            f"No holdout rounds found for min_test_round_number={args.min_test_round_number}"
        )

    candidates = _build_candidates(query_budget=int(args.query_budget))
    print(f"candidate_count={len(candidates)}")
    print(f"holdout_count={len(holdout_rounds)}")

    rows: list[HoldoutRow] = []
    trained_cache: dict[tuple[str, int], BaselineBWorldModelPredictor] = {}

    for holdout in holdout_rounds:
        train_rounds = [
            row
            for row in rounds_sorted
            if int(row.round_number or 0) < int(holdout.round_number or 0)
        ]
        if not train_rounds:
            continue
        train_max_round_number = max(int(row.round_number or 0) for row in train_rounds)
        for idx, candidate in enumerate(candidates, start=1):
            predictor = _get_or_train_predictor(
                cache=trained_cache,
                world_id=candidate.world_id,
                world_cfg=candidate.world_cfg,
                train_rounds=train_rounds,
                train_max_round_number=train_max_round_number,
            )
            model = RoundLatentConditionalModel(
                config=candidate.latent_cfg,
                base_predictor=predictor,
            )
            policy = DeterministicThreePhaseQueryPolicy(config=candidate.policy_cfg)
            result = run_offline_round(
                policy=policy,
                model=model,
                round_id=holdout.round_id,
                round_record=holdout,
                logs_root=args.logs_root,
                query_budget=args.query_budget,
                include_replays=True,
                strict=args.strict,
                allow_missing_replays=False,
                probability_floor=candidate.world_cfg.probability_floor,
            )
            rows.append(
                HoldoutRow(
                    profile_id=candidate.profile_id,
                    world_id=candidate.world_id,
                    latent_id=candidate.latent_id,
                    policy_id=candidate.policy_id,
                    train_max_round_number=train_max_round_number,
                    test_round_number=int(holdout.round_number or 0),
                    round_id=holdout.round_id,
                    round_score=result.round_score,
                    mean_seed_weighted_kl=_mean([seed.weighted_kl for seed in result.per_seed]),
                )
            )
            if idx == len(candidates):
                print(
                    "evaluated holdout "
                    f"test_round={int(holdout.round_number or 0)} "
                    f"train_max={train_max_round_number}"
                )

    summaries = _summarize(rows, candidates)
    summaries_sorted = sorted(
        summaries,
        key=lambda row: (
            -row.mean_round_score,
            -row.p25_round_score,
            -row.median_round_score,
        ),
    )
    best = summaries_sorted[0]

    holdout_csv = output_dir / "holdout_results.csv"
    profile_csv = output_dir / "profile_summary.csv"
    summary_json = output_dir / "summary.json"
    run_summary_json = output_dir / "run_summary.json"
    best_profile_json = output_dir / "best_profile.json"

    _write_csv(holdout_csv, rows)
    _write_csv(profile_csv, summaries_sorted)

    summary = {
        "query_budget": int(args.query_budget),
        "min_test_round_number": int(args.min_test_round_number),
        "holdout_count": len(holdout_rounds),
        "candidate_count": len(candidates),
        "dataset_fingerprint": dataset_fingerprint,
        "best_profile": asdict(best),
        "profiles": [asdict(row) for row in summaries_sorted],
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    best_candidate = next(row for row in candidates if row.profile_id == best.profile_id)
    best_profile_json.write_text(
        json.dumps(
            {
                "profile_id": best.profile_id,
                "world_id": best_candidate.world_id,
                "latent_id": best_candidate.latent_id,
                "policy_id": best_candidate.policy_id,
                "world_cfg": asdict(best_candidate.world_cfg),
                "latent_cfg": asdict(best_candidate.latent_cfg),
                "policy_cfg": asdict(best_candidate.policy_cfg),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    run_summary_json.write_text(
        json.dumps(
            {
                "query_budget": int(args.query_budget),
                "min_test_round_number": int(args.min_test_round_number),
                "holdout_count": len(holdout_rounds),
                "candidate_count": len(candidates),
                "best_profile_id": best.profile_id,
                "best_mean_round_score": best.mean_round_score,
                "best_p25_round_score": best.p25_round_score,
                "best_median_round_score": best.median_round_score,
                "output_files": {
                    "holdout_csv": str(holdout_csv),
                    "profile_csv": str(profile_csv),
                    "summary_json": str(summary_json),
                    "best_profile_json": str(best_profile_json),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Rolling profile selection complete.")
    print(f"best_profile_id={best.profile_id}")
    print(f"best_mean_round_score={best.mean_round_score:.6f}")
    print(f"best_p25_round_score={best.p25_round_score:.6f}")
    print(f"best_median_round_score={best.median_round_score:.6f}")
    print("output_files:")
    print(f"  holdout_csv: {holdout_csv}")
    print(f"  profile_csv: {profile_csv}")
    print(f"  summary_json: {summary_json}")
    print(f"  run_summary_json: {run_summary_json}")
    print(f"  best_profile_json: {best_profile_json}")
    return 0


def _build_candidates(*, query_budget: int) -> list[CandidateProfile]:
    world_grid: dict[str, BaselineBConfig] = {
        "w_default": BaselineBConfig(
            learning_rate=0.06,
            epochs=4,
            l2=1e-4,
            samples_per_epoch=20000,
            max_cells_per_seed=1000,
            entropy_weight_power=0.8,
            min_entropy_weight=0.02,
            probability_floor=1e-4,
            random_seed=7,
        ),
        "w_more_cells": BaselineBConfig(
            learning_rate=0.07,
            epochs=4,
            l2=1e-4,
            samples_per_epoch=20000,
            max_cells_per_seed=1600,
            entropy_weight_power=0.8,
            min_entropy_weight=0.02,
            probability_floor=1e-4,
            random_seed=7,
        ),
        "w_conservative": BaselineBConfig(
            learning_rate=0.05,
            epochs=4,
            l2=1e-4,
            samples_per_epoch=20000,
            max_cells_per_seed=1600,
            entropy_weight_power=0.8,
            min_entropy_weight=0.02,
            probability_floor=1e-4,
            random_seed=7,
        ),
    }

    latent_grid: dict[str, RoundLatentConfig] = {
        "l_default": RoundLatentConfig(
            probability_floor=1e-4,
            enable_observation_blend=True,
        ),
        "l_balanced": RoundLatentConfig(
            probability_floor=1e-4,
            enable_observation_blend=True,
            empirical_prior_strength=0.55,
            observation_confidence_scale=2.2,
            repeated_observation_bonus=0.12,
            max_observed_blend_weight=0.9,
            dynamic_blend_boost=0.35,
        ),
        "l_cautious": RoundLatentConfig(
            probability_floor=1e-4,
            enable_observation_blend=True,
            empirical_prior_strength=0.45,
            observation_confidence_scale=2.0,
            repeated_observation_bonus=0.15,
            max_observed_blend_weight=0.92,
            dynamic_blend_boost=0.4,
        ),
    }

    policy_grid: dict[str, DeterministicThreePhasePolicyConfig] = {
        "p_default": DeterministicThreePhasePolicyConfig(query_budget=query_budget),
        "p_focus_mid": DeterministicThreePhasePolicyConfig(
            query_budget=query_budget,
            phase1_target=10,
            phase2_target=25,
            phase3_target=15,
            top_windows_per_seed=6,
            min_center_distance=6.0,
        ),
        "p_balanced_alt": DeterministicThreePhasePolicyConfig(
            query_budget=query_budget,
            phase1_target=12,
            phase2_target=23,
            phase3_target=15,
            top_windows_per_seed=8,
            min_center_distance=4.5,
        ),
    }

    profiles: list[CandidateProfile] = []
    for world_id, world_cfg in world_grid.items():
        for latent_id, latent_cfg in latent_grid.items():
            for policy_id, policy_cfg in policy_grid.items():
                profile_id = f"{world_id}__{latent_id}__{policy_id}"
                profiles.append(
                    CandidateProfile(
                        profile_id=profile_id,
                        world_id=world_id,
                        latent_id=latent_id,
                        policy_id=policy_id,
                        world_cfg=world_cfg,
                        latent_cfg=latent_cfg,
                        policy_cfg=policy_cfg,
                    )
                )
    return profiles


def _get_or_train_predictor(
    *,
    cache: dict[tuple[str, int], BaselineBWorldModelPredictor],
    world_id: str,
    world_cfg: BaselineBConfig,
    train_rounds: list[RoundRecord],
    train_max_round_number: int,
) -> BaselineBWorldModelPredictor:
    key = (world_id, int(train_max_round_number))
    cached = cache.get(key)
    if cached is not None:
        return cached
    trained = train_baseline_b_world_model_from_rounds(
        rounds=train_rounds,
        config=world_cfg,
    )
    predictor = BaselineBWorldModelPredictor(
        model=trained.model,
        probability_floor=world_cfg.probability_floor,
    )
    cache[key] = predictor
    return predictor


def _summarize(
    rows: list[HoldoutRow],
    candidates: list[CandidateProfile],
) -> list[ProfileSummaryRow]:
    by_profile: dict[str, list[HoldoutRow]] = {}
    for row in rows:
        by_profile.setdefault(row.profile_id, []).append(row)

    summaries: list[ProfileSummaryRow] = []
    for candidate in candidates:
        profile_rows = by_profile.get(candidate.profile_id, [])
        scores = [row.round_score for row in profile_rows]
        weighted_kls = [row.mean_seed_weighted_kl for row in profile_rows]
        summaries.append(
            ProfileSummaryRow(
                profile_id=candidate.profile_id,
                world_id=candidate.world_id,
                latent_id=candidate.latent_id,
                policy_id=candidate.policy_id,
                holdouts_evaluated=len(profile_rows),
                mean_round_score=_mean(scores),
                median_round_score=_median(scores),
                mean_seed_weighted_kl=_mean(weighted_kls),
                std_round_score=_std(scores),
                p10_round_score=_quantile(scores, 0.10),
                p25_round_score=_quantile(scores, 0.25),
                min_round_score=min(scores) if scores else 0.0,
                max_round_score=max(scores) if scores else 0.0,
            )
        )
    return summaries


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


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0.0:
        return min(values)
    if q >= 1.0:
        return max(values)
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    variance = sum((value - mu) ** 2 for value in values) / float(len(values))
    return variance**0.5


if __name__ == "__main__":
    raise SystemExit(main())
