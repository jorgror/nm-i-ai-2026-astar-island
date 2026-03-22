#!/usr/bin/env python3
"""Run Baseline B leave-one-round-out evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from astar_island.baseline_b import BaselineBConfig, evaluate_baseline_b_leave_one_round_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-root",
        default=str(REPO_ROOT / "logs"),
        help="Logs root with historical rounds.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs" / "step8_baseline_b"),
        help="Output folder for LOO evaluation artifacts.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--samples-per-epoch", type=int, default=20000)
    parser.add_argument("--max-cells-per-seed", type=int, default=1000)
    parser.add_argument("--entropy-weight-power", type=float, default=0.8)
    parser.add_argument("--min-entropy-weight", type=float, default=0.02)
    parser.add_argument("--probability-floor", type=float, default=1e-4)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require all rounds/seeds to contain required files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = BaselineBConfig(
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        l2=args.l2,
        samples_per_epoch=args.samples_per_epoch,
        max_cells_per_seed=args.max_cells_per_seed,
        entropy_weight_power=args.entropy_weight_power,
        min_entropy_weight=args.min_entropy_weight,
        probability_floor=args.probability_floor,
        random_seed=args.random_seed,
    )

    report = evaluate_baseline_b_leave_one_round_out(
        logs_root=args.logs_root,
        config=config,
        strict=args.strict,
    )
    output_paths = report.write(args.output_dir)

    print("Baseline B (LOO) complete.")
    print(f"rounds_evaluated={len(report.round_results)}")
    print(f"seeds_evaluated={len(report.seed_results)}")
    print(f"mean_seed_score_baseline_b={report.mean_seed_score_baseline_b:.6f}")
    print(f"mean_seed_score_prior_a={report.mean_seed_score_prior_a:.6f}")
    print(f"mean_seed_gain_vs_prior_a={report.mean_seed_gain_vs_prior_a:+.6f}")
    print(f"mean_round_score_baseline_b={report.mean_round_score_baseline_b:.6f}")
    print(f"mean_round_score_prior_a={report.mean_round_score_prior_a:.6f}")
    print(f"mean_round_gain_vs_prior_a={report.mean_round_gain_vs_prior_a:+.6f}")
    if report.mean_training_weighted_kl_by_epoch:
        print("mean_training_weighted_kl_by_epoch:")
        for epoch_idx, value in enumerate(report.mean_training_weighted_kl_by_epoch, start=1):
            print(f"  epoch_{epoch_idx}: {value:.6f}")
    if report.mean_training_round_score_by_epoch:
        print("mean_training_round_score_by_epoch:")
        for epoch_idx, value in enumerate(report.mean_training_round_score_by_epoch, start=1):
            print(f"  epoch_{epoch_idx}: {value:.6f}")
    print("output_files:")
    for key, value in output_paths.items():
        print(f"  {key}: {value}")

    extra = {
        "rounds_evaluated": len(report.round_results),
        "seeds_evaluated": len(report.seed_results),
        "mean_seed_score_baseline_b": report.mean_seed_score_baseline_b,
        "mean_seed_score_prior_a": report.mean_seed_score_prior_a,
        "mean_seed_gain_vs_prior_a": report.mean_seed_gain_vs_prior_a,
        "mean_round_score_baseline_b": report.mean_round_score_baseline_b,
        "mean_round_score_prior_a": report.mean_round_score_prior_a,
        "mean_round_gain_vs_prior_a": report.mean_round_gain_vs_prior_a,
        "mean_training_weighted_kl_by_epoch": report.mean_training_weighted_kl_by_epoch,
        "mean_training_round_score_by_epoch": report.mean_training_round_score_by_epoch,
        "output_files": output_paths,
    }
    summary_path = Path(args.output_dir) / "run_summary.json"
    summary_path.write_text(json.dumps(extra, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
