#!/usr/bin/env python3
"""Run Baseline C leave-one-round-out evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from astar_island.baseline_c import BaselineCConfig, evaluate_baseline_c_leave_one_round_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-root",
        default=str(REPO_ROOT / "logs"),
        help="Logs root with historical rounds.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs" / "step7_baseline_c"),
        help="Output folder for LOO evaluation artifacts.",
    )
    parser.add_argument("--patch-radius", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--samples-per-epoch", type=int, default=25000)
    parser.add_argument("--max-cells-per-seed", type=int, default=1000)
    parser.add_argument("--probability-floor", type=float, default=1e-4)
    parser.add_argument("--random-seed", type=int, default=11)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require all rounds/seeds to contain required files.",
    )
    parser.add_argument(
        "--baseline-b-seed-csv",
        default=None,
        help="Optional path to Baseline B seed CSV for direct C-vs-B mean comparison.",
    )
    parser.add_argument(
        "--baseline-b-round-csv",
        default=None,
        help="Optional path to Baseline B round CSV for direct C-vs-B mean comparison.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = BaselineCConfig(
        patch_radius=args.patch_radius,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        l2=args.l2,
        samples_per_epoch=args.samples_per_epoch,
        max_cells_per_seed=args.max_cells_per_seed,
        probability_floor=args.probability_floor,
        random_seed=args.random_seed,
    )

    report = evaluate_baseline_c_leave_one_round_out(
        logs_root=args.logs_root,
        config=config,
        strict=args.strict,
    )
    output_paths = report.write(args.output_dir)

    print("Baseline C (LOO) complete.")
    print(f"rounds_evaluated={len(report.round_results)}")
    print(f"seeds_evaluated={len(report.seed_results)}")
    print(f"mean_seed_score_baseline_c={report.mean_seed_score_baseline_c:.6f}")
    print(f"mean_seed_score_prior_a={report.mean_seed_score_prior_a:.6f}")
    print(f"mean_seed_gain_vs_prior_a={report.mean_seed_gain_vs_prior_a:+.6f}")
    print(f"mean_round_score_baseline_c={report.mean_round_score_baseline_c:.6f}")
    print(f"mean_round_score_prior_a={report.mean_round_score_prior_a:.6f}")
    print(f"mean_round_gain_vs_prior_a={report.mean_round_gain_vs_prior_a:+.6f}")

    comparison: dict[str, float] = {}
    if args.baseline_b_seed_csv and Path(args.baseline_b_seed_csv).exists():
        b_seed_mean = _mean_csv_column(args.baseline_b_seed_csv, "score_baseline_b")
        c_minus_b_seed = report.mean_seed_score_baseline_c - b_seed_mean
        comparison["mean_seed_score_baseline_b"] = b_seed_mean
        comparison["mean_seed_gain_c_vs_b"] = c_minus_b_seed
        print(f"mean_seed_score_baseline_b={b_seed_mean:.6f}")
        print(f"mean_seed_gain_c_vs_b={c_minus_b_seed:+.6f}")

    if args.baseline_b_round_csv and Path(args.baseline_b_round_csv).exists():
        b_round_mean = _mean_csv_column(args.baseline_b_round_csv, "round_score_baseline_b")
        c_minus_b_round = report.mean_round_score_baseline_c - b_round_mean
        comparison["mean_round_score_baseline_b"] = b_round_mean
        comparison["mean_round_gain_c_vs_b"] = c_minus_b_round
        print(f"mean_round_score_baseline_b={b_round_mean:.6f}")
        print(f"mean_round_gain_c_vs_b={c_minus_b_round:+.6f}")

    print("output_files:")
    for key, value in output_paths.items():
        print(f"  {key}: {value}")

    extra = {
        "rounds_evaluated": len(report.round_results),
        "seeds_evaluated": len(report.seed_results),
        "mean_seed_score_baseline_c": report.mean_seed_score_baseline_c,
        "mean_seed_score_prior_a": report.mean_seed_score_prior_a,
        "mean_seed_gain_vs_prior_a": report.mean_seed_gain_vs_prior_a,
        "mean_round_score_baseline_c": report.mean_round_score_baseline_c,
        "mean_round_score_prior_a": report.mean_round_score_prior_a,
        "mean_round_gain_vs_prior_a": report.mean_round_gain_vs_prior_a,
        "comparison": comparison,
        "output_files": output_paths,
    }
    run_summary = Path(args.output_dir) / "run_summary.json"
    run_summary.write_text(json.dumps(extra, indent=2), encoding="utf-8")
    return 0


def _mean_csv_column(path: str | Path, column: str) -> float:
    values: list[float] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            values.append(float(row[column]))
    if not values:
        return 0.0
    return sum(values) / float(len(values))


if __name__ == "__main__":
    raise SystemExit(main())
