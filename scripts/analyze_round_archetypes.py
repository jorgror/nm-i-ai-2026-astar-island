#!/usr/bin/env python3
"""Build round fingerprints + clustering artifacts for step 6."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from astar_island.archetypes import (
    COMPACT_FEATURES,
    DEFAULT_FEATURES,
    DYNAMICS_FEATURES,
    build_archetype_report,
)
from astar_island.round_data import load_round_dataset


FEATURE_SETS = {
    "default": DEFAULT_FEATURES,
    "compact": COMPACT_FEATURES,
    "dynamics": DYNAMICS_FEATURES,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-root",
        default=str(REPO_ROOT / "logs"),
        help="Root folder with per-round logs directories.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs" / "step6_round_archetypes"),
        help="Output directory for tables and plots.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Target number of k-means clusters.",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=8,
        help="Max k to evaluate in elbow plot.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=7,
        help="Deterministic seed for clustering + PCA.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require all analysis files to exist when loading dataset.",
    )
    parser.add_argument(
        "--feature-set",
        default="default",
        choices=sorted(FEATURE_SETS.keys()),
        help="Named feature set for clustering.",
    )
    parser.add_argument(
        "--features",
        default=None,
        help="Optional comma-separated feature names to override --feature-set.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.features:
        feature_names = [item.strip() for item in args.features.split(",") if item.strip()]
        if not feature_names:
            raise ValueError("--features provided but no valid names parsed")
    else:
        feature_names = FEATURE_SETS[args.feature_set]

    rounds = load_round_dataset(
        args.logs_root,
        include_replays=True,
        strict=args.strict,
    )

    report = build_archetype_report(
        rounds,
        k=args.k,
        max_k=args.max_k,
        random_seed=args.random_seed,
        feature_names=feature_names,
    )
    output_paths = report.write(args.output_dir)

    print(f"rounds_loaded={len(rounds)}")
    print(f"fingerprints={len(report.fingerprints)}")
    print(f"clusters_k={report.clustering.k}")
    print(f"feature_count={len(feature_names)}")
    print(f"feature_names={','.join(feature_names)}")
    print(f"inertia={report.clustering.inertia:.6f}")
    print("output_files:")
    for key, value in output_paths.items():
        print(f"  {key}: {value}")

    summary_path = Path(args.output_dir) / "report_summary.json"
    summary_payload = {
        "rounds_loaded": len(rounds),
        "fingerprints": len(report.fingerprints),
        "clusters_k": report.clustering.k,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "inertia": report.clustering.inertia,
        "output_files": output_paths,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
