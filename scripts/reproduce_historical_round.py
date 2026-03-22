#!/usr/bin/env python3
"""Reproduce historical Astar Island per-seed and per-round scores locally."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from astar_island.models import RoundScoreResult, SeedScoreResult
from astar_island.parsing import load_json, parse_analysis_seed, parse_round_detail, parse_round_score
from astar_island.scoring import score_seed
from astar_island.submission import validate_prediction_tensor
from astar_island.visualization import write_argmax_svg, write_entropy_svg, write_initial_grid_svg

SEED_FILE_RE = re.compile(r"seed[_-]?(\d+).*\.json$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-json", required=True, help="Path to /rounds/{round_id} JSON")
    parser.add_argument(
        "--analysis-dir",
        required=True,
        help="Directory with per-seed analysis files (expects filenames containing seed index)",
    )
    parser.add_argument(
        "--my-round-json",
        default=None,
        help="Optional /my-rounds JSON (row or list) to compare local round score",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/reproduce_historical_round",
        help="Directory for score reports and visualizations",
    )
    parser.add_argument(
        "--sum-tolerance",
        type=float,
        default=0.01,
        help="Submission validation tolerance for per-cell probability sums",
    )
    parser.add_argument(
        "--score-tolerance",
        type=float,
        default=1e-6,
        help="Strict mode tolerance between local and official scores",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing seeds or score mismatches",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip SVG visualization output",
    )
    return parser.parse_args()


def discover_analysis_files(analysis_dir: Path) -> dict[int, Path]:
    files_by_seed: dict[int, Path] = {}
    preferred = sorted(analysis_dir.glob("analysis-seed-*.json"))
    if preferred:
        for path in preferred:
            match = SEED_FILE_RE.search(path.name)
            if not match:
                continue
            seed_index = int(match.group(1))
            files_by_seed[seed_index] = path
        return files_by_seed

    for path in sorted(analysis_dir.glob("*.json")):
        match = SEED_FILE_RE.search(path.name)
        if not match:
            continue
        seed_index = int(match.group(1))
        files_by_seed[seed_index] = path
    return files_by_seed


def main() -> int:
    args = parse_args()

    round_payload = load_json(args.round_json)
    round_detail = parse_round_detail(round_payload)

    analysis_dir = Path(args.analysis_dir)
    if not analysis_dir.exists():
        raise FileNotFoundError(f"Analysis directory not found: {analysis_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files_by_seed = discover_analysis_files(analysis_dir)

    per_seed: list[SeedScoreResult] = []

    for seed_index in range(round_detail.seeds_count):
        seed_file = files_by_seed.get(seed_index)
        if seed_file is None:
            per_seed.append(
                SeedScoreResult(
                    seed_index=seed_index,
                    weighted_kl=0.0,
                    score=0.0,
                    official_score=None,
                    score_diff=None,
                    source_path=None,
                    missing_submission=True,
                )
            )
            continue

        analysis_payload = load_json(seed_file)
        analysis = parse_analysis_seed(analysis_payload)

        if analysis.prediction is None:
            per_seed.append(
                SeedScoreResult(
                    seed_index=seed_index,
                    weighted_kl=0.0,
                    score=0.0,
                    official_score=analysis.score,
                    score_diff=(0.0 - analysis.score) if analysis.score is not None else None,
                    source_path=str(seed_file),
                    missing_submission=True,
                )
            )
            continue

        validation = validate_prediction_tensor(
            analysis.prediction,
            expected_width=analysis.width,
            expected_height=analysis.height,
            sum_tolerance=args.sum_tolerance,
        )
        if not validation.ok:
            problems = "\n".join(validation.errors[:20])
            raise ValueError(f"Invalid prediction tensor in {seed_file}:\n{problems}")

        wkl, score = score_seed(analysis.ground_truth, analysis.prediction)

        official_seed_score = analysis.score
        score_diff = None
        if official_seed_score is not None:
            score_diff = score - official_seed_score

        per_seed.append(
            SeedScoreResult(
                seed_index=seed_index,
                weighted_kl=wkl,
                score=score,
                official_score=official_seed_score,
                score_diff=score_diff,
                source_path=str(seed_file),
                missing_submission=False,
            )
        )

        if not args.no_viz:
            seed_dir = output_dir / f"seed_{seed_index}"
            if analysis.initial_grid is not None:
                write_initial_grid_svg(analysis.initial_grid, seed_dir / "initial_grid.svg")
            elif seed_index < len(round_detail.initial_states):
                write_initial_grid_svg(
                    round_detail.initial_states[seed_index].grid,
                    seed_dir / "initial_grid.svg",
                )

            write_argmax_svg(analysis.ground_truth, seed_dir / "ground_truth_argmax.svg")
            write_entropy_svg(analysis.ground_truth, seed_dir / "ground_truth_entropy.svg")
            write_argmax_svg(analysis.prediction, seed_dir / "prediction_argmax.svg")

    round_score = sum(seed.score for seed in per_seed) / float(round_detail.seeds_count)

    official_round_score = None
    if args.my_round_json:
        official_round_payload = load_json(args.my_round_json)
        official_round_score = parse_round_score(
            official_round_payload,
            round_id=round_detail.round_id,
        )

    round_score_diff = None
    if official_round_score is not None:
        round_score_diff = round_score - official_round_score

    result = RoundScoreResult(
        round_id=round_detail.round_id,
        per_seed=per_seed,
        round_score=round_score,
        official_round_score=official_round_score,
        round_score_diff=round_score_diff,
    )

    failed = False
    if args.strict:
        for seed in per_seed:
            if seed.missing_submission:
                print(f"ERROR: missing analysis file for seed {seed.seed_index}")
                failed = True
            elif seed.score_diff is not None and abs(seed.score_diff) > args.score_tolerance:
                print(
                    "ERROR: seed score mismatch "
                    f"seed={seed.seed_index} local={seed.score:.10f} "
                    f"official={seed.official_score:.10f} diff={seed.score_diff:.10f}"
                )
                failed = True

        if (
            result.round_score_diff is not None
            and abs(result.round_score_diff) > args.score_tolerance
        ):
            print(
                "ERROR: round score mismatch "
                f"local={result.round_score:.10f} official={result.official_round_score:.10f} "
                f"diff={result.round_score_diff:.10f}"
            )
            failed = True

    print_summary(result)
    write_summary_json(result, output_dir / "summary.json")

    if failed:
        return 1
    return 0


def print_summary(result: RoundScoreResult) -> None:
    print(f"Round ID: {result.round_id}")
    print("seed  weighted_kl       local_score      official_score   diff")
    for seed in result.per_seed:
        if seed.missing_submission:
            print(f"{seed.seed_index:<5}{'-':<17}{0.0:<16.8f}{'-':<16}{'-'}")
            continue

        official = "-" if seed.official_score is None else f"{seed.official_score:.8f}"
        diff = "-" if seed.score_diff is None else f"{seed.score_diff:+.8f}"
        print(
            f"{seed.seed_index:<5}{seed.weighted_kl:<17.10f}{seed.score:<16.8f}"
            f"{official:<16}{diff}"
        )

    print()
    print(f"Local round score: {result.round_score:.8f}")
    if result.official_round_score is not None:
        print(f"Official round score: {result.official_round_score:.8f}")
        print(f"Round diff: {result.round_score_diff:+.8f}")


def write_summary_json(result: RoundScoreResult, path: Path) -> None:
    output = {
        "round_id": result.round_id,
        "round_score": result.round_score,
        "official_round_score": result.official_round_score,
        "round_score_diff": result.round_score_diff,
        "per_seed": [
            {
                "seed_index": seed.seed_index,
                "weighted_kl": seed.weighted_kl,
                "score": seed.score,
                "official_score": seed.official_score,
                "score_diff": seed.score_diff,
                "source_path": seed.source_path,
                "missing_submission": seed.missing_submission,
            }
            for seed in result.per_seed
        ],
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
