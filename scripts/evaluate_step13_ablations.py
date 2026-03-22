#!/usr/bin/env python3
"""Step 13 ablation summary across query policy, latent, blending, and baseline model families."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

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
from astar_island.reproducibility import build_round_dataset_fingerprint
from astar_island.round_data import RoundRecord, load_round_dataset
from astar_island.round_latent import RoundLatentConditionalModel, RoundLatentConfig
from astar_island.scoring import cell_entropy


@dataclass(slots=True)
class RoundScenarioRow:
    round_id: str
    round_number: int | None
    scenario: str
    queries_used: int
    round_score: float
    mean_seed_score: float
    mean_seed_weighted_kl: float
    dynamic_cells: int
    dynamic_ece: float
    dynamic_mean_confidence: float
    dynamic_mean_expected_accuracy: float
    value_per_query_vs_no_query: float


@dataclass(slots=True)
class ScenarioSummaryRow:
    scenario: str
    query_budget: int
    rounds_evaluated: int
    mean_round_score: float
    mean_seed_score: float
    mean_seed_weighted_kl: float
    mean_queries_used: float
    mean_value_per_query_vs_no_query: float
    mean_dynamic_ece: float
    mean_dynamic_confidence: float
    mean_dynamic_expected_accuracy: float


@dataclass(slots=True)
class ScenarioConfig:
    scenario: str
    query_budget: int
    policy_factory: Callable[[int], object]
    model_factory: Callable[[], object]


@dataclass(slots=True)
class CalibrationSummary:
    dynamic_cells: int
    ece: float
    mean_confidence: float
    mean_expected_accuracy: float


class CalibrationAccumulator:
    def __init__(self, *, bins: int) -> None:
        self._bins = max(1, int(bins))
        self._count = [0 for _ in range(self._bins)]
        self._conf_sum = [0.0 for _ in range(self._bins)]
        self._acc_sum = [0.0 for _ in range(self._bins)]
        self._total = 0

    def add(self, confidence: float, expected_accuracy: float) -> None:
        conf = min(max(float(confidence), 0.0), 1.0)
        acc = min(max(float(expected_accuracy), 0.0), 1.0)
        idx = min(self._bins - 1, int(conf * self._bins))
        self._count[idx] += 1
        self._conf_sum[idx] += conf
        self._acc_sum[idx] += acc
        self._total += 1

    def summary(self) -> CalibrationSummary:
        if self._total <= 0:
            return CalibrationSummary(
                dynamic_cells=0,
                ece=0.0,
                mean_confidence=0.0,
                mean_expected_accuracy=0.0,
            )

        ece = 0.0
        conf_total = 0.0
        acc_total = 0.0
        for idx in range(self._bins):
            count = self._count[idx]
            if count <= 0:
                continue
            avg_conf = self._conf_sum[idx] / float(count)
            avg_acc = self._acc_sum[idx] / float(count)
            weight = count / float(self._total)
            ece += weight * abs(avg_acc - avg_conf)
            conf_total += self._conf_sum[idx]
            acc_total += self._acc_sum[idx]

        return CalibrationSummary(
            dynamic_cells=self._total,
            ece=ece,
            mean_confidence=conf_total / float(self._total),
            mean_expected_accuracy=acc_total / float(self._total),
        )


class NoQueryPolicy:
    def next_query(self, state: OfflineRoundState) -> None:  # noqa: ARG002
        return None


class CenterSweepPolicy:
    def __init__(self, *, query_budget: int, default_window: int = 15) -> None:
        self.query_budget = int(query_budget)
        self.default_window = int(default_window)

    def next_query(self, state: OfflineRoundState) -> ViewportQuery | None:
        idx = int(state.queries_used)
        if idx >= self.query_budget or state.seeds_count <= 0:
            return None
        window = min(
            max(5, self.default_window),
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
    parser.add_argument("--logs-root", default=str(REPO_ROOT / "logs"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "step13_ablations"))
    parser.add_argument("--query-budget", type=int, default=50)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--dynamic-entropy-min", type=float, default=1e-6)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument(
        "--baseline-b-seed-csv",
        default=str(REPO_ROOT / "outputs" / "step8_production_b" / "loo_seed_results.csv"),
    )
    parser.add_argument(
        "--baseline-c-seed-csv",
        default=str(REPO_ROOT / "outputs" / "step8_production_c" / "loo_seed_results.csv"),
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
    if not rounds_sorted:
        raise ValueError("No rounds loaded for Step 13 evaluation.")
    dataset_fingerprint = build_round_dataset_fingerprint(
        rounds=rounds_sorted,
        logs_root=args.logs_root,
    )

    scenarios = [
        ScenarioConfig(
            scenario="no_query_prior",
            query_budget=0,
            policy_factory=lambda _budget: NoQueryPolicy(),
            model_factory=lambda: PriorOnlyModel(),
        ),
        ScenarioConfig(
            scenario="fixed_query_latent_blend",
            query_budget=args.query_budget,
            policy_factory=lambda budget: CenterSweepPolicy(query_budget=budget),
            model_factory=lambda: RoundLatentConditionalModel(
                config=RoundLatentConfig(enable_observation_blend=True)
            ),
        ),
        ScenarioConfig(
            scenario="adaptive_query_latent_blend",
            query_budget=args.query_budget,
            policy_factory=lambda budget: DeterministicThreePhaseQueryPolicy(
                config=DeterministicThreePhasePolicyConfig(query_budget=budget)
            ),
            model_factory=lambda: RoundLatentConditionalModel(
                config=RoundLatentConfig(enable_observation_blend=True)
            ),
        ),
        ScenarioConfig(
            scenario="adaptive_query_prior",
            query_budget=args.query_budget,
            policy_factory=lambda budget: DeterministicThreePhaseQueryPolicy(
                config=DeterministicThreePhasePolicyConfig(query_budget=budget)
            ),
            model_factory=lambda: PriorOnlyModel(),
        ),
        ScenarioConfig(
            scenario="adaptive_query_latent_no_blend",
            query_budget=args.query_budget,
            policy_factory=lambda budget: DeterministicThreePhaseQueryPolicy(
                config=DeterministicThreePhasePolicyConfig(query_budget=budget)
            ),
            model_factory=lambda: RoundLatentConditionalModel(
                config=RoundLatentConfig(enable_observation_blend=False)
            ),
        ),
    ]

    round_rows: list[RoundScenarioRow] = []
    for scenario in scenarios:
        print(f"Evaluating scenario={scenario.scenario} ...")
        scenario_rows = _evaluate_scenario(
            rounds=rounds_sorted,
            scenario=scenario,
            logs_root=args.logs_root,
            strict=args.strict,
            dynamic_entropy_min=args.dynamic_entropy_min,
            ece_bins=args.ece_bins,
        )
        round_rows.extend(scenario_rows)

    round_rows = _attach_value_per_query(round_rows)
    scenario_summary_rows = _summarize_round_rows(round_rows)

    comparisons = _build_comparisons(scenario_summary_rows)
    feature_vs_cnn = _load_feature_vs_cnn(
        baseline_b_seed_csv=Path(args.baseline_b_seed_csv),
        baseline_c_seed_csv=Path(args.baseline_c_seed_csv),
    )

    round_csv = output_dir / "round_scenario_results.csv"
    summary_csv = output_dir / "scenario_summary.csv"
    summary_json = output_dir / "summary.json"
    markdown_path = output_dir / "step13_summary.md"
    run_summary_json = output_dir / "run_summary.json"

    _write_csv(round_csv, round_rows)
    _write_csv(summary_csv, scenario_summary_rows)

    summary = {
        "query_budget": int(args.query_budget),
        "rounds_evaluated": len(rounds_sorted),
        "dataset_fingerprint": dataset_fingerprint,
        "scenarios": [asdict(row) for row in scenario_summary_rows],
        "comparisons": comparisons,
        "feature_vs_cnn": feature_vs_cnn,
        "notes": {
            "dynamic_calibration_metric": (
                "Dynamic-cell ECE computed from confidence bins where expected accuracy "
                "is ground_truth[predicted_argmax]."
            )
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    markdown_path.write_text(
        _render_markdown_summary(
            rounds_evaluated=len(rounds_sorted),
            query_budget=int(args.query_budget),
            dataset_fingerprint=dataset_fingerprint,
            scenario_rows=scenario_summary_rows,
            comparisons=comparisons,
            feature_vs_cnn=feature_vs_cnn,
        ),
        encoding="utf-8",
    )

    run_summary = {
        "rounds_evaluated": len(rounds_sorted),
        "scenario_count": len(scenario_summary_rows),
        "query_budget": int(args.query_budget),
        "dataset_fingerprint": dataset_fingerprint,
        "mean_round_score_no_query_prior": _scenario_metric(
            scenario_summary_rows,
            "no_query_prior",
            "mean_round_score",
        ),
        "mean_round_score_adaptive_query_latent_blend": _scenario_metric(
            scenario_summary_rows,
            "adaptive_query_latent_blend",
            "mean_round_score",
        ),
        "mean_round_delta_adaptive_vs_fixed": comparisons.get("adaptive_vs_fixed_mean_round_delta"),
        "mean_round_delta_latent_vs_prior": comparisons.get("latent_vs_prior_mean_round_delta"),
        "mean_round_delta_blend_vs_no_blend": comparisons.get("blend_vs_no_blend_mean_round_delta"),
        "feature_vs_cnn_delta": feature_vs_cnn.get("mean_round_delta_c_minus_b"),
        "output_files": {
            "round_csv": str(round_csv),
            "summary_csv": str(summary_csv),
            "summary_json": str(summary_json),
            "markdown": str(markdown_path),
        },
    }
    run_summary_json.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    print("Step 13 ablation evaluation complete.")
    print(f"rounds_evaluated={len(rounds_sorted)}")
    for row in scenario_summary_rows:
        print(
            f"scenario={row.scenario} "
            f"mean_round_score={row.mean_round_score:.6f} "
            f"mean_wkl={row.mean_seed_weighted_kl:.6f} "
            f"mean_dynamic_ece={row.mean_dynamic_ece:.6f} "
            f"mean_value_per_query={row.mean_value_per_query_vs_no_query:.6f}"
        )
    print("output_files:")
    print(f"  round_csv: {round_csv}")
    print(f"  summary_csv: {summary_csv}")
    print(f"  summary_json: {summary_json}")
    print(f"  markdown: {markdown_path}")
    print(f"  run_summary_json: {run_summary_json}")
    return 0


def _evaluate_scenario(
    *,
    rounds: list[RoundRecord],
    scenario: ScenarioConfig,
    logs_root: str,
    strict: bool,
    dynamic_entropy_min: float,
    ece_bins: int,
) -> list[RoundScenarioRow]:
    rows: list[RoundScenarioRow] = []
    for record in rounds:
        model = scenario.model_factory()
        result = run_offline_round(
            policy=scenario.policy_factory(scenario.query_budget),
            model=model,
            round_id=record.round_id,
            round_record=record,
            logs_root=logs_root,
            query_budget=scenario.query_budget,
            include_replays=True,
            strict=strict,
            allow_missing_replays=False,
        )

        state = OfflineRoundState(
            round_id=record.round_id,
            map_width=record.map_width,
            map_height=record.map_height,
            seeds_count=record.seeds_count,
            initial_states=[seed.initial_state for seed in record.seeds],
            replay_available=[seed.replay is not None for seed in record.seeds],
            queries_max=scenario.query_budget,
            queries_used=result.queries_used,
            observations=result.observations,
        )

        calibration = CalibrationAccumulator(bins=ece_bins)
        for seed in record.seeds:
            if seed.analysis is None:
                continue
            prediction = model.predict(
                round_state=state,
                seed_initial_state=seed.initial_state,
                seed_index=seed.seed_index,
            )
            _accumulate_dynamic_calibration(
                calibration=calibration,
                ground_truth=seed.analysis.ground_truth,
                prediction=prediction,
                dynamic_entropy_min=dynamic_entropy_min,
            )

        cal = calibration.summary()
        per_seed_scores = [seed.score for seed in result.per_seed]
        per_seed_wkl = [seed.weighted_kl for seed in result.per_seed]
        rows.append(
            RoundScenarioRow(
                round_id=record.round_id,
                round_number=record.round_number,
                scenario=scenario.scenario,
                queries_used=result.queries_used,
                round_score=result.round_score,
                mean_seed_score=_mean(per_seed_scores),
                mean_seed_weighted_kl=_mean(per_seed_wkl),
                dynamic_cells=cal.dynamic_cells,
                dynamic_ece=cal.ece,
                dynamic_mean_confidence=cal.mean_confidence,
                dynamic_mean_expected_accuracy=cal.mean_expected_accuracy,
                value_per_query_vs_no_query=0.0,
            )
        )
    return rows


def _accumulate_dynamic_calibration(
    *,
    calibration: CalibrationAccumulator,
    ground_truth: list[list[list[float]]],
    prediction: list[list[list[float]]],
    dynamic_entropy_min: float,
) -> None:
    for gt_row, pred_row in zip(ground_truth, prediction):
        for gt_cell, pred_cell in zip(gt_row, pred_row):
            if cell_entropy(gt_cell) <= dynamic_entropy_min:
                continue
            pred_class = max(range(len(pred_cell)), key=lambda idx: pred_cell[idx])
            confidence = float(pred_cell[pred_class])
            expected_accuracy = float(gt_cell[pred_class])
            calibration.add(confidence, expected_accuracy)


def _attach_value_per_query(rows: list[RoundScenarioRow]) -> list[RoundScenarioRow]:
    no_query_by_round: dict[str, float] = {}
    for row in rows:
        if row.scenario == "no_query_prior":
            no_query_by_round[row.round_id] = row.round_score

    out: list[RoundScenarioRow] = []
    for row in rows:
        baseline = no_query_by_round.get(row.round_id, row.round_score)
        if row.queries_used > 0:
            value_per_query = (row.round_score - baseline) / float(row.queries_used)
        else:
            value_per_query = 0.0
        out.append(
            RoundScenarioRow(
                round_id=row.round_id,
                round_number=row.round_number,
                scenario=row.scenario,
                queries_used=row.queries_used,
                round_score=row.round_score,
                mean_seed_score=row.mean_seed_score,
                mean_seed_weighted_kl=row.mean_seed_weighted_kl,
                dynamic_cells=row.dynamic_cells,
                dynamic_ece=row.dynamic_ece,
                dynamic_mean_confidence=row.dynamic_mean_confidence,
                dynamic_mean_expected_accuracy=row.dynamic_mean_expected_accuracy,
                value_per_query_vs_no_query=value_per_query,
            )
        )
    return out


def _summarize_round_rows(rows: list[RoundScenarioRow]) -> list[ScenarioSummaryRow]:
    by_scenario: dict[str, list[RoundScenarioRow]] = {}
    for row in rows:
        by_scenario.setdefault(row.scenario, []).append(row)

    out: list[ScenarioSummaryRow] = []
    for scenario, scenario_rows in sorted(by_scenario.items()):
        out.append(
            ScenarioSummaryRow(
                scenario=scenario,
                query_budget=max([row.queries_used for row in scenario_rows], default=0),
                rounds_evaluated=len(scenario_rows),
                mean_round_score=_mean([row.round_score for row in scenario_rows]),
                mean_seed_score=_mean([row.mean_seed_score for row in scenario_rows]),
                mean_seed_weighted_kl=_mean([row.mean_seed_weighted_kl for row in scenario_rows]),
                mean_queries_used=_mean([row.queries_used for row in scenario_rows]),
                mean_value_per_query_vs_no_query=_mean(
                    [row.value_per_query_vs_no_query for row in scenario_rows]
                ),
                mean_dynamic_ece=_mean([row.dynamic_ece for row in scenario_rows]),
                mean_dynamic_confidence=_mean(
                    [row.dynamic_mean_confidence for row in scenario_rows]
                ),
                mean_dynamic_expected_accuracy=_mean(
                    [row.dynamic_mean_expected_accuracy for row in scenario_rows]
                ),
            )
        )
    return out


def _build_comparisons(scenarios: list[ScenarioSummaryRow]) -> dict[str, float | None]:
    return {
        "adaptive_vs_fixed_mean_round_delta": _scenario_delta(
            scenarios,
            winner="adaptive_query_latent_blend",
            baseline="fixed_query_latent_blend",
            metric="mean_round_score",
        ),
        "latent_vs_prior_mean_round_delta": _scenario_delta(
            scenarios,
            winner="adaptive_query_latent_no_blend",
            baseline="adaptive_query_prior",
            metric="mean_round_score",
        ),
        "blend_vs_no_blend_mean_round_delta": _scenario_delta(
            scenarios,
            winner="adaptive_query_latent_blend",
            baseline="adaptive_query_latent_no_blend",
            metric="mean_round_score",
        ),
    }


def _load_feature_vs_cnn(
    *,
    baseline_b_seed_csv: Path,
    baseline_c_seed_csv: Path,
) -> dict[str, float | None]:
    b_scores, b_wkl = _read_seed_score_wkl(
        path=baseline_b_seed_csv,
        score_key="score_baseline_b",
        wkl_key="weighted_kl_baseline_b",
    )
    c_scores, c_wkl = _read_seed_score_wkl(
        path=baseline_c_seed_csv,
        score_key="score_baseline_c",
        wkl_key="weighted_kl_baseline_c",
    )
    if not b_scores or not c_scores:
        return {
            "mean_round_score_baseline_b": None,
            "mean_round_score_baseline_c": None,
            "mean_round_delta_c_minus_b": None,
            "mean_seed_weighted_kl_baseline_b": None,
            "mean_seed_weighted_kl_baseline_c": None,
            "mean_seed_weighted_kl_delta_c_minus_b": None,
        }
    return {
        "mean_round_score_baseline_b": _mean(b_scores),
        "mean_round_score_baseline_c": _mean(c_scores),
        "mean_round_delta_c_minus_b": _mean(c_scores) - _mean(b_scores),
        "mean_seed_weighted_kl_baseline_b": _mean(b_wkl),
        "mean_seed_weighted_kl_baseline_c": _mean(c_wkl),
        "mean_seed_weighted_kl_delta_c_minus_b": _mean(c_wkl) - _mean(b_wkl),
    }


def _read_seed_score_wkl(
    *,
    path: Path,
    score_key: str,
    wkl_key: str,
) -> tuple[list[float], list[float]]:
    if not path.exists():
        return [], []
    scores: list[float] = []
    wkls: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                scores.append(float(row[score_key]))
                wkls.append(float(row[wkl_key]))
            except (KeyError, TypeError, ValueError):
                continue
    return scores, wkls


def _render_markdown_summary(
    *,
    rounds_evaluated: int,
    query_budget: int,
    dataset_fingerprint: dict[str, object],
    scenario_rows: list[ScenarioSummaryRow],
    comparisons: dict[str, float | None],
    feature_vs_cnn: dict[str, float | None],
) -> str:
    lines = [
        "# Step 13 Ablation Summary",
        "",
        f"- Rounds evaluated: {rounds_evaluated}",
        f"- Query budget (query-based scenarios): {query_budget}",
        f"- Dataset fingerprint (sha256): `{dataset_fingerprint.get('sha256')}`",
        "",
        "## Scenario Metrics",
        "",
        "| Scenario | Mean Round Score | Mean Seed WKL | Mean Dynamic ECE | Value / Query |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in scenario_rows:
        lines.append(
            f"| {row.scenario} | {row.mean_round_score:.6f} | {row.mean_seed_weighted_kl:.6f} "
            f"| {row.mean_dynamic_ece:.6f} | {row.mean_value_per_query_vs_no_query:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Key Deltas",
            "",
            f"- Adaptive vs fixed policy (latent+blend): {_fmt_signed(comparisons.get('adaptive_vs_fixed_mean_round_delta'))}",
            f"- Latent vs prior (adaptive, no blend): {_fmt_signed(comparisons.get('latent_vs_prior_mean_round_delta'))}",
            f"- Blend vs no blend (adaptive latent): {_fmt_signed(comparisons.get('blend_vs_no_blend_mean_round_delta'))}",
            "",
            "## Feature vs CNN (LOO)",
            "",
            f"- Baseline B mean round score: {_fmt(feature_vs_cnn.get('mean_round_score_baseline_b'))}",
            f"- Baseline C mean round score: {_fmt(feature_vs_cnn.get('mean_round_score_baseline_c'))}",
            f"- C - B mean round delta: {_fmt_signed(feature_vs_cnn.get('mean_round_delta_c_minus_b'))}",
            "",
            "## Notes",
            "",
            "- Dynamic-cell ECE uses confidence binning on cells where entropy(ground truth) > threshold.",
            "- Expected accuracy per cell is ground_truth probability assigned to predicted argmax class.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[object]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _scenario_metric(
    scenarios: list[ScenarioSummaryRow],
    scenario: str,
    metric: str,
) -> float | None:
    for row in scenarios:
        if row.scenario == scenario:
            return float(getattr(row, metric))
    return None


def _scenario_delta(
    scenarios: list[ScenarioSummaryRow],
    *,
    winner: str,
    baseline: str,
    metric: str,
) -> float | None:
    a = _scenario_metric(scenarios, winner, metric)
    b = _scenario_metric(scenarios, baseline, metric)
    if a is None or b is None:
        return None
    return a - b


def _mean(values: list[float | int]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def _fmt(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:.6f}"


def _fmt_signed(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:+.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
