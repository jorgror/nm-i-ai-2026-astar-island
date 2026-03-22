from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from astar_island.baseline_b import (
    FEATURE_NAMES,
    BaselineBConfig,
    build_seed_feature_grid,
    evaluate_baseline_b_leave_one_round_out,
    predict_with_model,
    train_multinomial_logistic_regression,
)
from astar_island.models import SeedInitialState, Settlement
from astar_island.round_data import load_round_dataset


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _one_hot(cls: int) -> list[float]:
    out = [0.0] * 6
    out[cls] = 1.0
    return out


def _grid_to_tensor(grid: list[list[int]]) -> list[list[list[float]]]:
    mapping = {0: 0, 10: 0, 11: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
    return [
        [_one_hot(mapping.get(int(value), 0)) for value in row]
        for row in grid
    ]


def _make_round(logs_root: Path, *, round_id: str, round_number: int, seed_grids: list[list[list[int]]]) -> None:
    round_dir = logs_root / round_id
    round_dir.mkdir(parents=True, exist_ok=True)

    initial_states = []
    for grid in seed_grids:
        initial_states.append(
            {
                "grid": grid,
                "settlements": [{"x": 1, "y": 1, "has_port": False, "alive": True}],
            }
        )

    _write_json(
        round_dir / "round-details.json",
        {
            "id": round_id,
            "round_number": round_number,
            "status": "completed",
            "map_width": len(seed_grids[0][0]),
            "map_height": len(seed_grids[0]),
            "seeds_count": len(seed_grids),
            "initial_states": initial_states,
        },
    )

    for seed_idx, grid in enumerate(seed_grids):
        _write_json(
            round_dir / f"analysis-seed-{seed_idx}.json",
            {
                "prediction": None,
                "ground_truth": _grid_to_tensor(grid),
                "score": 100.0,
                "width": len(grid[0]),
                "height": len(grid),
                "initial_grid": grid,
            },
        )


class TestBaselineB(unittest.TestCase):
    def test_build_seed_feature_grid_shape(self) -> None:
        state = SeedInitialState(
            grid=[
                [10, 10, 11, 11],
                [11, 1, 11, 4],
                [11, 2, 5, 11],
                [0, 11, 11, 3],
            ],
            settlements=[Settlement(x=1, y=1, has_port=False, alive=True)],
        )
        features = build_seed_feature_grid(state)
        self.assertEqual(len(features), 4)
        self.assertEqual(len(features[0]), 4)
        self.assertEqual(len(features[0][0]), len(FEATURE_NAMES))

    def test_train_and_predict_probabilities(self) -> None:
        state = SeedInitialState(
            grid=[
                [11, 11, 4, 5],
                [10, 1, 2, 3],
                [11, 11, 4, 5],
                [10, 1, 2, 3],
            ],
            settlements=[Settlement(x=1, y=1, has_port=False, alive=True)],
        )
        features_grid = build_seed_feature_grid(state)
        tensor = _grid_to_tensor(state.grid)

        features: list[list[float]] = []
        targets: list[list[float]] = []
        for y in range(4):
            for x in range(4):
                features.append(features_grid[y][x])
                targets.append(tensor[y][x])

        model = train_multinomial_logistic_regression(
            features=features,
            targets=targets,
            config=BaselineBConfig(
                learning_rate=0.08,
                epochs=8,
                l2=1e-4,
                samples_per_epoch=64,
                max_cells_per_seed=16,
                probability_floor=1e-4,
                random_seed=3,
            ),
            rng=__import__("random").Random(3),
        )
        pred = predict_with_model(model, features_grid=features_grid, probability_floor=1e-4)
        for row in pred:
            for cell in row:
                self.assertEqual(len(cell), 6)
                self.assertTrue(all(value > 0.0 for value in cell))
                self.assertTrue(math.isclose(sum(cell), 1.0, abs_tol=1e-9))

    def test_leave_one_round_out_evaluation_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp)
            _make_round(
                logs_root,
                round_id="round-a",
                round_number=1,
                seed_grids=[
                    [
                        [11, 11, 4, 5],
                        [10, 1, 2, 3],
                        [11, 11, 4, 5],
                        [10, 1, 2, 3],
                    ],
                    [
                        [11, 11, 11, 11],
                        [11, 4, 4, 11],
                        [10, 1, 2, 10],
                        [11, 5, 3, 11],
                    ],
                ],
            )
            _make_round(
                logs_root,
                round_id="round-b",
                round_number=2,
                seed_grids=[
                    [
                        [11, 11, 11, 11],
                        [10, 2, 3, 10],
                        [11, 4, 5, 11],
                        [11, 1, 11, 11],
                    ],
                    [
                        [10, 10, 11, 11],
                        [11, 1, 11, 4],
                        [11, 2, 5, 11],
                        [0, 11, 11, 3],
                    ],
                ],
            )

            # Sanity check loader expectations used by the evaluator.
            rounds = load_round_dataset(logs_root, include_replays=False, strict=True)
            self.assertEqual(len(rounds), 2)

            report = evaluate_baseline_b_leave_one_round_out(
                logs_root=logs_root,
                config=BaselineBConfig(
                    learning_rate=0.06,
                    epochs=4,
                    l2=1e-4,
                    samples_per_epoch=128,
                    max_cells_per_seed=16,
                    probability_floor=1e-4,
                    random_seed=7,
                ),
                strict=True,
            )
            self.assertEqual(len(report.round_results), 2)
            self.assertEqual(len(report.seed_results), 4)
            self.assertTrue(all(0.0 <= row.score_baseline_b <= 100.0 for row in report.seed_results))
            self.assertTrue(all(0.0 <= row.score_prior_a <= 100.0 for row in report.seed_results))


if __name__ == "__main__":
    unittest.main()
