from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from astar_island.baseline_b import BaselineBConfig
from astar_island.round_data import load_round_dataset
from astar_island.world_model import (
    BaselineBWorldModelPredictor,
    train_baseline_b_world_model_from_logs,
    train_baseline_b_world_model_from_rounds,
)


NUM_CLASSES = 6


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _one_hot(cls: int) -> list[float]:
    out = [0.0] * NUM_CLASSES
    out[cls] = 1.0
    return out


def _constant_tensor(width: int, height: int, cls: int) -> list[list[list[float]]]:
    row = [_one_hot(cls) for _ in range(width)]
    return [row[:] for _ in range(height)]


def _make_round(
    logs_root: Path,
    round_id: str,
    round_number: int,
    *,
    width: int = 6,
    height: int = 6,
) -> None:
    round_dir = logs_root / round_id
    round_dir.mkdir(parents=True, exist_ok=True)

    initial_states = [
        {
            "grid": [[11 for _ in range(width)] for _ in range(height)],
            "settlements": [{"x": 1, "y": 1, "has_port": False, "alive": True}],
        }
    ]
    _write_json(
        round_dir / "round-details.json",
        {
            "id": round_id,
            "round_number": round_number,
            "status": "completed",
            "map_width": width,
            "map_height": height,
            "seeds_count": 1,
            "initial_states": initial_states,
        },
    )

    target_class = round_number % NUM_CLASSES
    _write_json(
        round_dir / "analysis-seed-0.json",
        {
            "prediction": None,
            "ground_truth": _constant_tensor(width=width, height=height, cls=target_class),
            "score": 100.0,
            "width": width,
            "height": height,
            "initial_grid": [[11 for _ in range(width)] for _ in range(height)],
        },
    )
    _write_json(
        round_dir / "replay-seed-0.json",
        {
            "round_id": round_id,
            "seed_index": 0,
            "frames": [
                {
                    "step": 50,
                    "grid": [[11 for _ in range(width)] for _ in range(height)],
                    "settlements": [],
                }
            ],
        },
    )


class TestWorldModelAndStep14(unittest.TestCase):
    def test_train_world_model_from_rounds_and_predict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            _make_round(logs_root, "round-a", 1)
            _make_round(logs_root, "round-b", 2)
            _make_round(logs_root, "round-c", 3)

            rounds = load_round_dataset(logs_root, include_replays=False, strict=True)
            cfg = BaselineBConfig(epochs=1, samples_per_epoch=200, max_cells_per_seed=20)
            trained = train_baseline_b_world_model_from_rounds(
                rounds=rounds,
                config=cfg,
                exclude_round_id="round-c",
            )
            self.assertEqual(trained.rounds_used, 2)
            self.assertGreater(trained.samples_used, 0)

            predictor = BaselineBWorldModelPredictor(
                model=trained.model,
                probability_floor=trained.config.probability_floor,
            )
            seed_state = rounds[0].seeds[0].initial_state
            self.assertIsNotNone(seed_state)
            prediction = predictor(seed_state, 0)  # type: ignore[arg-type]

            self.assertEqual(len(prediction), 6)
            self.assertEqual(len(prediction[0]), 6)
            self.assertEqual(len(prediction[0][0]), NUM_CLASSES)
            self.assertTrue(math.isclose(sum(prediction[0][0]), 1.0, abs_tol=1e-9))

    def test_step14_script_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_root = root / "logs"
            output_dir = root / "outputs"
            _make_round(logs_root, "round-01", 1)
            _make_round(logs_root, "round-02", 2)

            script_path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_step14_world_model.py"
            cmd = [
                sys.executable,
                str(script_path),
                "--logs-root",
                str(logs_root),
                "--output-dir",
                str(output_dir),
                "--query-budget",
                "3",
                "--epochs",
                "1",
                "--samples-per-epoch",
                "200",
                "--max-cells-per-seed",
                "20",
                "--strict",
            ]
            completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if completed.returncode != 0:
                self.fail(
                    "step14 script failed\n"
                    f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
                )

            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            run_summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["rounds_evaluated"], 2)
            self.assertEqual(run_summary["rounds_evaluated"], 2)
            self.assertIn("dataset_fingerprint", summary)
            self.assertIn("dataset_fingerprint", run_summary)
            self.assertIn("mean_round_delta_world_vs_prior", summary)

    def test_train_world_model_from_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            _make_round(logs_root, "round-a", 1)
            cfg = BaselineBConfig(epochs=1, samples_per_epoch=200, max_cells_per_seed=20)
            trained = train_baseline_b_world_model_from_logs(
                logs_root=logs_root,
                config=cfg,
                strict=True,
            )
            self.assertEqual(trained.rounds_used, 1)
            self.assertGreater(trained.samples_used, 0)

    def test_holdout_train_to_test_script_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_root = root / "logs"
            output_dir = root / "outputs"
            _make_round(logs_root, "round-01", 1)
            _make_round(logs_root, "round-02", 2)

            script_path = (
                Path(__file__).resolve().parents[1]
                / "scripts"
                / "evaluate_holdout_train_to_test.py"
            )
            cmd = [
                sys.executable,
                str(script_path),
                "--logs-root",
                str(logs_root),
                "--output-dir",
                str(output_dir),
                "--train-max-round-number",
                "1",
                "--test-round-number",
                "2",
                "--query-budget",
                "3",
                "--strict",
            ]
            completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if completed.returncode != 0:
                self.fail(
                    "holdout script failed\n"
                    f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
                )

            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            run_summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["train_max_round_number"], 1)
            self.assertEqual(summary["test_round_number"], 2)
            self.assertEqual(run_summary["train_max_round_number"], 1)
            self.assertEqual(run_summary["test_round_number"], 2)
            self.assertIn("scenarios", summary)
            self.assertEqual(len(summary["scenarios"]), 3)
            self.assertIn("deltas", summary)


if __name__ == "__main__":
    unittest.main()
