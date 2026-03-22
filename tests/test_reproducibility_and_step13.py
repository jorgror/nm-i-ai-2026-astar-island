from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from astar_island.reproducibility import build_round_dataset_fingerprint
from astar_island.round_data import load_round_dataset


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


def _make_round(logs_root: Path, round_id: str, round_number: int, *, seeds_count: int = 1) -> None:
    width = 6
    height = 6
    round_dir = logs_root / round_id
    round_dir.mkdir(parents=True, exist_ok=True)

    initial_states = []
    for _ in range(seeds_count):
        initial_states.append(
            {
                "grid": [[11 for _ in range(width)] for _ in range(height)],
                "settlements": [{"x": 1, "y": 1, "has_port": False, "alive": True}],
            }
        )

    _write_json(
        round_dir / "round-details.json",
        {
            "id": round_id,
            "round_number": round_number,
            "status": "completed",
            "map_width": width,
            "map_height": height,
            "seeds_count": seeds_count,
            "initial_states": initial_states,
        },
    )

    for seed_idx in range(seeds_count):
        _write_json(
            round_dir / f"analysis-seed-{seed_idx}.json",
            {
                "prediction": None,
                "ground_truth": _constant_tensor(width=width, height=height, cls=0),
                "score": 100.0,
                "width": width,
                "height": height,
                "initial_grid": [[11 for _ in range(width)] for _ in range(height)],
            },
        )
        _write_json(
            round_dir / f"replay-seed-{seed_idx}.json",
            {
                "round_id": round_id,
                "seed_index": seed_idx,
                "frames": [
                    {
                        "step": 50,
                        "grid": [[11 for _ in range(width)] for _ in range(height)],
                        "settlements": [],
                    }
                ],
            },
        )


class TestReproducibilityAndStep13(unittest.TestCase):
    def test_round_dataset_fingerprint_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            _make_round(logs_root, "round-b", round_number=2)
            _make_round(logs_root, "round-a", round_number=1)

            rounds = load_round_dataset(logs_root, include_replays=True, strict=True)
            fp_a = build_round_dataset_fingerprint(rounds=rounds, logs_root=logs_root)
            fp_b = build_round_dataset_fingerprint(rounds=list(reversed(rounds)), logs_root=logs_root)

            self.assertEqual(fp_a["sha256"], fp_b["sha256"])
            self.assertEqual(fp_a["round_count"], 2)
            self.assertEqual(fp_a["round_ids"], ["round-a", "round-b"])
            self.assertEqual(fp_a["round_numbers"], [1, 2])

    def test_step13_script_writes_dataset_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_root = root / "logs"
            output_dir = root / "outputs"
            _make_round(logs_root, "round-01", round_number=1)
            _make_round(logs_root, "round-02", round_number=2)

            script_path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_step13_ablations.py"
            cmd = [
                sys.executable,
                str(script_path),
                "--logs-root",
                str(logs_root),
                "--output-dir",
                str(output_dir),
                "--query-budget",
                "3",
                "--strict",
                "--baseline-b-seed-csv",
                str(root / "missing-b.csv"),
                "--baseline-c-seed-csv",
                str(root / "missing-c.csv"),
            ]
            completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if completed.returncode != 0:
                self.fail(
                    "step13 script failed\n"
                    f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
                )

            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            run_summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))

            self.assertEqual(summary["rounds_evaluated"], 2)
            self.assertEqual(run_summary["rounds_evaluated"], 2)
            self.assertEqual(run_summary["scenario_count"], 5)
            self.assertIn("dataset_fingerprint", summary)
            self.assertIn("dataset_fingerprint", run_summary)
            self.assertEqual(
                summary["dataset_fingerprint"]["sha256"],
                run_summary["dataset_fingerprint"]["sha256"],
            )
            self.assertEqual(summary["dataset_fingerprint"]["round_count"], 2)
            self.assertEqual(
                summary["dataset_fingerprint"]["round_ids"],
                ["round-01", "round-02"],
            )


if __name__ == "__main__":
    unittest.main()
