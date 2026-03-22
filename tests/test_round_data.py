from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from astar_island.round_data import leave_one_round_out_splits, load_round_dataset


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_round(logs_root: Path, round_id: str, round_number: int, seeds_count: int = 2) -> None:
    round_dir = logs_root / round_id
    round_dir.mkdir(parents=True, exist_ok=True)

    initial_states = []
    for _ in range(seeds_count):
        initial_states.append(
            {
                "grid": [[11]],
                "settlements": [],
            }
        )

    _write_json(
        round_dir / "round-details.json",
        {
            "id": round_id,
            "round_number": round_number,
            "status": "completed",
            "map_width": 1,
            "map_height": 1,
            "seeds_count": seeds_count,
            "initial_states": initial_states,
        },
    )

    for seed_index in range(seeds_count):
        _write_json(
            round_dir / f"analysis-seed-{seed_index}.json",
            {
                "prediction": None,
                "ground_truth": [[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
                "score": 100.0,
                "width": 1,
                "height": 1,
                "initial_grid": [[11]],
            },
        )


class TestRoundData(unittest.TestCase):
    def test_load_round_dataset_has_round_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp)
            _make_round(logs_root, "round-a", round_number=2)
            _make_round(logs_root, "round-b", round_number=1)
            _write_json(
                logs_root / "round-a" / "replay-seed-0.json",
                {"round_id": "round-a", "seed_index": 0, "frames": []},
            )

            rounds = load_round_dataset(logs_root, include_replays=True, strict=True)

            self.assertEqual(len(rounds), 2)
            self.assertEqual(rounds[0].round_id, "round-a")
            self.assertEqual(rounds[0].seeds_count, 2)
            self.assertIsNotNone(rounds[0].seeds[0].analysis)
            self.assertIsNotNone(rounds[0].seeds[0].replay)
            self.assertIsNone(rounds[0].seeds[1].replay)

    def test_load_round_dataset_strict_requires_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp)
            _make_round(logs_root, "round-a", round_number=1)
            (logs_root / "round-a" / "analysis-seed-1.json").unlink()

            with self.assertRaises(FileNotFoundError):
                load_round_dataset(logs_root, strict=True)

            rounds = load_round_dataset(logs_root, strict=False)
            self.assertEqual(len(rounds), 1)
            self.assertIsNone(rounds[0].seeds[1].analysis)

    def test_leave_one_round_out_is_round_based(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp)
            _make_round(logs_root, "round-a", round_number=3)
            _make_round(logs_root, "round-b", round_number=2)
            _make_round(logs_root, "round-c", round_number=1)

            rounds = load_round_dataset(logs_root, strict=True)
            splits = leave_one_round_out_splits(rounds)

            self.assertEqual(len(splits), 3)
            for split in splits:
                train_ids = {record.round_id for record in split.training_rounds}
                self.assertNotIn(split.holdout_round_id, train_ids)
                self.assertEqual(split.validation_round.round_id, split.holdout_round_id)
                self.assertEqual(len(split.validation_round.seeds), 2)


if __name__ == "__main__":
    unittest.main()
