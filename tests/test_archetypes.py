from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from astar_island.archetypes import (
    build_archetype_report,
    cluster_round_fingerprints,
    compute_round_fingerprints,
)
from astar_island.round_data import load_round_dataset


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _grid(width: int, height: int, default: int = 11) -> list[list[int]]:
    return [[default for _ in range(width)] for _ in range(height)]


def _make_round(
    logs_root: Path,
    *,
    round_id: str,
    round_number: int,
    owner_offset: int,
    growth_boost: int,
    seeds_count: int = 2,
    width: int = 6,
    height: int = 6,
) -> None:
    round_dir = logs_root / round_id
    round_dir.mkdir(parents=True, exist_ok=True)

    initial_states = []
    for _ in range(seeds_count):
        initial_states.append(
            {
                "grid": _grid(width, height, default=11),
                "settlements": [
                    {"x": 1, "y": 1, "has_port": False, "alive": True},
                    {"x": 4, "y": 4, "has_port": True, "alive": True},
                ],
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

    for seed in range(seeds_count):
        _write_json(
            round_dir / f"analysis-seed-{seed}.json",
            {
                "prediction": None,
                "ground_truth": [
                    [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(width)]
                    for _ in range(height)
                ],
                "score": 100.0,
                "width": width,
                "height": height,
                "initial_grid": _grid(width, height, default=11),
            },
        )

        initial_settlements = [
            {
                "x": 1,
                "y": 1,
                "owner_id": owner_offset + seed,
                "has_port": False,
                "alive": True,
                "population": 1.0,
                "food": 0.7,
                "wealth": 0.6,
                "defense": 0.5,
            },
            {
                "x": 4,
                "y": 4,
                "owner_id": owner_offset + seed + 10,
                "has_port": True,
                "alive": True,
                "population": 1.1,
                "food": 0.8,
                "wealth": 0.7,
                "defense": 0.6,
            },
        ]

        final_settlements = [
            {
                "x": 1,
                "y": 1,
                "owner_id": owner_offset + seed + growth_boost,
                "has_port": True,
                "alive": True,
                "population": 1.3,
                "food": 0.9,
                "wealth": 1.0,
                "defense": 0.9,
            },
            {
                "x": 4,
                "y": 4,
                "owner_id": owner_offset + seed + 10,
                "has_port": True,
                "alive": True,
                "population": 1.0,
                "food": 0.7,
                "wealth": 0.7,
                "defense": 0.8,
            },
        ]
        # Add extra settlements for growth differentiation.
        for idx in range(growth_boost):
            final_settlements.append(
                {
                    "x": (2 + idx) % width,
                    "y": (1 + idx) % height,
                    "owner_id": owner_offset + 30 + idx,
                    "has_port": idx % 2 == 0,
                    "alive": True,
                    "population": 0.8,
                    "food": 0.6,
                    "wealth": 0.5,
                    "defense": 0.5,
                }
            )

        final_grid = _grid(width, height, default=11)
        final_grid[0][0] = 3
        final_grid[0][1] = 4

        _write_json(
            round_dir / f"replay-seed-{seed}.json",
            {
                "round_id": round_id,
                "seed_index": seed,
                "frames": [
                    {
                        "step": 0,
                        "grid": _grid(width, height, default=11),
                        "settlements": initial_settlements,
                    },
                    {
                        "step": 50,
                        "grid": final_grid,
                        "settlements": final_settlements,
                    },
                ],
            },
        )


class TestArchetypes(unittest.TestCase):
    def test_compute_round_fingerprints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp)
            _make_round(
                logs_root,
                round_id="round-a",
                round_number=1,
                owner_offset=0,
                growth_boost=1,
            )
            _make_round(
                logs_root,
                round_id="round-b",
                round_number=2,
                owner_offset=100,
                growth_boost=3,
            )
            rounds = load_round_dataset(logs_root, include_replays=True, strict=True)
            fingerprints = compute_round_fingerprints(rounds)

            self.assertEqual(len(fingerprints), 2)
            self.assertTrue(all(fp.seeds_used == 2 for fp in fingerprints))
            self.assertTrue(
                all(0.0 <= fp.settlement_survival_rate <= 1.0 for fp in fingerprints)
            )
            self.assertTrue(all(fp.final_settlements >= fp.initial_settlements for fp in fingerprints))
            self.assertTrue(all(fp.ruin_frequency >= 0.0 for fp in fingerprints))

    def test_clustering_and_report_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            _make_round(
                logs_root,
                round_id="round-a",
                round_number=1,
                owner_offset=0,
                growth_boost=1,
            )
            _make_round(
                logs_root,
                round_id="round-b",
                round_number=2,
                owner_offset=100,
                growth_boost=3,
            )
            rounds = load_round_dataset(logs_root, include_replays=True, strict=True)
            fingerprints = compute_round_fingerprints(rounds)
            clustering = cluster_round_fingerprints(
                fingerprints,
                k=2,
                max_k=2,
                random_seed=11,
            )

            self.assertEqual(clustering.k, 2)
            self.assertEqual(len(clustering.assignments), 2)
            self.assertEqual(len(clustering.elbow), 1)

            report = build_archetype_report(rounds, k=2, max_k=2, random_seed=11)
            out_dir = Path(tmp) / "outputs"
            outputs = report.write(out_dir)
            self.assertTrue(Path(outputs["table_csv"]).exists())
            self.assertTrue(Path(outputs["scatter_svg"]).exists())
            self.assertTrue(Path(outputs["elbow_svg"]).exists())


if __name__ == "__main__":
    unittest.main()
