from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from astar_island.offline_emulator import run_offline_round


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


def _make_round(logs_root: Path, round_id: str, seeds_count: int = 2, width: int = 6, height: int = 6) -> None:
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
            "round_number": 1,
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
                        "step": 0,
                        "grid": [[11 for _ in range(width)] for _ in range(height)],
                        "settlements": [],
                    },
                    {
                        "step": 50,
                        "grid": [[1 for _ in range(width)] for _ in range(height)],
                        "settlements": [
                            {
                                "x": 1,
                                "y": 1,
                                "population": 1.0,
                                "food": 0.5,
                                "wealth": 0.5,
                                "defense": 0.5,
                                "has_port": False,
                                "alive": True,
                                "owner_id": 0,
                            },
                            {
                                "x": width - 1,
                                "y": height - 1,
                                "population": 0.7,
                                "food": 0.4,
                                "wealth": 0.3,
                                "defense": 0.4,
                                "has_port": False,
                                "alive": True,
                                "owner_id": 1,
                            },
                        ],
                    },
                ],
            },
        )


class FixedPolicy:
    def __init__(self, queries: list[dict[str, int]]) -> None:
        self._queries = list(queries)

    def next_query(self, state) -> dict[str, int] | None:  # noqa: ANN001
        if not self._queries:
            return None
        return self._queries.pop(0)


class PerfectModel:
    def __init__(self, width: int, height: int) -> None:
        self._tensor = _constant_tensor(width=width, height=height, cls=0)

    def predict(self, round_state, seed_initial_state, seed_index):  # noqa: ANN001, ARG002
        return self._tensor


class TestOfflineEmulator(unittest.TestCase):
    def test_run_offline_round_scores_perfect_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp)
            _make_round(logs_root, "round-x", seeds_count=2, width=6, height=6)

            policy = FixedPolicy(
                [
                    {"seed_index": 0, "viewport_x": 0, "viewport_y": 0, "viewport_w": 5, "viewport_h": 5},
                    {"seed_index": 1, "viewport_x": 1, "viewport_y": 1, "viewport_w": 5, "viewport_h": 5},
                ]
            )
            model = PerfectModel(width=6, height=6)

            result = run_offline_round(
                policy=policy,
                model=model,
                round_id="round-x",
                logs_root=str(logs_root),
                query_budget=50,
                strict=True,
            )

            self.assertEqual(result.queries_used, 2)
            self.assertEqual(len(result.observations), 2)
            self.assertEqual(len(result.per_seed), 2)
            self.assertTrue(all(seed.score == 100.0 for seed in result.per_seed))
            self.assertEqual(result.round_score, 100.0)

    def test_viewport_clamps_to_edges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp)
            _make_round(logs_root, "round-y", seeds_count=1, width=6, height=6)

            policy = FixedPolicy(
                [
                    {"seed_index": 0, "viewport_x": 5, "viewport_y": 5, "viewport_w": 5, "viewport_h": 5},
                ]
            )
            model = PerfectModel(width=6, height=6)

            result = run_offline_round(
                policy=policy,
                model=model,
                round_id="round-y",
                logs_root=str(logs_root),
                query_budget=1,
                strict=True,
            )

            obs = result.observations[0]
            self.assertEqual(obs.viewport, {"x": 1, "y": 1, "w": 5, "h": 5})
            self.assertEqual(len(obs.grid), 5)
            self.assertEqual(len(obs.grid[0]), 5)

    def test_missing_replay_returns_unavailable_observation_when_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp)
            _make_round(logs_root, "round-z", seeds_count=2, width=6, height=6)
            (logs_root / "round-z" / "replay-seed-1.json").unlink()

            policy = FixedPolicy(
                [
                    {"seed_index": 1, "viewport_x": 0, "viewport_y": 0, "viewport_w": 5, "viewport_h": 5},
                ]
            )
            model = PerfectModel(width=6, height=6)

            result = run_offline_round(
                policy=policy,
                model=model,
                round_id="round-z",
                logs_root=str(logs_root),
                query_budget=1,
                strict=True,
                allow_missing_replays=True,
            )

            obs = result.observations[0]
            self.assertFalse(obs.available)
            self.assertEqual(obs.grid, [])
            self.assertEqual(obs.source, "missing-replay")


if __name__ == "__main__":
    unittest.main()
