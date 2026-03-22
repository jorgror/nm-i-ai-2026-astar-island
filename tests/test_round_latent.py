from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from astar_island.models import SeedInitialState, Settlement
from astar_island.offline_emulator import (
    OfflineRoundState,
    ViewportObservation,
    run_offline_round,
)
from astar_island.round_latent import (
    RoundLatentConditionalModel,
    RoundLatentConfig,
    RoundLatentEncoder,
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


def _make_seed_state(width: int, height: int) -> SeedInitialState:
    return SeedInitialState(
        grid=[[11 for _ in range(width)] for _ in range(height)],
        settlements=[Settlement(x=1, y=1, has_port=False, alive=True)],
    )


def _make_state(
    *,
    observations: list[ViewportObservation],
    width: int = 6,
    height: int = 6,
    seeds: int = 2,
    queries_used: int | None = None,
) -> OfflineRoundState:
    initial_states = [_make_seed_state(width, height) for _ in range(seeds)]
    return OfflineRoundState(
        round_id="r-latent",
        map_width=width,
        map_height=height,
        seeds_count=seeds,
        initial_states=initial_states,
        replay_available=[True for _ in range(seeds)],
        queries_max=50,
        queries_used=queries_used if queries_used is not None else len(observations),
        observations=observations,
    )


def _ruin_observation(seed_index: int = 1) -> ViewportObservation:
    grid = [[3 for _ in range(5)] for _ in range(5)]
    settlements = [
        {
            "x": 2,
            "y": 2,
            "population": 0.1,
            "food": 0.1,
            "wealth": 0.1,
            "defense": 0.1,
            "has_port": False,
            "alive": False,
        }
    ]
    return ViewportObservation(
        seed_index=seed_index,
        query_index=1,
        grid=grid,
        settlements=settlements,
        viewport={"x": 0, "y": 0, "w": 5, "h": 5},
        width=6,
        height=6,
        queries_used=1,
        queries_max=50,
        available=True,
        source="unit-test",
    )


class FixedPolicy:
    def __init__(self, queries: list[dict[str, int]]) -> None:
        self._queries = list(queries)

    def next_query(self, state) -> dict[str, int] | None:  # noqa: ANN001
        if not self._queries:
            return None
        return self._queries.pop(0)


def _make_round(logs_root: Path, round_id: str, width: int = 6, height: int = 6) -> None:
    round_dir = logs_root / round_id
    round_dir.mkdir(parents=True, exist_ok=True)

    initial_states = []
    for _ in range(2):
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
            "seeds_count": 2,
            "initial_states": initial_states,
        },
    )

    for seed_idx in range(2):
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

        replay_grid = [[11 for _ in range(width)] for _ in range(height)]
        if seed_idx == 1:
            for y in range(3):
                for x in range(3):
                    replay_grid[y][x] = 3

        _write_json(
            round_dir / f"replay-seed-{seed_idx}.json",
            {
                "round_id": round_id,
                "seed_index": seed_idx,
                "frames": [
                    {"step": 50, "grid": replay_grid, "settlements": []},
                ],
            },
        )


class TestRoundLatent(unittest.TestCase):
    def test_encoder_picks_up_ruin_heavy_signal(self) -> None:
        state = _make_state(observations=[_ruin_observation(seed_index=1)])
        encoder = RoundLatentEncoder()

        latent = encoder.infer(state)

        self.assertGreater(latent.confidence, 0.0)
        self.assertGreater(latent.class_logit_offsets[3], latent.class_logit_offsets[1])
        self.assertGreater(latent.class_logit_offsets[3], latent.class_logit_offsets[2])

    def test_observation_from_one_seed_conditions_other_seed_prediction(self) -> None:
        model = RoundLatentConditionalModel(
            config=RoundLatentConfig(
                min_cell_strength=0.25,
                dynamic_strength_scale=1.0,
                static_cell_dampen=1.0,
            )
        )
        empty_state = _make_state(observations=[], queries_used=0)
        observed_state = _make_state(observations=[_ruin_observation(seed_index=1)], queries_used=1)
        seed0 = _make_seed_state(width=6, height=6)

        pred_no_obs = model.predict(empty_state, seed0, seed_index=0)
        pred_with_obs = model.predict(observed_state, seed0, seed_index=0)

        self.assertGreater(pred_with_obs[0][0][3], pred_no_obs[0][0][3])
        self.assertLess(pred_with_obs[0][0][1], pred_no_obs[0][0][1])
        self.assertTrue(math.isclose(sum(pred_with_obs[0][0]), 1.0, abs_tol=1e-9))

    def test_model_runs_inside_offline_emulator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp)
            _make_round(logs_root, "round-latent", width=6, height=6)

            policy = FixedPolicy(
                [
                    {"seed_index": 1, "viewport_x": 0, "viewport_y": 0, "viewport_w": 5, "viewport_h": 5}
                ]
            )
            model = RoundLatentConditionalModel()
            result = run_offline_round(
                policy=policy,
                model=model,
                round_id="round-latent",
                logs_root=str(logs_root),
                query_budget=1,
                strict=True,
            )

            self.assertEqual(result.queries_used, 1)
            self.assertEqual(len(result.per_seed), 2)
            self.assertTrue(all(0.0 <= seed.score <= 100.0 for seed in result.per_seed))


if __name__ == "__main__":
    unittest.main()
