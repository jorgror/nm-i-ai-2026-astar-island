from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from astar_island.importance import (
    dynamic_importance_map_from_initial_state,
    dynamic_importance_maps_for_round,
)
from astar_island.models import RoundDetail, SeedInitialState, Settlement
from astar_island.priors import baseline_prior_from_initial_grid


class TestImportance(unittest.TestCase):
    def test_importance_map_shape_and_bounds(self) -> None:
        grid = [
            [10, 10, 10, 10, 10, 10],
            [11, 11, 11, 11, 11, 11],
            [11, 11, 1, 11, 4, 11],
            [11, 11, 11, 11, 11, 11],
            [11, 5, 11, 11, 11, 11],
            [11, 11, 11, 11, 11, 11],
        ]
        prior = baseline_prior_from_initial_grid(grid)
        importance = dynamic_importance_map_from_initial_state(grid, prior=prior)

        self.assertEqual(len(importance), 6)
        self.assertEqual(len(importance[0]), 6)
        for row in importance:
            for value in row:
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)

    def test_settlement_and_coast_proximity_increase_importance(self) -> None:
        grid = [
            [10, 10, 10, 10, 10, 10, 10],
            [11, 11, 11, 11, 11, 11, 11],
            [11, 11, 1, 11, 11, 11, 11],
            [11, 11, 11, 11, 11, 11, 11],
            [11, 11, 11, 11, 11, 11, 11],
            [11, 11, 11, 11, 11, 11, 11],
            [11, 11, 11, 11, 11, 11, 11],
        ]
        importance = dynamic_importance_map_from_initial_state(grid)

        near = importance[2][2]
        far = importance[6][6]
        self.assertGreater(near, far)

    def test_port_settlement_boosts_nearby_importance(self) -> None:
        grid = [
            [10, 10, 10, 10, 10, 10],
            [11, 11, 11, 11, 11, 11],
            [11, 11, 11, 11, 11, 11],
            [11, 11, 11, 11, 11, 11],
            [11, 11, 11, 11, 11, 11],
            [11, 11, 11, 11, 11, 11],
        ]
        no_port = [Settlement(x=2, y=1, has_port=False, alive=True)]
        with_port = [Settlement(x=2, y=1, has_port=True, alive=True)]

        map_no_port = dynamic_importance_map_from_initial_state(grid, settlements=no_port)
        map_with_port = dynamic_importance_map_from_initial_state(grid, settlements=with_port)

        # A nearby coastal cell should be more important when a nearby port exists.
        self.assertGreater(map_with_port[1][2], map_no_port[1][2])

    def test_round_helper_returns_per_seed_maps(self) -> None:
        seed0 = SeedInitialState(
            grid=[[11 for _ in range(4)] for _ in range(4)],
            settlements=[Settlement(x=1, y=1, has_port=False, alive=True)],
        )
        seed1 = SeedInitialState(
            grid=[[10, 10, 10, 10], [11, 11, 11, 11], [11, 5, 11, 11], [11, 11, 11, 11]],
            settlements=[Settlement(x=1, y=1, has_port=True, alive=True)],
        )
        round_detail = RoundDetail(
            round_id="round-importance",
            round_number=1,
            status="completed",
            map_width=4,
            map_height=4,
            seeds_count=2,
            initial_states=[seed0, seed1],
            raw={},
        )

        maps = dynamic_importance_maps_for_round(round_detail)
        self.assertEqual(len(maps), 2)
        self.assertEqual(len(maps[0]), 4)
        self.assertEqual(len(maps[1][0]), 4)


if __name__ == "__main__":
    unittest.main()
