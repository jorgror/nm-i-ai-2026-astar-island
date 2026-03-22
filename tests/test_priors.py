from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from astar_island.models import RoundDetail, SeedInitialState
from astar_island.priors import (
    baseline_prior_for_round,
    baseline_prior_from_initial_grid,
    dynamic_importance_from_prior,
)


class TestPriors(unittest.TestCase):
    def test_baseline_prior_tensor_is_normalized(self) -> None:
        grid = [
            [5, 4, 11],
            [10, 1, 3],
            [0, 2, 11],
        ]
        prior = baseline_prior_from_initial_grid(grid)

        self.assertEqual(len(prior), 3)
        self.assertEqual(len(prior[0]), 3)
        self.assertEqual(len(prior[0][0]), 6)
        for row in prior:
            for cell in row:
                self.assertTrue(all(p > 0.0 for p in cell))
                self.assertTrue(math.isclose(sum(cell), 1.0, rel_tol=0.0, abs_tol=1e-9))

    def test_core_mechanics_biases(self) -> None:
        grid = [[5, 4, 11, 10, 0, 1, 2, 3]]
        prior = baseline_prior_from_initial_grid(grid)

        mountain = prior[0][0]
        forest = prior[0][1]
        plains = prior[0][2]
        ocean = prior[0][3]
        empty = prior[0][4]

        self.assertGreater(mountain[5], 0.95)
        self.assertEqual(max(range(6), key=lambda i: mountain[i]), 5)

        self.assertGreater(forest[4], 0.75)
        self.assertEqual(max(range(6), key=lambda i: forest[i]), 4)

        self.assertGreater(plains[0], 0.75)
        self.assertGreater(ocean[0], 0.75)
        self.assertGreater(empty[0], 0.75)

    def test_coastal_and_settlement_proximity_raise_dynamic_mass(self) -> None:
        # (1,1) is settlement. (1,2) is coastal (ocean above) and near settlement.
        grid = [
            [10, 10, 10, 10, 10],
            [11, 1, 11, 11, 11],
            [11, 11, 11, 11, 11],
            [11, 11, 11, 11, 11],
            [11, 11, 11, 11, 11],
        ]
        prior = baseline_prior_from_initial_grid(grid)

        near_coastal = prior[2][1]
        far_inland = prior[4][4]

        near_dynamic = near_coastal[1] + near_coastal[2] + near_coastal[3]
        far_dynamic = far_inland[1] + far_inland[2] + far_inland[3]
        self.assertGreater(near_dynamic, far_dynamic)

    def test_baseline_prior_for_round_returns_one_tensor_per_seed(self) -> None:
        seed0 = SeedInitialState(grid=[[11, 11], [11, 11]], settlements=[])
        seed1 = SeedInitialState(grid=[[4, 5], [1, 2]], settlements=[])
        round_detail = RoundDetail(
            round_id="r1",
            round_number=1,
            status="completed",
            map_width=2,
            map_height=2,
            seeds_count=2,
            initial_states=[seed0, seed1],
            raw={},
        )

        priors = baseline_prior_for_round(round_detail)
        self.assertEqual(len(priors), 2)
        self.assertEqual(len(priors[0]), 2)
        self.assertEqual(len(priors[1][0][0]), 6)

    def test_dynamic_importance_from_prior_bounds(self) -> None:
        grid = [
            [5, 5, 5],
            [11, 1, 11],
            [11, 11, 11],
        ]
        prior = baseline_prior_from_initial_grid(grid)
        importance = dynamic_importance_from_prior(prior)

        self.assertEqual(len(importance), 3)
        self.assertEqual(len(importance[0]), 3)
        for row in importance:
            for value in row:
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)

        # Center settlement-adjacent zone should be more dynamic than mountain corner.
        self.assertGreater(importance[1][1], importance[0][0])


if __name__ == "__main__":
    unittest.main()
