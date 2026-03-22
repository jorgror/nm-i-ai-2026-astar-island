from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from astar_island.scoring import score_round, score_seed, weighted_kl


class TestScoring(unittest.TestCase):
    def test_perfect_prediction_scores_100(self) -> None:
        gt = [[[0.5, 0.5, 0.0, 0.0, 0.0, 0.0]]]
        pred = [[[0.5, 0.5, 0.0, 0.0, 0.0, 0.0]]]

        wkl, score = score_seed(gt, pred)

        self.assertTrue(math.isclose(wkl, 0.0, abs_tol=1e-12))
        self.assertTrue(math.isclose(score, 100.0, abs_tol=1e-12))

    def test_zero_prediction_probability_gives_zero_score(self) -> None:
        gt = [[[0.5, 0.5, 0.0, 0.0, 0.0, 0.0]]]
        pred = [[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]

        wkl, score = score_seed(gt, pred)

        self.assertTrue(math.isinf(wkl))
        self.assertEqual(score, 0.0)

    def test_round_score_counts_missing_seed_as_zero(self) -> None:
        result = score_round([100.0, 50.0], expected_seeds=5)
        self.assertTrue(math.isclose(result, 30.0, abs_tol=1e-12))

    def test_weighted_kl_ignores_zero_entropy_cells(self) -> None:
        # First cell is static (entropy 0), second cell is dynamic.
        gt = [[
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
        ]]
        pred = [[
            [0.1, 0.9, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
        ]]

        self.assertTrue(math.isclose(weighted_kl(gt, pred), 0.0, abs_tol=1e-12))


if __name__ == "__main__":
    unittest.main()
