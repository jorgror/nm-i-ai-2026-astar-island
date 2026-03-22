from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from astar_island.submission import floor_and_normalize, validate_prediction_tensor
from astar_island.submission import build_safe_round_submission, missing_seed_indices


class TestSubmission(unittest.TestCase):
    def test_validate_prediction_tensor_ok(self) -> None:
        pred = [[[1 / 6] * 6, [1 / 6] * 6]]
        result = validate_prediction_tensor(pred, expected_width=2, expected_height=1)
        self.assertTrue(result.ok)
        self.assertEqual(result.errors, [])

    def test_validate_prediction_tensor_finds_errors(self) -> None:
        pred = [[[1.0, 0.0, 0.0], [-0.1, 1.1, 0.0, 0.0, 0.0, 0.0]]]
        result = validate_prediction_tensor(pred, expected_width=2, expected_height=1)
        self.assertFalse(result.ok)
        self.assertTrue(any("expected 6 probs" in err for err in result.errors))
        self.assertTrue(any("negative probability" in err for err in result.errors))

    def test_floor_and_normalize(self) -> None:
        pred = [[[0.9, 0.1, 0.0, 0.0, 0.0, 0.0]]]
        out = floor_and_normalize(pred, floor=0.01)

        probs = out[0][0]
        self.assertTrue(all(p > 0.0 for p in probs))
        self.assertTrue(math.isclose(sum(probs), 1.0, abs_tol=1e-12))

    def test_build_safe_round_submission_uses_fallback_for_missing_seed(self) -> None:
        model_seed0 = [[[1 / 6] * 6]]
        fallback = [[[0.9, 0.1, 0.0, 0.0, 0.0, 0.0]]]

        plan = build_safe_round_submission(
            round_id="r1",
            seeds_count=2,
            map_width=1,
            map_height=1,
            predictions_by_seed={0: model_seed0},
            fallback_prediction_for_seed=lambda _: fallback,
            probability_floor=0.01,
        )

        self.assertEqual(len(plan), 2)
        self.assertFalse(plan[0].used_fallback)
        self.assertTrue(plan[1].used_fallback)
        pred = plan[1].payload["prediction"]
        self.assertIsInstance(pred, list)
        self.assertTrue(math.isclose(sum(pred[0][0]), 1.0, abs_tol=1e-12))

    def test_build_safe_round_submission_uses_fallback_for_invalid_model_prediction(self) -> None:
        invalid_seed0 = [[[1.0, 0.0, 0.0]]]  # Wrong class count.
        fallback = [[[1 / 6] * 6]]

        plan = build_safe_round_submission(
            round_id="r1",
            seeds_count=1,
            map_width=1,
            map_height=1,
            predictions_by_seed={0: invalid_seed0},
            fallback_prediction_for_seed=lambda _: fallback,
            probability_floor=0.01,
        )

        self.assertEqual(len(plan), 1)
        self.assertTrue(plan[0].used_fallback)
        self.assertTrue(any("expected 6 probs" in err for err in plan[0].errors))

    def test_missing_seed_indices(self) -> None:
        missing = missing_seed_indices(submitted_seed_indices=[0, 2, 2], seeds_count=5)
        self.assertEqual(missing, [1, 3, 4])


if __name__ == "__main__":
    unittest.main()
