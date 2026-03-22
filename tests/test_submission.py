from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from astar_island.submission import floor_and_normalize, validate_prediction_tensor


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


if __name__ == "__main__":
    unittest.main()
