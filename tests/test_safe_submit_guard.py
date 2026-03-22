from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from safe_submit_round import PRODUCTION_WORLD_CFG, world_override_diffs


def _args_with_world(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "world_learning_rate": PRODUCTION_WORLD_CFG.learning_rate,
        "world_epochs": PRODUCTION_WORLD_CFG.epochs,
        "world_l2": PRODUCTION_WORLD_CFG.l2,
        "world_samples_per_epoch": PRODUCTION_WORLD_CFG.samples_per_epoch,
        "world_max_cells_per_seed": PRODUCTION_WORLD_CFG.max_cells_per_seed,
        "world_entropy_weight_power": PRODUCTION_WORLD_CFG.entropy_weight_power,
        "world_min_entropy_weight": PRODUCTION_WORLD_CFG.min_entropy_weight,
        "world_random_seed": PRODUCTION_WORLD_CFG.random_seed,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class TestSafeSubmitGuard(unittest.TestCase):
    def test_world_override_diffs_empty_for_production_values(self) -> None:
        args = _args_with_world()
        self.assertEqual(world_override_diffs(args), {})

    def test_world_override_diffs_detects_non_production_values(self) -> None:
        args = _args_with_world(world_learning_rate=0.06, world_max_cells_per_seed=1000)
        diffs = world_override_diffs(args)
        self.assertIn("learning_rate", diffs)
        self.assertIn("max_cells_per_seed", diffs)
        self.assertEqual(diffs["learning_rate"][0], 0.06)
        self.assertEqual(diffs["learning_rate"][1], PRODUCTION_WORLD_CFG.learning_rate)


if __name__ == "__main__":
    unittest.main()
