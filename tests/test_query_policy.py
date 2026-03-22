from __future__ import annotations

import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from astar_island.models import SeedInitialState, Settlement
from astar_island.offline_emulator import OfflineRoundState
from astar_island.query_policy import (
    DeterministicThreePhasePolicyConfig,
    DeterministicThreePhaseQueryPolicy,
)


def _make_seed_state(width: int, height: int, seed_index: int) -> SeedInitialState:
    grid: list[list[int]] = []
    for y in range(height):
        row: list[int] = []
        for x in range(width):
            if x < 5:
                row.append(10)  # ocean strip for coast targeting
            elif (x + y + seed_index) % 13 == 0:
                row.append(5)  # mountains
            elif (x * 3 + y + seed_index) % 11 == 0:
                row.append(4)  # forest
            else:
                row.append(11)  # plains
        grid.append(row)

    settlements = [
        Settlement(x=6 + (seed_index * 2), y=8 + seed_index, has_port=(seed_index % 2 == 0), alive=True),
        Settlement(x=20 + seed_index, y=22, has_port=False, alive=True),
    ]
    return SeedInitialState(grid=grid, settlements=settlements)


def _make_state(width: int = 40, height: int = 40, seeds_count: int = 5) -> OfflineRoundState:
    initial_states = [
        _make_seed_state(width, height, seed_idx) for seed_idx in range(seeds_count)
    ]
    return OfflineRoundState(
        round_id="round-policy-test",
        map_width=width,
        map_height=height,
        seeds_count=seeds_count,
        initial_states=initial_states,
        replay_available=[True for _ in range(seeds_count)],
        queries_max=50,
        queries_used=0,
        observations=[],
    )


class TestDeterministicThreePhasePolicy(unittest.TestCase):
    def test_default_plan_has_50_queries_and_valid_bounds(self) -> None:
        state = _make_state()
        policy = DeterministicThreePhaseQueryPolicy()
        plan = policy.planned_queries(state)

        self.assertEqual(len(plan), 50)
        for query in plan:
            self.assertGreaterEqual(query.seed_index, 0)
            self.assertLess(query.seed_index, state.seeds_count)
            self.assertGreaterEqual(query.viewport_x, 0)
            self.assertGreaterEqual(query.viewport_y, 0)
            self.assertEqual(query.viewport_w, 15)
            self.assertEqual(query.viewport_h, 15)
            self.assertLessEqual(query.viewport_x, state.map_width - query.viewport_w)
            self.assertLessEqual(query.viewport_y, state.map_height - query.viewport_h)

    def test_phase_1_covers_each_seed_three_times(self) -> None:
        state = _make_state()
        policy = DeterministicThreePhaseQueryPolicy()
        phase1 = policy.planned_queries(state)[:15]
        counts = Counter(query.seed_index for query in phase1)
        self.assertEqual(set(counts.keys()), set(range(state.seeds_count)))
        self.assertTrue(all(counts[seed] == 3 for seed in range(state.seeds_count)))

    def test_phase_2_has_repeated_windows_per_seed(self) -> None:
        state = _make_state()
        policy = DeterministicThreePhaseQueryPolicy()
        phase2 = policy.planned_queries(state)[15:35]

        windows_by_seed: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for query in phase2:
            windows_by_seed[query.seed_index].append((query.viewport_x, query.viewport_y))

        for seed in range(state.seeds_count):
            windows = windows_by_seed[seed]
            self.assertEqual(len(windows), 4)
            counts = Counter(windows)
            self.assertTrue(all(count >= 2 for count in counts.values()))

    def test_phase_3_spreads_across_seeds(self) -> None:
        state = _make_state()
        policy = DeterministicThreePhaseQueryPolicy()
        phase3 = policy.planned_queries(state)[35:]
        counts = Counter(query.seed_index for query in phase3)
        self.assertEqual(set(counts.keys()), set(range(state.seeds_count)))
        self.assertTrue(all(counts[seed] == 3 for seed in range(state.seeds_count)))

    def test_plan_is_deterministic(self) -> None:
        state = _make_state()
        a = DeterministicThreePhaseQueryPolicy().planned_queries(state)
        b = DeterministicThreePhaseQueryPolicy().planned_queries(state)

        self.assertEqual(
            [(q.seed_index, q.viewport_x, q.viewport_y, q.viewport_w, q.viewport_h) for q in a],
            [(q.seed_index, q.viewport_x, q.viewport_y, q.viewport_w, q.viewport_h) for q in b],
        )

    def test_scaled_budget_keeps_queries_in_budget(self) -> None:
        state = _make_state()
        policy = DeterministicThreePhaseQueryPolicy(
            config=DeterministicThreePhasePolicyConfig(query_budget=17)
        )
        plan = policy.planned_queries(state)
        self.assertEqual(len(plan), 17)

        phase1 = plan[:5]  # with proportional scaling: phase1 gets 5
        seed_set = {query.seed_index for query in phase1}
        self.assertEqual(seed_set, set(range(state.seeds_count)))


if __name__ == "__main__":
    unittest.main()
