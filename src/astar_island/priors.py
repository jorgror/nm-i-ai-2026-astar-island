"""Mechanics-first prior generators from initial map state only."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .constants import NUM_CLASSES
from .models import Grid2D, RoundDetail, Settlement, Tensor3D
from .scoring import cell_entropy

CLASS_EMPTY = 0
CLASS_SETTLEMENT = 1
CLASS_PORT = 2
CLASS_RUIN = 3
CLASS_FOREST = 4
CLASS_MOUNTAIN = 5

CLASS0_TERRAINS = {0, 10, 11}

BASE_CELL_PRIORS: dict[int, tuple[float, float, float, float, float, float]] = {
    # empty / ocean / plains-like terrain: mostly class 0, but keep dynamic mass alive
    0: (0.92, 0.03, 0.01, 0.02, 0.01, 0.01),
    10: (0.92, 0.03, 0.01, 0.02, 0.01, 0.01),
    11: (0.92, 0.03, 0.01, 0.02, 0.01, 0.01),
    # dynamic terrain classes
    1: (0.08, 0.62, 0.12, 0.12, 0.04, 0.02),  # settlement
    2: (0.08, 0.20, 0.58, 0.08, 0.04, 0.02),  # port
    3: (0.12, 0.12, 0.08, 0.54, 0.12, 0.02),  # ruin
    # mostly-static classes
    4: (0.05, 0.02, 0.01, 0.03, 0.86, 0.03),  # forest
    5: (0.005, 0.002, 0.001, 0.002, 0.005, 0.985),  # mountain
}


@dataclass(slots=True)
class PriorConfig:
    floor: float = 1e-4
    settlement_radius: int = 2
    settlement_boost: float = 0.08
    ruin_boost: float = 0.04
    coastal_port_boost: float = 0.08
    coastal_settlement_boost: float = 0.02
    remote_class0_boost: float = 0.06
    remote_distance: int = 6


def baseline_prior_from_initial_grid(
    initial_grid: Grid2D,
    *,
    settlements: list[Settlement] | None = None,
    config: PriorConfig | None = None,
) -> Tensor3D:
    """Generate a mechanics-first baseline prior tensor from initial map only."""
    cfg = config or PriorConfig()
    _validate_grid(initial_grid)

    height = len(initial_grid)
    width = len(initial_grid[0]) if height > 0 else 0
    settlement_positions = _collect_settlement_positions(initial_grid, settlements)

    prior: Tensor3D = []
    for y, row in enumerate(initial_grid):
        out_row: list[list[float]] = []
        for x, terrain in enumerate(row):
            base = list(BASE_CELL_PRIORS.get(int(terrain), BASE_CELL_PRIORS[0]))

            # Mountains are treated as highly static and not adjusted by local heuristics.
            if int(terrain) != 5:
                coastal = _is_coastal(initial_grid, x=x, y=y)
                dist_to_settlement = _min_manhattan_distance(x, y, settlement_positions)

                if dist_to_settlement <= cfg.settlement_radius:
                    base[CLASS_SETTLEMENT] += cfg.settlement_boost
                    base[CLASS_RUIN] += cfg.ruin_boost

                if coastal:
                    base[CLASS_PORT] += cfg.coastal_port_boost
                    if dist_to_settlement <= cfg.settlement_radius + 1:
                        base[CLASS_SETTLEMENT] += cfg.coastal_settlement_boost

                if (
                    int(terrain) in CLASS0_TERRAINS
                    and dist_to_settlement >= cfg.remote_distance
                ):
                    base[CLASS_EMPTY] += cfg.remote_class0_boost

            out_row.append(_normalize(base, floor=cfg.floor))
        prior.append(out_row)

    return prior


def baseline_prior_for_round(
    round_detail: RoundDetail,
    *,
    config: PriorConfig | None = None,
) -> list[Tensor3D]:
    """Generate one prior tensor per seed from round initial states only."""
    priors: list[Tensor3D] = []
    for seed in round_detail.initial_states:
        priors.append(
            baseline_prior_from_initial_grid(
                seed.grid,
                settlements=seed.settlements,
                config=config,
            )
        )
    return priors


def dynamic_importance_from_prior(prior: Tensor3D) -> list[list[float]]:
    """Compute a dynamic-cell importance map from a prior tensor."""
    max_entropy = math.log(NUM_CLASSES)
    importance: list[list[float]] = []
    for row in prior:
        out_row: list[float] = []
        for probs in row:
            dynamic_mass = probs[CLASS_SETTLEMENT] + probs[CLASS_PORT] + probs[CLASS_RUIN]
            ent_norm = 0.0 if max_entropy <= 0.0 else cell_entropy(probs) / max_entropy
            value = (0.6 * dynamic_mass) + (0.4 * ent_norm)
            out_row.append(max(0.0, min(1.0, value)))
        importance.append(out_row)
    return importance


def _collect_settlement_positions(
    grid: Grid2D,
    settlements: list[Settlement] | None,
) -> list[tuple[int, int]]:
    positions: set[tuple[int, int]] = set()
    for y, row in enumerate(grid):
        for x, terrain in enumerate(row):
            if int(terrain) == 1:
                positions.add((x, y))

    if settlements:
        for settlement in settlements:
            if settlement.alive:
                positions.add((int(settlement.x), int(settlement.y)))
    return sorted(positions)


def _is_coastal(grid: Grid2D, *, x: int, y: int) -> bool:
    for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
        nx = x + dx
        ny = y + dy
        if ny < 0 or ny >= len(grid):
            continue
        if nx < 0 or nx >= len(grid[ny]):
            continue
        if int(grid[ny][nx]) == 10:
            return True
    return False


def _min_manhattan_distance(x: int, y: int, points: list[tuple[int, int]]) -> int:
    if not points:
        return 10**9
    return min(abs(x - px) + abs(y - py) for px, py in points)


def _normalize(probs: list[float], *, floor: float) -> list[float]:
    if floor < 0.0:
        raise ValueError("floor must be non-negative")

    out = [max(float(value), floor) for value in probs]
    total = sum(out)
    if total <= 0.0:
        return [1.0 / NUM_CLASSES] * NUM_CLASSES
    return [value / total for value in out]


def _validate_grid(grid: Grid2D) -> None:
    if not isinstance(grid, list):
        raise ValueError("initial_grid must be a list")
    if not grid:
        return

    width = len(grid[0])
    for y, row in enumerate(grid):
        if not isinstance(row, list):
            raise ValueError(f"grid row {y} is not a list")
        if len(row) != width:
            raise ValueError(
                f"grid row width mismatch at row {y}: {len(row)} vs expected {width}"
            )
