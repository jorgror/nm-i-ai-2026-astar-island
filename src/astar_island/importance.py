"""Heuristic dynamic-cell importance maps from initial state features."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from .models import Grid2D, RoundDetail, Settlement, Tensor3D
from .priors import baseline_prior_for_round

INF_DISTANCE = 10**9
OCEAN_TERRAIN = 10
FOREST_TERRAIN = 4
MOUNTAIN_TERRAIN = 5
PORT_TERRAIN = 2
SETTLEMENT_TERRAIN = 1


@dataclass(slots=True)
class ImportanceConfig:
    settlement_decay: float = 3.0
    coast_decay: float = 3.2
    port_decay: float = 4.0
    density_radius: int = 5
    density_cap: float = 6.0
    mountain_static_dampen: float = 0.4
    ocean_static_dampen: float = 0.6

    w_prior_dynamic: float = 0.24
    w_settlement_proximity: float = 0.18
    w_coast_proximity: float = 0.12
    w_port_influence: float = 0.10
    w_settlement_density: float = 0.12
    w_forest_adjacency: float = 0.07
    w_mountain_barrier: float = 0.06
    w_chokepoint: float = 0.05
    w_fjordness: float = 0.04
    w_component_size: float = 0.02


def dynamic_importance_map_from_initial_state(
    initial_grid: Grid2D,
    *,
    settlements: list[Settlement] | None = None,
    prior: Tensor3D | None = None,
    config: ImportanceConfig | None = None,
) -> list[list[float]]:
    """Build per-cell dynamic importance scores in [0,1]."""
    cfg = config or ImportanceConfig()
    _validate_grid(initial_grid)
    height = len(initial_grid)
    width = len(initial_grid[0]) if height > 0 else 0

    settlement_points = _collect_settlement_points(initial_grid, settlements)
    port_points = _collect_port_points(initial_grid, settlements)
    ocean_points = _collect_ocean_points(initial_grid)

    dist_settlement = _distance_map(height=height, width=width, points=settlement_points)
    dist_port = _distance_map(height=height, width=width, points=port_points)
    dist_ocean = _distance_map(height=height, width=width, points=ocean_points)

    component_sizes, max_component = _land_component_sizes(initial_grid)

    output: list[list[float]] = []
    for y in range(height):
        row_values: list[float] = []
        for x in range(width):
            terrain = int(initial_grid[y][x])

            prior_dynamic = _prior_dynamic_mass(prior, x=x, y=y, terrain=terrain)
            settlement_prox = _exp_decay(dist_settlement[y][x], cfg.settlement_decay)
            coast_prox = _exp_decay(dist_ocean[y][x], cfg.coast_decay)
            port_influence = _exp_decay(dist_port[y][x], cfg.port_decay)

            settlement_density = _settlement_density(
                x=x,
                y=y,
                settlements=settlement_points,
                radius=cfg.density_radius,
                cap=cfg.density_cap,
            )
            forest_adj = _adjacency_ratio(initial_grid, x=x, y=y, terrain_code=FOREST_TERRAIN)
            mountain_adj = _adjacency_ratio(initial_grid, x=x, y=y, terrain_code=MOUNTAIN_TERRAIN)
            chokepoint = 1.0 if _is_chokepoint(initial_grid, x=x, y=y) else 0.0
            fjordness = _ocean_neighbor_ratio(initial_grid, x=x, y=y)

            component_size = component_sizes[y][x]
            if component_size <= 0 or max_component <= 0:
                component_factor = 0.0
            else:
                component_factor = 1.0 - min(1.0, component_size / float(max_component))

            score = (
                cfg.w_prior_dynamic * prior_dynamic
                + cfg.w_settlement_proximity * settlement_prox
                + cfg.w_coast_proximity * coast_prox
                + cfg.w_port_influence * port_influence
                + cfg.w_settlement_density * settlement_density
                + cfg.w_forest_adjacency * forest_adj
                + cfg.w_mountain_barrier * mountain_adj
                + cfg.w_chokepoint * chokepoint
                + cfg.w_fjordness * fjordness
                + cfg.w_component_size * component_factor
            )

            if terrain == MOUNTAIN_TERRAIN:
                score *= cfg.mountain_static_dampen
            elif terrain == OCEAN_TERRAIN:
                score *= cfg.ocean_static_dampen

            row_values.append(_clamp01(score))
        output.append(row_values)

    return output


def dynamic_importance_maps_for_round(
    round_detail: RoundDetail,
    *,
    priors: list[Tensor3D] | None = None,
    config: ImportanceConfig | None = None,
) -> list[list[list[float]]]:
    """Build one importance map per seed for a round."""
    seed_priors = priors if priors is not None else baseline_prior_for_round(round_detail)
    maps: list[list[list[float]]] = []
    for seed_index, seed in enumerate(round_detail.initial_states):
        prior = seed_priors[seed_index] if seed_index < len(seed_priors) else None
        maps.append(
            dynamic_importance_map_from_initial_state(
                seed.grid,
                settlements=seed.settlements,
                prior=prior,
                config=config,
            )
        )
    return maps


def _prior_dynamic_mass(prior: Tensor3D | None, *, x: int, y: int, terrain: int) -> float:
    if prior is not None:
        probs = prior[y][x]
        return _clamp01(float(probs[1] + probs[2] + probs[3]))

    if terrain in (SETTLEMENT_TERRAIN, PORT_TERRAIN, 3):
        return 0.75
    if terrain == FOREST_TERRAIN:
        return 0.15
    if terrain == MOUNTAIN_TERRAIN:
        return 0.02
    return 0.25


def _settlement_density(
    *,
    x: int,
    y: int,
    settlements: list[tuple[int, int]],
    radius: int,
    cap: float,
) -> float:
    if not settlements or radius <= 0 or cap <= 0.0:
        return 0.0
    count = 0
    for sx, sy in settlements:
        if abs(x - sx) + abs(y - sy) <= radius:
            count += 1
    return _clamp01(count / cap)


def _collect_settlement_points(
    grid: Grid2D,
    settlements: list[Settlement] | None,
) -> list[tuple[int, int]]:
    points: set[tuple[int, int]] = set()
    for y, row in enumerate(grid):
        for x, terrain in enumerate(row):
            if int(terrain) in (SETTLEMENT_TERRAIN, PORT_TERRAIN):
                points.add((x, y))

    if settlements:
        for settlement in settlements:
            if settlement.alive:
                points.add((int(settlement.x), int(settlement.y)))
    return sorted(points)


def _collect_port_points(
    grid: Grid2D,
    settlements: list[Settlement] | None,
) -> list[tuple[int, int]]:
    points: set[tuple[int, int]] = set()
    for y, row in enumerate(grid):
        for x, terrain in enumerate(row):
            if int(terrain) == PORT_TERRAIN:
                points.add((x, y))
    if settlements:
        for settlement in settlements:
            if settlement.alive and settlement.has_port:
                points.add((int(settlement.x), int(settlement.y)))
    return sorted(points)


def _collect_ocean_points(grid: Grid2D) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for y, row in enumerate(grid):
        for x, terrain in enumerate(row):
            if int(terrain) == OCEAN_TERRAIN:
                points.append((x, y))
    return points


def _distance_map(*, height: int, width: int, points: list[tuple[int, int]]) -> list[list[int]]:
    distances = [[INF_DISTANCE for _ in range(width)] for _ in range(height)]
    if not points:
        return distances

    q: deque[tuple[int, int]] = deque()
    for x, y in points:
        if 0 <= x < width and 0 <= y < height and distances[y][x] != 0:
            distances[y][x] = 0
            q.append((x, y))

    while q:
        x, y = q.popleft()
        base = distances[y][x]
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            nx = x + dx
            ny = y + dy
            if ny < 0 or ny >= height or nx < 0 or nx >= width:
                continue
            candidate = base + 1
            if candidate < distances[ny][nx]:
                distances[ny][nx] = candidate
                q.append((nx, ny))
    return distances


def _land_component_sizes(grid: Grid2D) -> tuple[list[list[int]], int]:
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    comp_sizes = [[0 for _ in range(width)] for _ in range(height)]
    visited = [[False for _ in range(width)] for _ in range(height)]
    max_size = 0

    for y in range(height):
        for x in range(width):
            if visited[y][x] or int(grid[y][x]) == OCEAN_TERRAIN:
                continue

            q: deque[tuple[int, int]] = deque([(x, y)])
            visited[y][x] = True
            cells: list[tuple[int, int]] = []

            while q:
                cx, cy = q.popleft()
                cells.append((cx, cy))
                for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                    nx = cx + dx
                    ny = cy + dy
                    if ny < 0 or ny >= height or nx < 0 or nx >= width:
                        continue
                    if visited[ny][nx] or int(grid[ny][nx]) == OCEAN_TERRAIN:
                        continue
                    visited[ny][nx] = True
                    q.append((nx, ny))

            size = len(cells)
            max_size = max(max_size, size)
            for cx, cy in cells:
                comp_sizes[cy][cx] = size

    return comp_sizes, max_size


def _adjacency_ratio(grid: Grid2D, *, x: int, y: int, terrain_code: int) -> float:
    count = 0
    for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
        nx = x + dx
        ny = y + dy
        if ny < 0 or ny >= len(grid):
            continue
        if nx < 0 or nx >= len(grid[ny]):
            continue
        if int(grid[ny][nx]) == terrain_code:
            count += 1
    return count / 4.0


def _ocean_neighbor_ratio(grid: Grid2D, *, x: int, y: int) -> float:
    terrain = int(grid[y][x])
    if terrain == OCEAN_TERRAIN:
        return 0.0

    neighbors = 0
    ocean = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx = x + dx
            ny = y + dy
            if ny < 0 or ny >= len(grid):
                continue
            if nx < 0 or nx >= len(grid[ny]):
                continue
            neighbors += 1
            if int(grid[ny][nx]) == OCEAN_TERRAIN:
                ocean += 1
    if neighbors == 0:
        return 0.0
    return ocean / neighbors


def _is_chokepoint(grid: Grid2D, *, x: int, y: int) -> bool:
    terrain = int(grid[y][x])
    if terrain in (OCEAN_TERRAIN, MOUNTAIN_TERRAIN):
        return False

    height = len(grid)
    width = len(grid[0]) if height > 0 else 0

    def blocked(nx: int, ny: int) -> bool:
        if ny < 0 or ny >= height or nx < 0 or nx >= width:
            return True
        return int(grid[ny][nx]) in (OCEAN_TERRAIN, MOUNTAIN_TERRAIN)

    up = blocked(x, y - 1)
    down = blocked(x, y + 1)
    left = blocked(x - 1, y)
    right = blocked(x + 1, y)

    vertical_pass = (left and right) and (not up and not down)
    horizontal_pass = (up and down) and (not left and not right)
    return vertical_pass or horizontal_pass


def _exp_decay(distance: int, decay: float) -> float:
    if decay <= 0.0 or distance >= INF_DISTANCE:
        return 0.0
    return math.exp(-distance / decay)


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


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
