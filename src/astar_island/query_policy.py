"""Deterministic query scheduling policies for offline/live round probing."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .importance import dynamic_importance_map_from_initial_state
from .models import SeedInitialState
from .offline_emulator import OfflineRoundState, ViewportQuery
from .priors import baseline_prior_from_initial_grid


@dataclass(slots=True)
class DeterministicThreePhasePolicyConfig:
    query_budget: int = 50
    default_window: int = 15
    phase1_target: int = 15
    phase2_target: int = 20
    phase3_target: int = 15
    top_windows_per_seed: int = 8
    min_center_distance: float = 6.0


class DeterministicThreePhaseQueryPolicy:
    """Hand-designed 3-phase query scheduler (broad -> repeated -> exploit)."""

    def __init__(
        self,
        *,
        config: DeterministicThreePhasePolicyConfig | None = None,
    ) -> None:
        self.config = config or DeterministicThreePhasePolicyConfig()
        self._cached_signature: tuple[str, int, int, int, int] | None = None
        self._cached_plan: list[ViewportQuery] = []

    def next_query(self, state: OfflineRoundState) -> ViewportQuery | None:
        plan = self._plan_for_state(state)
        idx = int(state.queries_used)
        if idx < 0 or idx >= len(plan):
            return None
        return plan[idx]

    def planned_queries(self, state: OfflineRoundState) -> list[ViewportQuery]:
        """Return deterministic full-plan queries for the provided state."""
        return list(self._plan_for_state(state))

    def _plan_for_state(self, state: OfflineRoundState) -> list[ViewportQuery]:
        cfg = self.config
        signature = (
            str(state.round_id),
            int(state.map_width),
            int(state.map_height),
            int(state.seeds_count),
            int(cfg.query_budget),
        )
        if self._cached_signature == signature and self._cached_plan:
            return self._cached_plan
        plan = self._build_plan(state)
        self._cached_signature = signature
        self._cached_plan = plan
        return plan

    def _build_plan(self, state: OfflineRoundState) -> list[ViewportQuery]:
        cfg = self.config
        if cfg.query_budget <= 0 or state.seeds_count <= 0:
            return []

        window = min(
            max(5, int(cfg.default_window)),
            int(state.map_width),
            int(state.map_height),
        )
        if window < 5:
            return []

        phase1_count, phase2_count, phase3_count = _phase_counts(
            budget=cfg.query_budget,
            targets=(cfg.phase1_target, cfg.phase2_target, cfg.phase3_target),
        )
        seed_windows: dict[int, list[tuple[int, int]]] = {}
        for seed_index, seed_initial_state in enumerate(state.initial_states):
            seed_windows[seed_index] = _seed_windows(
                seed_initial_state=seed_initial_state,
                width=state.map_width,
                height=state.map_height,
                window=window,
                top_windows_per_seed=cfg.top_windows_per_seed,
                min_center_distance=cfg.min_center_distance,
            )

        plan: list[ViewportQuery] = []
        plan.extend(
            _phase_queries(
                seeds_count=state.seeds_count,
                windows_by_seed=seed_windows,
                window=window,
                query_count=phase1_count,
                index_pattern=(0, 1, 2),
            )
        )
        plan.extend(
            _phase_queries(
                seeds_count=state.seeds_count,
                windows_by_seed=seed_windows,
                window=window,
                query_count=phase2_count,
                index_pattern=(0, 1, 0, 1),
            )
        )
        plan.extend(
            _phase_queries(
                seeds_count=state.seeds_count,
                windows_by_seed=seed_windows,
                window=window,
                query_count=phase3_count,
                index_pattern=(2, 3, 4),
            )
        )
        return plan[: cfg.query_budget]


def _phase_counts(*, budget: int, targets: tuple[int, int, int]) -> tuple[int, int, int]:
    if budget <= 0:
        return (0, 0, 0)
    total_target = sum(targets)
    if total_target <= 0:
        return (budget, 0, 0)

    raw = [budget * (target / total_target) for target in targets]
    counts = [int(math.floor(value)) for value in raw]
    remainder = budget - sum(counts)
    fractional = sorted(
        ((raw[idx] - counts[idx], idx) for idx in range(len(counts))),
        reverse=True,
    )
    for _, idx in fractional[:remainder]:
        counts[idx] += 1
    return counts[0], counts[1], counts[2]


def _phase_queries(
    *,
    seeds_count: int,
    windows_by_seed: dict[int, list[tuple[int, int]]],
    window: int,
    query_count: int,
    index_pattern: tuple[int, ...],
) -> list[ViewportQuery]:
    out: list[ViewportQuery] = []
    if query_count <= 0 or seeds_count <= 0:
        return out

    pattern_len = len(index_pattern)
    for idx in range(query_count):
        seed_index = idx % seeds_count
        pattern_step = (idx // seeds_count) % pattern_len
        preferred_window_idx = index_pattern[pattern_step]
        windows = windows_by_seed.get(seed_index, [(0, 0)])
        x, y = windows[min(preferred_window_idx, len(windows) - 1)]
        out.append(
            ViewportQuery(
                seed_index=seed_index,
                viewport_x=x,
                viewport_y=y,
                viewport_w=window,
                viewport_h=window,
            )
        )
    return out


def _seed_windows(
    *,
    seed_initial_state: SeedInitialState | None,
    width: int,
    height: int,
    window: int,
    top_windows_per_seed: int,
    min_center_distance: float,
) -> list[tuple[int, int]]:
    center = _center_window(width=width, height=height, window=window)
    if seed_initial_state is None:
        return [center]

    prior = baseline_prior_from_initial_grid(
        seed_initial_state.grid,
        settlements=seed_initial_state.settlements,
    )
    importance = dynamic_importance_map_from_initial_state(
        seed_initial_state.grid,
        settlements=seed_initial_state.settlements,
        prior=prior,
    )
    top_windows = _top_windows(
        importance=importance,
        window=window,
        count=top_windows_per_seed,
        min_center_distance=min_center_distance,
    )

    settlement = _settlement_window(
        settlements=seed_initial_state.settlements,
        width=width,
        height=height,
        window=window,
    )
    coast_or_port = _coast_or_port_window(
        grid=seed_initial_state.grid,
        width=width,
        height=height,
        window=window,
    )
    ordered = [settlement, coast_or_port, (top_windows[0] if top_windows else center), center]
    ordered.extend(top_windows[1:])
    return _dedupe_windows(ordered)


def _top_windows(
    *,
    importance: list[list[float]],
    window: int,
    count: int,
    min_center_distance: float,
) -> list[tuple[int, int]]:
    height = len(importance)
    width = len(importance[0]) if height > 0 else 0
    if width <= 0 or height <= 0:
        return [(0, 0)]

    window = min(window, width, height)
    max_x = width - window
    max_y = height - window
    integral = _integral_image(importance)

    candidates: list[tuple[float, int, int]] = []
    for y in range(max_y + 1):
        for x in range(max_x + 1):
            score = _window_sum(integral=integral, x=x, y=y, w=window, h=window)
            candidates.append((score, x, y))
    candidates.sort(reverse=True, key=lambda row: row[0])

    selected: list[tuple[int, int]] = []
    centers: list[tuple[float, float]] = []
    for _, x, y in candidates:
        cx = x + (window / 2.0)
        cy = y + (window / 2.0)
        too_close = any(math.dist((cx, cy), (sx, sy)) < min_center_distance for sx, sy in centers)
        if too_close:
            continue
        selected.append((x, y))
        centers.append((cx, cy))
        if len(selected) >= count:
            break
    if not selected:
        selected.append(_center_window(width=width, height=height, window=window))
    return selected


def _integral_image(values: list[list[float]]) -> list[list[float]]:
    height = len(values)
    width = len(values[0]) if height > 0 else 0
    integral = [[0.0 for _ in range(width + 1)] for _ in range(height + 1)]
    for y in range(height):
        row_sum = 0.0
        for x in range(width):
            row_sum += values[y][x]
            integral[y + 1][x + 1] = integral[y][x + 1] + row_sum
    return integral


def _window_sum(integral: list[list[float]], *, x: int, y: int, w: int, h: int) -> float:
    y2 = y + h
    x2 = x + w
    return integral[y2][x2] - integral[y][x2] - integral[y2][x] + integral[y][x]


def _settlement_window(
    *,
    settlements,
    width: int,
    height: int,
    window: int,
) -> tuple[int, int]:  # noqa: ANN001
    alive = [settlement for settlement in settlements if bool(settlement.alive)]
    pick = alive[0] if alive else (settlements[0] if settlements else None)
    if pick is None:
        return _center_window(width=width, height=height, window=window)
    return _clamp_window_top_left(
        x=int(pick.x) - (window // 2),
        y=int(pick.y) - (window // 2),
        width=width,
        height=height,
        window=window,
    )


def _coast_or_port_window(
    *,
    grid: list[list[int]],
    width: int,
    height: int,
    window: int,
) -> tuple[int, int]:
    best_score = float("-inf")
    best_xy = _center_window(width=width, height=height, window=window)
    for y in range(height):
        for x in range(width):
            terrain = int(grid[y][x])
            coastal_neighbors = 0
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                nx = x + dx
                ny = y + dy
                if ny < 0 or ny >= height or nx < 0 or nx >= width:
                    continue
                if int(grid[ny][nx]) == 10:
                    coastal_neighbors += 1

            score = 0.0
            if terrain == 2:
                score += 2.0
            if terrain == 1:
                score += 0.6
            score += 0.4 * coastal_neighbors
            if score > best_score:
                best_score = score
                best_xy = (x, y)
    return _clamp_window_top_left(
        x=best_xy[0] - (window // 2),
        y=best_xy[1] - (window // 2),
        width=width,
        height=height,
        window=window,
    )


def _center_window(*, width: int, height: int, window: int) -> tuple[int, int]:
    max_x = max(0, width - window)
    max_y = max(0, height - window)
    return (max_x // 2, max_y // 2)


def _clamp_window_top_left(
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    window: int,
) -> tuple[int, int]:
    max_x = max(0, width - window)
    max_y = max(0, height - window)
    return (min(max(x, 0), max_x), min(max(y, 0), max_y))


def _dedupe_windows(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for point in points:
        if point in seen:
            continue
        seen.add(point)
        out.append(point)
    return out
