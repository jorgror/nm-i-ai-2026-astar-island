"""Round-latent inference and latent-conditioned seed prediction."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from .constants import GRID_TO_CLASS, NUM_CLASSES
from .importance import dynamic_importance_map_from_initial_state
from .models import SeedInitialState, Tensor3D
from .priors import baseline_prior_from_initial_grid
from .submission import floor_and_normalize

if TYPE_CHECKING:
    from .offline_emulator import OfflineRoundState, ViewportObservation


@dataclass(slots=True)
class RoundLatentConfig:
    epsilon: float = 1e-6
    probability_floor: float = 1e-4
    reference_class_probs: tuple[float, float, float, float, float, float] = (
        0.82,
        0.07,
        0.03,
        0.03,
        0.03,
        0.02,
    )
    class_bias_scale: float = 0.85
    bias_clip: float = 1.5

    coverage_for_full_confidence: float = 0.10
    settlement_samples_for_full_confidence: int = 40
    repeated_ratio_for_full_confidence: float = 0.25

    min_cell_strength: float = 0.15
    dynamic_strength_scale: float = 0.90
    static_cell_dampen: float = 0.35
    preserve_mountains: bool = True

    enable_observation_blend: bool = True
    empirical_prior_strength: float = 0.4
    empirical_neighbor_smoothing: float = 0.2
    observation_confidence_scale: float = 1.8
    repeated_observation_bonus: float = 0.2
    repeated_observation_saturation: int = 3
    min_observed_blend_weight: float = 0.05
    max_observed_blend_weight: float = 0.92
    dynamic_blend_boost: float = 0.45
    static_blend_dampen: float = 0.65


@dataclass(slots=True)
class RoundLatentVector:
    values: list[float]
    class_logit_offsets: list[float]
    confidence: float
    observed_cells: int
    observed_settlements: int
    repeated_probe_ratio: float


class SeedBasePredictor(Protocol):
    def __call__(self, seed_initial_state: SeedInitialState, seed_index: int) -> Tensor3D:
        ...


class RoundLatentEncoder:
    """Infer a compact latent vector from current round observations."""

    def __init__(self, config: RoundLatentConfig | None = None) -> None:
        self.config = config or RoundLatentConfig()

    def infer(self, round_state: OfflineRoundState) -> RoundLatentVector:
        cfg = self.config
        class_counts = [0 for _ in range(NUM_CLASSES)]
        observed_cells: Counter[tuple[int, int, int]] = Counter()

        settlement_count = 0
        alive_sum = 0.0
        port_sum = 0.0
        prosperity_sum = 0.0
        prosperity_obs = 0

        for obs in round_state.observations:
            self._accumulate_observation(
                obs=obs,
                class_counts=class_counts,
                observed_cells=observed_cells,
            )
            for settlement in obs.settlements:
                if not isinstance(settlement, dict):
                    continue
                settlement_count += 1
                alive_sum += 1.0 if bool(settlement.get("alive", True)) else 0.0
                port_sum += 1.0 if bool(settlement.get("has_port", False)) else 0.0

                sample_values: list[float] = []
                for key in ("population", "food", "wealth", "defense"):
                    if key in settlement:
                        sample_values.append(_normalize_scalar(settlement[key]))
                if sample_values:
                    prosperity_sum += sum(sample_values) / float(len(sample_values))
                    prosperity_obs += 1

        total_cells = sum(class_counts)
        repeated = sum(max(0, count - 1) for count in observed_cells.values())
        repeated_probe_ratio = (
            repeated / float(total_cells) if total_cells > 0 else 0.0
        )
        class_freq = (
            [count / float(total_cells) for count in class_counts]
            if total_cells > 0
            else list(cfg.reference_class_probs)
        )

        alive_rate = alive_sum / float(settlement_count) if settlement_count > 0 else 0.5
        port_rate = (
            port_sum / float(settlement_count)
            if settlement_count > 0
            else class_freq[2]
        )
        ruin_rate = class_freq[3]
        prosperity = prosperity_sum / float(prosperity_obs) if prosperity_obs > 0 else 0.5

        confidence = self._confidence(
            round_state=round_state,
            unique_observed_cells=len(observed_cells),
            observed_settlements=settlement_count,
            repeated_probe_ratio=repeated_probe_ratio,
        )
        class_offsets = self._class_offsets(
            class_freq=class_freq,
            alive_rate=alive_rate,
            port_rate=port_rate,
            ruin_rate=ruin_rate,
            prosperity=prosperity,
            confidence=confidence,
        )

        values = [
            class_offsets[1],
            class_offsets[2],
            class_offsets[3],
            alive_rate,
            port_rate,
            ruin_rate,
            prosperity,
            repeated_probe_ratio,
            confidence,
        ]
        return RoundLatentVector(
            values=values,
            class_logit_offsets=class_offsets,
            confidence=confidence,
            observed_cells=total_cells,
            observed_settlements=settlement_count,
            repeated_probe_ratio=repeated_probe_ratio,
        )

    def _accumulate_observation(
        self,
        *,
        obs: ViewportObservation,
        class_counts: list[int],
        observed_cells: Counter[tuple[int, int, int]],
    ) -> None:
        if not obs.available or not obs.grid:
            return

        vx = int(obs.viewport.get("x", 0))
        vy = int(obs.viewport.get("y", 0))
        for local_y, row in enumerate(obs.grid):
            if not isinstance(row, list):
                continue
            for local_x, terrain in enumerate(row):
                class_idx = GRID_TO_CLASS.get(int(terrain), 0)
                class_counts[class_idx] += 1
                key = (obs.seed_index, vx + local_x, vy + local_y)
                observed_cells[key] += 1

    def _confidence(
        self,
        *,
        round_state: OfflineRoundState,
        unique_observed_cells: int,
        observed_settlements: int,
        repeated_probe_ratio: float,
    ) -> float:
        cfg = self.config
        total_cells = (
            max(1, round_state.map_width)
            * max(1, round_state.map_height)
            * max(1, round_state.seeds_count)
        )
        coverage = unique_observed_cells / float(total_cells)
        coverage_conf = _clamp01(coverage / cfg.coverage_for_full_confidence)
        settlement_conf = _clamp01(
            observed_settlements / float(max(1, cfg.settlement_samples_for_full_confidence))
        )
        repeat_conf = _clamp01(
            repeated_probe_ratio / max(cfg.repeated_ratio_for_full_confidence, cfg.epsilon)
        )
        return _clamp01(0.55 * coverage_conf + 0.30 * settlement_conf + 0.15 * repeat_conf)

    def _class_offsets(
        self,
        *,
        class_freq: list[float],
        alive_rate: float,
        port_rate: float,
        ruin_rate: float,
        prosperity: float,
        confidence: float,
    ) -> list[float]:
        cfg = self.config
        offsets: list[float] = []
        for freq, ref in zip(class_freq, cfg.reference_class_probs):
            raw = math.log((freq + cfg.epsilon) / (ref + cfg.epsilon))
            scaled = raw * cfg.class_bias_scale * confidence
            offsets.append(_clamp(scaled, -cfg.bias_clip, cfg.bias_clip))

        survival_shift = _clamp((alive_rate - 0.5) * 2.0, -1.0, 1.0) * confidence
        port_shift = _clamp((port_rate - 0.35) * 2.0, -1.0, 1.0) * confidence
        ruin_shift = _clamp((ruin_rate - 0.08) * 2.0, -1.0, 1.0) * confidence
        prosperity_shift = _clamp((prosperity - 0.5) * 2.0, -1.0, 1.0) * confidence

        offsets[1] += 0.35 * survival_shift + 0.10 * port_shift + 0.15 * prosperity_shift
        offsets[2] += 0.20 * survival_shift + 0.25 * port_shift + 0.10 * prosperity_shift
        offsets[3] += -0.35 * survival_shift + 0.30 * ruin_shift - 0.12 * prosperity_shift
        offsets[0] += -0.08 * survival_shift - 0.08 * prosperity_shift

        return [_clamp(value, -cfg.bias_clip, cfg.bias_clip) for value in offsets]


class RoundLatentConditionalModel:
    """Latent-conditioned predictor compatible with offline emulator model interface."""

    def __init__(
        self,
        *,
        config: RoundLatentConfig | None = None,
        encoder: RoundLatentEncoder | None = None,
        base_predictor: SeedBasePredictor | None = None,
    ) -> None:
        cfg = config or RoundLatentConfig()
        self.config = cfg
        self.encoder = encoder or RoundLatentEncoder(config=cfg)
        self.base_predictor = base_predictor or _default_seed_prior_predictor
        self._cached_latent_key: tuple[str, int, int, int] | None = None
        self._cached_latent: RoundLatentVector | None = None

    def infer_latent(self, round_state: OfflineRoundState) -> RoundLatentVector:
        observations_signature = _observations_signature(round_state)
        key = (
            str(round_state.round_id),
            int(round_state.queries_used),
            len(round_state.observations),
            observations_signature,
        )
        if self._cached_latent_key == key and self._cached_latent is not None:
            return self._cached_latent
        latent = self.encoder.infer(round_state)
        self._cached_latent_key = key
        self._cached_latent = latent
        return latent

    def predict(
        self,
        round_state: OfflineRoundState,
        seed_initial_state: SeedInitialState | None,
        seed_index: int = 0,
    ) -> Tensor3D:
        if seed_initial_state is None:
            return _uniform_tensor(
                width=round_state.map_width,
                height=round_state.map_height,
            )

        base = self.base_predictor(seed_initial_state, seed_index)
        latent = self.infer_latent(round_state)
        if latent.confidence <= 0.0:
            return floor_and_normalize(base, floor=self.config.probability_floor)

        importance = dynamic_importance_map_from_initial_state(
            seed_initial_state.grid,
            settlements=seed_initial_state.settlements,
            prior=base,
        )
        adjusted: Tensor3D = []
        for y, row in enumerate(base):
            out_row: list[list[float]] = []
            for x, base_cell in enumerate(row):
                terrain = int(seed_initial_state.grid[y][x])
                if self.config.preserve_mountains and terrain == 5:
                    out_row.append(list(base_cell))
                    continue

                local_strength = (
                    self.config.min_cell_strength
                    + self.config.dynamic_strength_scale * importance[y][x]
                )
                if terrain in (0, 4, 10, 11):
                    local_strength *= self.config.static_cell_dampen
                local_strength *= latent.confidence

                out_row.append(
                    _apply_logit_offsets(
                        base_cell=base_cell,
                        offsets=latent.class_logit_offsets,
                        strength=local_strength,
                        epsilon=self.config.epsilon,
                    )
                )
            adjusted.append(out_row)

        blended = _blend_with_observations(
            prediction=adjusted,
            round_state=round_state,
            seed_index=seed_index,
            initial_grid=seed_initial_state.grid,
            importance=importance,
            config=self.config,
        )
        return floor_and_normalize(blended, floor=self.config.probability_floor)


def _default_seed_prior_predictor(seed_initial_state: SeedInitialState, _: int) -> Tensor3D:
    return baseline_prior_from_initial_grid(
        seed_initial_state.grid,
        settlements=seed_initial_state.settlements,
    )


def _blend_with_observations(
    *,
    prediction: Tensor3D,
    round_state: OfflineRoundState,
    seed_index: int,
    initial_grid: list[list[int]],
    importance: list[list[float]],
    config: RoundLatentConfig,
) -> Tensor3D:
    if not config.enable_observation_blend:
        return prediction

    height = len(prediction)
    width = len(prediction[0]) if height > 0 else 0
    if width <= 0 or height <= 0:
        return prediction

    counts, totals = _observation_counts_for_seed(
        round_state=round_state,
        seed_index=seed_index,
        width=width,
        height=height,
    )
    if not any(total > 0 for row in totals for total in row):
        return prediction

    empirical = _empirical_posteriors_from_counts(
        counts=counts,
        totals=totals,
        prediction=prediction,
        prior_strength=config.empirical_prior_strength,
    )
    neighbor_empirical = _neighbor_empirical_posteriors(
        empirical=empirical,
        totals=totals,
    )

    out: Tensor3D = []
    for y in range(height):
        out_row: list[list[float]] = []
        for x in range(width):
            base_cell = prediction[y][x]
            samples = totals[y][x]
            if samples <= 0:
                out_row.append(list(base_cell))
                continue

            empirical_cell = empirical[y][x]
            if empirical_cell is None:
                out_row.append(list(base_cell))
                continue

            neighbor = neighbor_empirical[y][x]
            if (
                neighbor is not None
                and config.empirical_neighbor_smoothing > 0.0
            ):
                empirical_cell = _normalize_probs(
                    [
                        (1.0 - config.empirical_neighbor_smoothing) * empirical_cell[idx]
                        + config.empirical_neighbor_smoothing * neighbor[idx]
                        for idx in range(NUM_CLASSES)
                    ],
                    epsilon=config.epsilon,
                )

            weight = _blend_weight(
                samples=samples,
                terrain=int(initial_grid[y][x]),
                dynamic_importance=float(importance[y][x]),
                config=config,
            )
            out_row.append(
                _normalize_probs(
                    [
                        (1.0 - weight) * base_cell[idx] + weight * empirical_cell[idx]
                        for idx in range(NUM_CLASSES)
                    ],
                    epsilon=config.epsilon,
                )
            )
        out.append(out_row)
    return out


def _observation_counts_for_seed(
    *,
    round_state: OfflineRoundState,
    seed_index: int,
    width: int,
    height: int,
) -> tuple[list[list[list[int]]], list[list[int]]]:
    counts = [
        [[0 for _ in range(NUM_CLASSES)] for _ in range(width)]
        for _ in range(height)
    ]
    totals = [[0 for _ in range(width)] for _ in range(height)]

    for obs in round_state.observations:
        if obs.seed_index != seed_index or not obs.available or not obs.grid:
            continue
        vx = int(obs.viewport.get("x", 0))
        vy = int(obs.viewport.get("y", 0))
        for local_y, row in enumerate(obs.grid):
            gy = vy + local_y
            if gy < 0 or gy >= height:
                continue
            for local_x, terrain in enumerate(row):
                gx = vx + local_x
                if gx < 0 or gx >= width:
                    continue
                cls = GRID_TO_CLASS.get(int(terrain), 0)
                counts[gy][gx][cls] += 1
                totals[gy][gx] += 1
    return counts, totals


def _empirical_posteriors_from_counts(
    *,
    counts: list[list[list[int]]],
    totals: list[list[int]],
    prediction: Tensor3D,
    prior_strength: float,
) -> list[list[list[float] | None]]:
    height = len(totals)
    width = len(totals[0]) if height > 0 else 0
    out: list[list[list[float] | None]] = []
    for y in range(height):
        out_row: list[list[float] | None] = []
        for x in range(width):
            sample_count = totals[y][x]
            if sample_count <= 0:
                out_row.append(None)
                continue
            denom = float(sample_count) + max(0.0, prior_strength)
            probs: list[float] = []
            for cls in range(NUM_CLASSES):
                numerator = float(counts[y][x][cls]) + max(0.0, prior_strength) * prediction[y][x][cls]
                probs.append(numerator / denom if denom > 0.0 else prediction[y][x][cls])
            out_row.append(_normalize_probs(probs, epsilon=1e-12))
        out.append(out_row)
    return out


def _neighbor_empirical_posteriors(
    *,
    empirical: list[list[list[float] | None]],
    totals: list[list[int]],
) -> list[list[list[float] | None]]:
    height = len(empirical)
    width = len(empirical[0]) if height > 0 else 0
    out: list[list[list[float] | None]] = []
    for y in range(height):
        out_row: list[list[float] | None] = []
        for x in range(width):
            acc = [0.0 for _ in range(NUM_CLASSES)]
            weight_sum = 0.0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx = x + dx
                    ny = y + dy
                    if ny < 0 or ny >= height or nx < 0 or nx >= width:
                        continue
                    neighbor_empirical = empirical[ny][nx]
                    if neighbor_empirical is None:
                        continue
                    neighbor_weight = float(totals[ny][nx])
                    if neighbor_weight <= 0.0:
                        continue
                    for cls in range(NUM_CLASSES):
                        acc[cls] += neighbor_weight * neighbor_empirical[cls]
                    weight_sum += neighbor_weight
            if weight_sum <= 0.0:
                out_row.append(None)
            else:
                out_row.append([value / weight_sum for value in acc])
        out.append(out_row)
    return out


def _blend_weight(
    *,
    samples: int,
    terrain: int,
    dynamic_importance: float,
    config: RoundLatentConfig,
) -> float:
    base = float(samples) / (float(samples) + max(config.observation_confidence_scale, 1e-6))
    repeats = max(0, samples - 1)
    repeat_fraction = repeats / float(max(1, config.repeated_observation_saturation))
    repeat_fraction = _clamp01(repeat_fraction)
    base += config.repeated_observation_bonus * repeat_fraction

    dynamic_scale = 1.0 + (config.dynamic_blend_boost * _clamp01(dynamic_importance))
    weight = base * dynamic_scale
    if terrain in (0, 4, 5, 10, 11):
        weight *= config.static_blend_dampen

    weight = max(config.min_observed_blend_weight, weight)
    return _clamp(weight, 0.0, config.max_observed_blend_weight)


def _apply_logit_offsets(
    *,
    base_cell: list[float],
    offsets: list[float],
    strength: float,
    epsilon: float,
) -> list[float]:
    logits: list[float] = []
    for idx in range(NUM_CLASSES):
        p = max(float(base_cell[idx]), epsilon)
        logits.append(math.log(p) + (strength * offsets[idx]))

    max_logit = max(logits)
    exp_values = [math.exp(value - max_logit) for value in logits]
    denom = sum(exp_values)
    if denom <= 0.0:
        return [1.0 / NUM_CLASSES for _ in range(NUM_CLASSES)]
    return [value / denom for value in exp_values]


def _uniform_tensor(*, width: int, height: int) -> Tensor3D:
    cell = [1.0 / NUM_CLASSES for _ in range(NUM_CLASSES)]
    row = [cell[:] for _ in range(width)]
    return [row[:] for _ in range(height)]


def _normalize_probs(probs: list[float], *, epsilon: float) -> list[float]:
    out = [max(float(value), epsilon) for value in probs]
    denom = sum(out)
    if denom <= 0.0:
        return [1.0 / NUM_CLASSES for _ in range(NUM_CLASSES)]
    return [value / denom for value in out]


def _normalize_scalar(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.5
    if numeric < 0.0:
        return 0.0
    return numeric / (1.0 + numeric)


def _clamp01(value: float) -> float:
    return _clamp(value, 0.0, 1.0)


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


def _observations_signature(round_state: OfflineRoundState) -> int:
    """Compute a lightweight deterministic signature for latent-cache invalidation."""
    acc = 1469598103934665603  # FNV offset basis.
    for obs in round_state.observations:
        acc = _fnv_mix(acc, int(obs.seed_index))
        acc = _fnv_mix(acc, int(obs.query_index))
        viewport = obs.viewport or {}
        acc = _fnv_mix(acc, int(viewport.get("x", 0)))
        acc = _fnv_mix(acc, int(viewport.get("y", 0)))
        acc = _fnv_mix(acc, int(viewport.get("w", 0)))
        acc = _fnv_mix(acc, int(viewport.get("h", 0)))
        acc = _fnv_mix(acc, 1 if obs.available else 0)
        acc = _fnv_mix(acc, len(obs.grid))
        for row in obs.grid:
            if not isinstance(row, list):
                continue
            acc = _fnv_mix(acc, len(row))
            for value in row:
                try:
                    terrain = int(value)
                except (TypeError, ValueError):
                    terrain = 0
                acc = _fnv_mix(acc, terrain)
        acc = _fnv_mix(acc, len(obs.settlements))
        for settlement in obs.settlements:
            if not isinstance(settlement, dict):
                continue
            acc = _fnv_mix(acc, int(bool(settlement.get("alive", True))))
            acc = _fnv_mix(acc, int(bool(settlement.get("has_port", False))))
            for key in ("x", "y", "owner_id"):
                try:
                    acc = _fnv_mix(acc, int(settlement.get(key, 0)))
                except (TypeError, ValueError):
                    acc = _fnv_mix(acc, 0)
    return acc


def _fnv_mix(current: int, value: int) -> int:
    mixed = current ^ (int(value) & 0xFFFFFFFFFFFFFFFF)
    return (mixed * 1099511628211) & 0xFFFFFFFFFFFFFFFF
