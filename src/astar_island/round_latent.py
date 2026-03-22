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
        self._cached_latent_key: tuple[str, int, int] | None = None
        self._cached_latent: RoundLatentVector | None = None

    def infer_latent(self, round_state: OfflineRoundState) -> RoundLatentVector:
        key = (
            str(round_state.round_id),
            int(round_state.queries_used),
            len(round_state.observations),
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

        return floor_and_normalize(adjusted, floor=self.config.probability_floor)


def _default_seed_prior_predictor(seed_initial_state: SeedInitialState, _: int) -> Tensor3D:
    return baseline_prior_from_initial_grid(
        seed_initial_state.grid,
        settlements=seed_initial_state.settlements,
    )


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
