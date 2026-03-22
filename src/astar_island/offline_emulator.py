"""Offline round emulator for query-policy and model backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .models import SeedInitialState, Tensor3D
from .round_data import RoundRecord, load_round_dataset
from .scoring import score_round, score_seed
from .submission import floor_and_normalize, validate_prediction_tensor


@dataclass(slots=True)
class ViewportQuery:
    seed_index: int
    viewport_x: int
    viewport_y: int
    viewport_w: int = 15
    viewport_h: int = 15


@dataclass(slots=True)
class ViewportObservation:
    seed_index: int
    query_index: int
    grid: list[list[int]]
    settlements: list[dict[str, Any]]
    viewport: dict[str, int]
    width: int
    height: int
    queries_used: int
    queries_max: int
    available: bool
    source: str


@dataclass(slots=True)
class OfflineRoundState:
    round_id: str
    map_width: int
    map_height: int
    seeds_count: int
    initial_states: list[SeedInitialState | None]
    replay_available: list[bool]
    queries_max: int
    queries_used: int
    observations: list[ViewportObservation]


@dataclass(slots=True)
class SeedEvaluation:
    seed_index: int
    weighted_kl: float
    score: float


@dataclass(slots=True)
class OfflineRoundResult:
    round_id: str
    queries_used: int
    queries_max: int
    per_seed: list[SeedEvaluation]
    round_score: float
    observations: list[ViewportObservation]


class PolicyProtocol(Protocol):
    def next_query(self, state: OfflineRoundState) -> ViewportQuery | None:
        ...


class ModelProtocol(Protocol):
    def predict(
        self,
        round_state: OfflineRoundState,
        seed_initial_state: SeedInitialState | None,
        seed_index: int,
    ) -> Tensor3D:
        ...


def run_offline_round(
    policy: PolicyProtocol | Any,
    model: ModelProtocol | Any,
    round_id: str,
    *,
    logs_root: str = "logs",
    query_budget: int = 50,
    include_replays: bool = True,
    strict: bool = True,
    allow_missing_replays: bool = True,
    probability_floor: float | None = None,
) -> OfflineRoundResult:
    """Run a local offline emulation of a historical round."""
    round_record = _load_round_by_id(
        round_id=round_id,
        logs_root=logs_root,
        include_replays=include_replays,
        strict=strict,
    )

    state = OfflineRoundState(
        round_id=round_record.round_id,
        map_width=round_record.map_width,
        map_height=round_record.map_height,
        seeds_count=round_record.seeds_count,
        initial_states=[seed.initial_state for seed in round_record.seeds],
        replay_available=[seed.replay is not None for seed in round_record.seeds],
        queries_max=query_budget,
        queries_used=0,
        observations=[],
    )

    while state.queries_used < state.queries_max:
        query = _next_query(policy, state)
        if query is None:
            break
        obs = _simulate_viewport_query(
            round_record=round_record,
            query=query,
            query_index=state.queries_used + 1,
            queries_max=query_budget,
            allow_missing_replays=allow_missing_replays,
        )
        state.queries_used += 1
        state.observations.append(obs)

    per_seed: list[SeedEvaluation] = []
    for seed_record in round_record.seeds:
        if seed_record.analysis is None:
            if strict:
                raise ValueError(
                    f"Missing analysis for round={round_record.round_id} "
                    f"seed={seed_record.seed_index}"
                )
            per_seed.append(
                SeedEvaluation(
                    seed_index=seed_record.seed_index,
                    weighted_kl=0.0,
                    score=0.0,
                )
            )
            continue

        prediction = _predict_seed(
            model=model,
            state=state,
            seed_index=seed_record.seed_index,
            initial_state=seed_record.initial_state,
        )
        if probability_floor is not None:
            prediction = floor_and_normalize(prediction, floor=probability_floor)

        validation = validate_prediction_tensor(
            prediction,
            expected_width=round_record.map_width,
            expected_height=round_record.map_height,
        )
        if not validation.ok:
            first_err = validation.errors[0] if validation.errors else "unknown validation error"
            raise ValueError(
                f"Invalid prediction for round={round_record.round_id} seed={seed_record.seed_index}: "
                f"{first_err}"
            )

        wkl, score = score_seed(seed_record.analysis.ground_truth, prediction)
        per_seed.append(
            SeedEvaluation(
                seed_index=seed_record.seed_index,
                weighted_kl=wkl,
                score=score,
            )
        )

    round_score = score_round([seed.score for seed in per_seed], expected_seeds=round_record.seeds_count)
    return OfflineRoundResult(
        round_id=round_record.round_id,
        queries_used=state.queries_used,
        queries_max=query_budget,
        per_seed=per_seed,
        round_score=round_score,
        observations=state.observations,
    )


def _load_round_by_id(
    *,
    round_id: str,
    logs_root: str,
    include_replays: bool,
    strict: bool,
) -> RoundRecord:
    rounds = load_round_dataset(logs_root, include_replays=include_replays, strict=strict)
    by_id = {record.round_id: record for record in rounds}
    if round_id not in by_id:
        available = ", ".join(sorted(by_id.keys()))
        raise KeyError(f"Round id not found: {round_id}. Available: {available}")
    return by_id[round_id]


def _next_query(policy: Any, state: OfflineRoundState) -> ViewportQuery | None:
    if hasattr(policy, "next_query"):
        query = policy.next_query(state)
    elif callable(policy):
        query = policy(state)
    else:
        raise TypeError("policy must be callable or implement next_query(state)")

    if query is None:
        return None
    if isinstance(query, ViewportQuery):
        validated = query
    elif isinstance(query, dict):
        validated = ViewportQuery(**query)
    else:
        raise TypeError("policy query must be ViewportQuery, dict, or None")
    _validate_query(validated, state)
    return validated


def _predict_seed(
    *,
    model: Any,
    state: OfflineRoundState,
    seed_index: int,
    initial_state: SeedInitialState | None,
) -> Tensor3D:
    if hasattr(model, "predict"):
        prediction = model.predict(
            round_state=state,
            seed_initial_state=initial_state,
            seed_index=seed_index,
        )
    elif callable(model):
        prediction = model(state, initial_state, seed_index)
    else:
        raise TypeError(
            "model must be callable or implement "
            "predict(round_state, seed_initial_state, seed_index)"
        )

    if not isinstance(prediction, list):
        raise TypeError(f"Model prediction for seed {seed_index} must be a list tensor")
    return prediction


def _validate_query(query: ViewportQuery, state: OfflineRoundState) -> None:
    if query.seed_index < 0 or query.seed_index >= state.seeds_count:
        raise ValueError(f"seed_index out of range: {query.seed_index}")
    if query.viewport_x < 0 or query.viewport_y < 0:
        raise ValueError("viewport_x and viewport_y must be >= 0")
    if query.viewport_w < 5 or query.viewport_w > 15:
        raise ValueError(f"viewport_w must be 5..15, got {query.viewport_w}")
    if query.viewport_h < 5 or query.viewport_h > 15:
        raise ValueError(f"viewport_h must be 5..15, got {query.viewport_h}")


def _simulate_viewport_query(
    *,
    round_record: RoundRecord,
    query: ViewportQuery,
    query_index: int,
    queries_max: int,
    allow_missing_replays: bool,
) -> ViewportObservation:
    seed_record = round_record.seeds[query.seed_index]
    if seed_record.replay is None:
        if not allow_missing_replays:
            raise ValueError(
                f"Replay missing for round={round_record.round_id} seed={query.seed_index}"
            )
        x, y, w, h = _clamp_viewport(
            x=query.viewport_x,
            y=query.viewport_y,
            w=query.viewport_w,
            h=query.viewport_h,
            width=round_record.map_width,
            height=round_record.map_height,
        )
        return ViewportObservation(
            seed_index=query.seed_index,
            query_index=query_index,
            grid=[],
            settlements=[],
            viewport={"x": x, "y": y, "w": w, "h": h},
            width=round_record.map_width,
            height=round_record.map_height,
            queries_used=query_index,
            queries_max=queries_max,
            available=False,
            source="missing-replay",
        )

    frame = _final_frame(seed_record.replay, round_record.round_id, query.seed_index)
    full_grid = frame["grid"]
    settlements = frame.get("settlements", [])
    x, y, w, h = _clamp_viewport(
        x=query.viewport_x,
        y=query.viewport_y,
        w=query.viewport_w,
        h=query.viewport_h,
        width=round_record.map_width,
        height=round_record.map_height,
    )
    view_grid = [row[x : x + w] for row in full_grid[y : y + h]]
    view_settlements = [
        settlement
        for settlement in settlements
        if isinstance(settlement, dict)
        and int(settlement.get("x", -1)) >= x
        and int(settlement.get("x", -1)) < x + w
        and int(settlement.get("y", -1)) >= y
        and int(settlement.get("y", -1)) < y + h
    ]

    return ViewportObservation(
        seed_index=query.seed_index,
        query_index=query_index,
        grid=view_grid,
        settlements=view_settlements,
        viewport={"x": x, "y": y, "w": w, "h": h},
        width=round_record.map_width,
        height=round_record.map_height,
        queries_used=query_index,
        queries_max=queries_max,
        available=True,
        source="replay-frame-final",
    )


def _clamp_viewport(
    *,
    x: int,
    y: int,
    w: int,
    h: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    w = min(max(w, 5), min(15, width))
    h = min(max(h, 5), min(15, height))
    max_x = max(0, width - w)
    max_y = max(0, height - h)
    x = min(max(x, 0), max_x)
    y = min(max(y, 0), max_y)
    return x, y, w, h


def _final_frame(replay_payload: dict[str, Any], round_id: str, seed_index: int) -> dict[str, Any]:
    frames = replay_payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(
            f"Replay has no frames for round={round_id} seed={seed_index}"
        )

    # Highest step corresponds to the final world state.
    best = max(
        (frame for frame in frames if isinstance(frame, dict)),
        key=lambda frame: int(frame.get("step", 0)),
        default=None,
    )
    if best is None or "grid" not in best:
        raise ValueError(
            f"Replay frame missing grid for round={round_id} seed={seed_index}"
        )
    return best
