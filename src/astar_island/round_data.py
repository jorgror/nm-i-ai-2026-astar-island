"""Round-centric dataset loading and leave-one-round-out splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .models import AnalysisSeedData, RoundDetail, SeedInitialState
from .parsing import load_json, parse_analysis_seed, parse_round_detail


@dataclass(slots=True)
class RoundSeedRecord:
    seed_index: int
    initial_state: SeedInitialState | None
    analysis: AnalysisSeedData | None
    replay: dict[str, Any] | None
    analysis_path: str | None
    replay_path: str | None


@dataclass(slots=True)
class RoundRecord:
    round_id: str
    round_number: int | None
    status: str | None
    map_width: int
    map_height: int
    seeds_count: int
    round_detail: RoundDetail
    metadata: dict[str, Any]
    seeds: list[RoundSeedRecord]
    round_dir: str


@dataclass(slots=True)
class LeaveOneRoundOutSplit:
    holdout_round_id: str
    training_rounds: list[RoundRecord]
    validation_round: RoundRecord


def load_round_dataset(
    logs_root: str | Path,
    *,
    include_replays: bool = True,
    strict: bool = True,
) -> list[RoundRecord]:
    """Load local logs into one canonical record per round."""
    logs_path = Path(logs_root)
    if not logs_path.exists():
        raise FileNotFoundError(f"Logs path does not exist: {logs_path}")

    round_records: list[RoundRecord] = []
    for round_dir in _iter_round_dirs(logs_path):
        round_details_path = round_dir / "round-details.json"
        if not round_details_path.exists():
            if strict:
                raise FileNotFoundError(f"Missing round-details.json in {round_dir}")
            continue

        round_detail = parse_round_detail(load_json(round_details_path))
        if strict and len(round_detail.initial_states) < round_detail.seeds_count:
            raise ValueError(
                f"Round {round_detail.round_id} has {len(round_detail.initial_states)} initial states, "
                f"expected {round_detail.seeds_count}"
            )

        seeds: list[RoundSeedRecord] = []
        for seed_index in range(round_detail.seeds_count):
            initial_state = (
                round_detail.initial_states[seed_index]
                if seed_index < len(round_detail.initial_states)
                else None
            )
            analysis_path = round_dir / f"analysis-seed-{seed_index}.json"
            replay_path = round_dir / f"replay-seed-{seed_index}.json"

            analysis = None
            if analysis_path.exists():
                analysis = parse_analysis_seed(load_json(analysis_path))
                _validate_analysis_shape(round_detail, analysis, round_dir, seed_index)
            elif strict:
                raise FileNotFoundError(
                    f"Missing analysis file for round={round_detail.round_id} seed={seed_index}"
                )

            replay = None
            if include_replays and replay_path.exists():
                loaded = load_json(replay_path)
                if not isinstance(loaded, dict):
                    raise ValueError(
                        f"Replay payload must be an object: round={round_detail.round_id} "
                        f"seed={seed_index}"
                    )
                replay = loaded

            seeds.append(
                RoundSeedRecord(
                    seed_index=seed_index,
                    initial_state=initial_state,
                    analysis=analysis,
                    replay=replay,
                    analysis_path=str(analysis_path) if analysis_path.exists() else None,
                    replay_path=str(replay_path) if replay_path.exists() else None,
                )
            )

        metadata = _extract_round_metadata(round_detail.raw)
        round_records.append(
            RoundRecord(
                round_id=round_detail.round_id,
                round_number=round_detail.round_number,
                status=round_detail.status,
                map_width=round_detail.map_width,
                map_height=round_detail.map_height,
                seeds_count=round_detail.seeds_count,
                round_detail=round_detail,
                metadata=metadata,
                seeds=seeds,
                round_dir=str(round_dir),
            )
        )

    if strict and not round_records:
        raise ValueError(f"No rounds found in {logs_path}")
    return round_records


def leave_one_round_out_splits(rounds: Iterable[RoundRecord]) -> list[LeaveOneRoundOutSplit]:
    ordered_rounds = list(rounds)
    if len(ordered_rounds) < 2:
        raise ValueError("Need at least 2 rounds for leave-one-round-out")

    splits: list[LeaveOneRoundOutSplit] = []
    for holdout_idx, validation_round in enumerate(ordered_rounds):
        training_rounds = [
            candidate
            for idx, candidate in enumerate(ordered_rounds)
            if idx != holdout_idx
        ]
        splits.append(
            LeaveOneRoundOutSplit(
                holdout_round_id=validation_round.round_id,
                training_rounds=training_rounds,
                validation_round=validation_round,
            )
        )
    return splits


def load_leave_one_round_out(
    logs_root: str | Path,
    *,
    include_replays: bool = True,
    strict: bool = True,
) -> list[LeaveOneRoundOutSplit]:
    rounds = load_round_dataset(
        logs_root,
        include_replays=include_replays,
        strict=strict,
    )
    return leave_one_round_out_splits(rounds)


def _extract_round_metadata(round_payload: dict[str, Any]) -> dict[str, Any]:
    excluded = {"initial_states"}
    return {key: value for key, value in round_payload.items() if key not in excluded}


def _iter_round_dirs(logs_root: Path) -> list[Path]:
    round_dirs = [path for path in logs_root.iterdir() if path.is_dir()]

    def sort_key(round_dir: Path) -> tuple[int, int, str]:
        details_path = round_dir / "round-details.json"
        if details_path.exists():
            try:
                payload = load_json(details_path)
                round_number = int(payload.get("round_number") or 0)
                return (0, -round_number, round_dir.name)
            except Exception:
                pass
        return (1, 0, round_dir.name)

    return sorted(round_dirs, key=sort_key)


def _validate_analysis_shape(
    round_detail: RoundDetail,
    analysis: AnalysisSeedData,
    round_dir: Path,
    seed_index: int,
) -> None:
    if analysis.width != round_detail.map_width or analysis.height != round_detail.map_height:
        raise ValueError(
            "Analysis shape mismatch for "
            f"round_dir={round_dir} seed={seed_index}: "
            f"analysis=({analysis.width},{analysis.height}) "
            f"round=({round_detail.map_width},{round_detail.map_height})"
        )
