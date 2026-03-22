#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

from astar_island.offline_emulator import OfflineRoundState, ViewportObservation
from astar_island.parsing import parse_round_detail
from astar_island.priors import baseline_prior_from_initial_grid
from astar_island.query_policy import DeterministicThreePhaseQueryPolicy
from astar_island.round_latent import RoundLatentConditionalModel, RoundLatentConfig
from astar_island.submission import build_safe_round_submission, missing_seed_indices


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "https://api.ainm.no"


class ApiError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code
        self.message = message


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Safe one-button submission pipeline for Astar Island. "
            "Builds predictions for all seeds, applies floor+validation, submits, and verifies completeness."
        )
    )
    parser.add_argument("--round-id", default=None, help="Round id override. Defaults to active round.")
    parser.add_argument("--model", choices=["latent", "prior"], default="latent")
    parser.add_argument("--query-budget", type=int, default=50, help="Live simulate calls for latent model.")
    parser.add_argument("--disable-blending", action="store_true", help="Disable Step 11 empirical blending.")
    parser.add_argument("--probability-floor", type=float, default=0.01)
    parser.add_argument("--sum-tolerance", type=float, default=0.01)
    parser.add_argument("--submit-retries", type=int, default=3)
    parser.add_argument("--retry-sleep-seconds", type=float, default=0.75)
    parser.add_argument("--simulate-sleep-seconds", type=float, default=0.25)
    parser.add_argument(
        "--checkpoint-seconds",
        type=float,
        default=0.0,
        help="If > 0, resubmit checkpoint snapshots at this interval until max checkpoints/deadline.",
    )
    parser.add_argument("--max-checkpoints", type=int, default=1)
    parser.add_argument("--token", default=None, help="Bearer token override.")
    parser.add_argument("--cookie-token", default=None, help="Cookie token override.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API base URL.")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs" / "step12_safe_submission"),
        help="Directory for checkpoint run summaries.",
    )
    return parser.parse_args()


def build_headers(args: argparse.Namespace) -> dict[str, str]:
    bearer = args.token or os.environ.get("AINM_TOKEN") or os.environ.get("ASTAR_BEARER_TOKEN")
    cookie = args.cookie_token or os.environ.get("AINM_COOKIE_TOKEN") or os.environ.get("ASTAR_ACCESS_TOKEN")

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "astar-island-safe-submit/0.1",
    }
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    if cookie:
        headers["Cookie"] = f"access_token={cookie}"
    return headers


def api_request(
    method: str,
    path: str,
    *,
    headers: dict[str, str],
    payload: dict[str, Any] | None = None,
    base_url: str = DEFAULT_BASE_URL,
) -> Any:
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")

    req = request.Request(
        url=f"{base_url.rstrip('/')}{path}",
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with request.urlopen(req, timeout=30.0) as resp:
            text = resp.read().decode("utf-8")
            return json.loads(text) if text else None
    except error.HTTPError as exc:
        detail: str = str(exc.reason)
        try:
            body_text = exc.read().decode("utf-8")
            if body_text:
                parsed = json.loads(body_text)
                if isinstance(parsed, dict):
                    detail = str(parsed.get("detail") or parsed.get("message") or body_text)
                else:
                    detail = body_text
        except Exception:
            pass
        raise ApiError(exc.code, detail) from exc


def select_round_id(*, requested_round_id: str | None, headers: dict[str, str], base_url: str) -> str:
    if requested_round_id:
        return requested_round_id

    rounds = api_request("GET", "/astar-island/rounds", headers=headers, base_url=base_url)
    if not isinstance(rounds, list):
        raise RuntimeError("Unexpected /astar-island/rounds response.")

    active_rows = [row for row in rounds if isinstance(row, dict) and str(row.get("status")) == "active"]
    if not active_rows:
        raise RuntimeError("No active round found.")
    active_rows = sorted(active_rows, key=lambda row: int(row.get("round_number") or 0), reverse=True)
    return str(active_rows[0]["id"])


def fetch_round_detail(round_id: str, *, headers: dict[str, str], base_url: str):
    payload = api_request(
        "GET",
        f"/astar-island/rounds/{round_id}",
        headers=headers,
        base_url=base_url,
    )
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected /rounds/{round_id} response.")
    return parse_round_detail(payload)


def make_live_state(*, round_id: str, map_width: int, map_height: int, seeds_count: int, initial_states, query_budget: int) -> OfflineRoundState:  # noqa: ANN001
    state_initials = list(initial_states)
    if len(state_initials) < seeds_count:
        state_initials.extend([None] * (seeds_count - len(state_initials)))
    return OfflineRoundState(
        round_id=round_id,
        map_width=map_width,
        map_height=map_height,
        seeds_count=seeds_count,
        initial_states=state_initials[:seeds_count],
        replay_available=[True for _ in range(seeds_count)],
        queries_max=max(0, int(query_budget)),
        queries_used=0,
        observations=[],
    )


def run_live_queries(
    *,
    round_state: OfflineRoundState,
    headers: dict[str, str],
    base_url: str,
    sleep_seconds: float,
) -> None:
    if round_state.queries_max <= 0:
        return

    policy = DeterministicThreePhaseQueryPolicy()
    while round_state.queries_used < round_state.queries_max:
        query = policy.next_query(round_state)
        if query is None:
            break

        try:
            payload = api_request(
                "POST",
                "/astar-island/simulate",
                headers=headers,
                base_url=base_url,
                payload={
                    "round_id": round_state.round_id,
                    "seed_index": int(query.seed_index),
                    "viewport_x": int(query.viewport_x),
                    "viewport_y": int(query.viewport_y),
                    "viewport_w": int(query.viewport_w),
                    "viewport_h": int(query.viewport_h),
                },
            )
        except ApiError as exc:
            if exc.status_code == 429:
                break
            raise

        if not isinstance(payload, dict):
            break

        queries_used = int(payload.get("queries_used") or (round_state.queries_used + 1))
        obs = ViewportObservation(
            seed_index=int(query.seed_index),
            query_index=queries_used,
            grid=payload.get("grid") if isinstance(payload.get("grid"), list) else [],
            settlements=payload.get("settlements") if isinstance(payload.get("settlements"), list) else [],
            viewport=payload.get("viewport") if isinstance(payload.get("viewport"), dict) else {},
            width=int(payload.get("width") or round_state.map_width),
            height=int(payload.get("height") or round_state.map_height),
            queries_used=queries_used,
            queries_max=int(payload.get("queries_max") or round_state.queries_max),
            available=True,
            source="live-simulate",
        )
        round_state.observations.append(obs)
        round_state.queries_used = min(queries_used, round_state.queries_max)
        if sleep_seconds > 0.0:
            time.sleep(sleep_seconds)


def build_predictions_by_seed(
    *,
    round_state: OfflineRoundState,
    model_name: str,
    probability_floor: float,
    enable_blending: bool,
) -> dict[int, list[list[list[float]]] | None]:
    predictions: dict[int, list[list[list[float]]] | None] = {}
    if model_name == "prior":
        for seed_index in range(round_state.seeds_count):
            initial_state = round_state.initial_states[seed_index]
            if initial_state is None:
                predictions[seed_index] = None
                continue
            predictions[seed_index] = baseline_prior_from_initial_grid(
                initial_state.grid,
                settlements=initial_state.settlements,
            )
        return predictions

    model = RoundLatentConditionalModel(
        config=RoundLatentConfig(
            probability_floor=probability_floor,
            enable_observation_blend=enable_blending,
        )
    )
    for seed_index in range(round_state.seeds_count):
        initial_state = round_state.initial_states[seed_index]
        try:
            predictions[seed_index] = model.predict(
                round_state=round_state,
                seed_initial_state=initial_state,
                seed_index=seed_index,
            )
        except Exception:
            predictions[seed_index] = None
    return predictions


def submit_with_retries(
    *,
    payload: dict[str, object],
    headers: dict[str, str],
    base_url: str,
    retries: int,
    retry_sleep_seconds: float,
) -> None:
    attempts = max(1, int(retries))
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            api_request(
                "POST",
                "/astar-island/submit",
                headers=headers,
                base_url=base_url,
                payload=payload,
            )
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= attempts:
                break
            if retry_sleep_seconds > 0.0:
                time.sleep(retry_sleep_seconds)
    if last_exc is not None:
        raise last_exc


def verify_submissions(
    *,
    round_id: str,
    seeds_count: int,
    headers: dict[str, str],
    base_url: str,
) -> dict[str, object]:
    predictions_payload = api_request(
        "GET",
        f"/astar-island/my-predictions/{round_id}",
        headers=headers,
        base_url=base_url,
    )
    submitted_seeds: list[int] = []
    if isinstance(predictions_payload, list):
        for row in predictions_payload:
            if isinstance(row, dict) and row.get("seed_index") is not None:
                try:
                    submitted_seeds.append(int(row["seed_index"]))
                except (TypeError, ValueError):
                    continue

    missing = missing_seed_indices(
        submitted_seed_indices=submitted_seeds,
        seeds_count=seeds_count,
    )

    my_rounds_payload = api_request("GET", "/astar-island/my-rounds", headers=headers, base_url=base_url)
    seeds_submitted = None
    if isinstance(my_rounds_payload, list):
        for row in my_rounds_payload:
            if not isinstance(row, dict):
                continue
            if str(row.get("id")) == str(round_id):
                if row.get("seeds_submitted") is not None:
                    seeds_submitted = int(row["seeds_submitted"])
                break

    return {
        "submitted_seed_indices": sorted(set(submitted_seeds)),
        "missing_seed_indices": missing,
        "seeds_submitted": seeds_submitted,
    }


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def fallback_prediction(round_state: OfflineRoundState, seed_index: int) -> list[list[list[float]]]:
    initial_state = round_state.initial_states[seed_index]
    if initial_state is None:
        cell = [1.0 / 6.0 for _ in range(6)]
        return [[cell[:] for _ in range(round_state.map_width)] for _ in range(round_state.map_height)]
    return baseline_prior_from_initial_grid(
        initial_state.grid,
        settlements=initial_state.settlements,
    )


def parse_iso8601(value: object) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def main() -> int:
    args = parse_args()
    headers = build_headers(args)

    try:
        round_id = select_round_id(
            requested_round_id=args.round_id,
            headers=headers,
            base_url=args.base_url,
        )
        print(f"Selected round_id={round_id}")
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to select round: {exc}", file=sys.stderr)
        return 2

    output_root = Path(args.output_dir)
    checkpoint_seconds = max(0.0, float(args.checkpoint_seconds))
    max_checkpoints = max(1, int(args.max_checkpoints))
    completed_checkpoints = 0
    last_missing: list[int] = []
    last_summary_path: Path | None = None

    while completed_checkpoints < max_checkpoints:
        checkpoint_idx = completed_checkpoints + 1
        try:
            detail = fetch_round_detail(round_id, headers=headers, base_url=args.base_url)
            state = make_live_state(
                round_id=detail.round_id,
                map_width=detail.map_width,
                map_height=detail.map_height,
                seeds_count=detail.seeds_count,
                initial_states=detail.initial_states,
                query_budget=args.query_budget if args.model == "latent" else 0,
            )

            if args.model == "latent":
                run_live_queries(
                    round_state=state,
                    headers=headers,
                    base_url=args.base_url,
                    sleep_seconds=max(0.0, float(args.simulate_sleep_seconds)),
                )

            predictions_by_seed = build_predictions_by_seed(
                round_state=state,
                model_name=args.model,
                probability_floor=float(args.probability_floor),
                enable_blending=not bool(args.disable_blending),
            )
            plans = build_safe_round_submission(
                round_id=detail.round_id,
                seeds_count=detail.seeds_count,
                map_width=detail.map_width,
                map_height=detail.map_height,
                predictions_by_seed=predictions_by_seed,
                fallback_prediction_for_seed=lambda seed_idx: fallback_prediction(state, seed_idx),
                probability_floor=float(args.probability_floor),
                sum_tolerance=float(args.sum_tolerance),
            )

            for plan in plans:
                submit_with_retries(
                    payload=plan.payload,
                    headers=headers,
                    base_url=args.base_url,
                    retries=args.submit_retries,
                    retry_sleep_seconds=float(args.retry_sleep_seconds),
                )
                if args.retry_sleep_seconds > 0.0:
                    time.sleep(float(args.retry_sleep_seconds))

            verification = verify_submissions(
                round_id=detail.round_id,
                seeds_count=detail.seeds_count,
                headers=headers,
                base_url=args.base_url,
            )
            missing = list(verification.get("missing_seed_indices") or [])

            if missing:
                plans_by_seed = {plan.seed_index: plan for plan in plans}
                for seed_index in missing:
                    if seed_index not in plans_by_seed:
                        continue
                    submit_with_retries(
                        payload=plans_by_seed[seed_index].payload,
                        headers=headers,
                        base_url=args.base_url,
                        retries=args.submit_retries,
                        retry_sleep_seconds=float(args.retry_sleep_seconds),
                    )
                    if args.retry_sleep_seconds > 0.0:
                        time.sleep(float(args.retry_sleep_seconds))
                verification = verify_submissions(
                    round_id=detail.round_id,
                    seeds_count=detail.seeds_count,
                    headers=headers,
                    base_url=args.base_url,
                )
                missing = list(verification.get("missing_seed_indices") or [])

            timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            summary = {
                "checkpoint_index": checkpoint_idx,
                "generated_at": datetime.now(tz=timezone.utc).isoformat(),
                "round_id": detail.round_id,
                "round_status": detail.status,
                "model": args.model,
                "queries_used": state.queries_used,
                "queries_max": state.queries_max,
                "probability_floor": float(args.probability_floor),
                "sum_tolerance": float(args.sum_tolerance),
                "fallback_seeds": [plan.seed_index for plan in plans if plan.used_fallback],
                "fallback_errors_by_seed": {
                    str(plan.seed_index): plan.errors for plan in plans if plan.errors
                },
                "verification": verification,
            }
            summary_path = output_root / detail.round_id / f"checkpoint_{checkpoint_idx}_{timestamp}.json"
            write_summary(summary_path, summary)
            last_summary_path = summary_path
            last_missing = missing

            print(
                f"checkpoint={checkpoint_idx} submitted={detail.seeds_count - len(missing)}/{detail.seeds_count} "
                f"queries_used={state.queries_used}/{state.queries_max} summary={summary_path}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Checkpoint {checkpoint_idx} failed: {exc}", file=sys.stderr)
            return 1

        completed_checkpoints += 1
        if checkpoint_seconds <= 0.0 or completed_checkpoints >= max_checkpoints:
            break

        closes_at = parse_iso8601(detail.raw.get("closes_at"))
        now = datetime.now(tz=timezone.utc)
        if closes_at is not None and closes_at.tzinfo is None:
            closes_at = closes_at.replace(tzinfo=timezone.utc)
        if closes_at is not None and now >= closes_at:
            break
        time.sleep(checkpoint_seconds)

    if last_missing:
        print(f"Incomplete submission; missing seeds={last_missing}", file=sys.stderr)
        if last_summary_path is not None:
            print(f"Last summary: {last_summary_path}", file=sys.stderr)
        return 1

    print("Safe submission pipeline finished with all seeds submitted.")
    if last_summary_path is not None:
        print(f"Last summary: {last_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
