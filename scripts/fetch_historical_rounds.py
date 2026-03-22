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


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "https://api.ainm.no"


class ApiError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code
        self.message = message


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def build_headers(args: argparse.Namespace) -> dict[str, str]:
    bearer = (
        args.token
        or os.environ.get("AINM_TOKEN")
        or os.environ.get("ASTAR_BEARER_TOKEN")
    )
    cookie = (
        args.cookie_token
        or os.environ.get("AINM_COOKIE_TOKEN")
        or os.environ.get("ASTAR_ACCESS_TOKEN")
    )

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "astar-island-tools/0.1",
    }
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    if cookie:
        headers["Cookie"] = f"access_token={cookie}"
    return headers


def api_request(
    method: str,
    path: str,
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
        detail = exc.reason
        try:
            body_text = exc.read().decode("utf-8")
            if body_text:
                parsed = json.loads(body_text)
                if isinstance(parsed, dict):
                    detail = parsed.get("detail") or parsed.get("message") or body_text
                else:
                    detail = body_text
        except Exception:
            pass
        raise ApiError(exc.code, str(detail)) from exc


def round_dir(logs_root: Path, round_id: str) -> Path:
    path = logs_root / round_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_log_path(logs_root: Path, round_id: str) -> Path:
    return round_dir(logs_root, round_id) / "runs.jsonl"


def analysis_seed_path(logs_root: Path, round_id: str, seed_index: int) -> Path:
    return round_dir(logs_root, round_id) / f"analysis-seed-{seed_index}.json"


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":"), ensure_ascii=True))
        handle.write("\n")


def append_run_log(file_path: Path, round_id: str, run_type: str, payload: dict[str, Any]) -> None:
    append_jsonl(
        file_path,
        {
            "event_type": "run",
            "logged_at": utc_now_iso(),
            "round_id": round_id,
            "run_type": run_type,
            "payload": payload,
        },
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def ensure_round_manifest(logs_root: Path, round_id: str, round_details: dict[str, Any]) -> Path:
    manifest_path = round_dir(logs_root, round_id) / "round_manifest.json"
    payload = {
        "created_at": utc_now_iso(),
        "round_id": round_id,
        "round_number": round_details.get("round_number"),
        "status": round_details.get("status"),
        "map_width": round_details.get("map_width"),
        "map_height": round_details.get("map_height"),
        "seeds_count": round_details.get("seeds_count"),
        "round_details": round_details,
    }
    write_json(manifest_path, payload)
    return manifest_path


def list_rounds(headers: dict[str, str], base_url: str) -> list[dict[str, Any]]:
    payload = api_request("GET", "/astar-island/rounds", headers=headers, base_url=base_url)
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected /rounds response")
    return payload


def get_round_details(round_id: str, headers: dict[str, str], base_url: str) -> dict[str, Any]:
    payload = api_request(
        "GET",
        f"/astar-island/rounds/{round_id}",
        headers=headers,
        base_url=base_url,
    )
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected /rounds/{round_id} response")
    return payload


def get_analysis(round_id: str, seed_index: int, headers: dict[str, str], base_url: str) -> dict[str, Any]:
    payload = api_request(
        "GET",
        f"/astar-island/analysis/{round_id}/{seed_index}",
        headers=headers,
        base_url=base_url,
    )
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected /analysis response for round={round_id} seed={seed_index}")
    return payload


def select_rounds(
    headers: dict[str, str],
    base_url: str,
    explicit_round_ids: list[str],
    status: str,
    max_rounds: int,
) -> list[dict[str, Any]]:
    if explicit_round_ids:
        selected: list[dict[str, Any]] = []
        for round_id in explicit_round_ids:
            details = get_round_details(round_id, headers=headers, base_url=base_url)
            selected.append(
                {
                    "id": str(details["id"]),
                    "round_number": details.get("round_number"),
                    "status": details.get("status"),
                }
            )
        return selected

    rounds = list_rounds(headers=headers, base_url=base_url)
    if status != "all":
        rounds = [row for row in rounds if row.get("status") == status]
    rounds = sorted(rounds, key=lambda row: int(row.get("round_number") or 0), reverse=True)
    if max_rounds > 0:
        rounds = rounds[:max_rounds]
    return rounds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch historical round details + analysis payloads into local logs."
    )
    parser.add_argument("--round-id", action="append", default=[], help="Specific round id (repeatable).")
    parser.add_argument(
        "--status",
        default="completed",
        choices=["completed", "scoring", "active", "pending", "all"],
        help="Filter for /rounds when --round-id is not set.",
    )
    parser.add_argument("--max-rounds", type=int, default=50, help="How many rounds to fetch.")
    parser.add_argument("--logs-root", default=str(REPO_ROOT / "logs"), help="Output root for logs.")
    parser.add_argument("--refresh", action="store_true", help="Refetch analysis even if file exists.")
    parser.add_argument("--sleep-seconds", type=float, default=0.1, help="Sleep between analysis requests.")
    parser.add_argument("--token", default=None, help="Bearer token override.")
    parser.add_argument("--cookie-token", default=None, help="Cookie token override.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API base URL.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    headers = build_headers(args)
    logs_root = Path(args.logs_root)

    try:
        rounds = select_rounds(
            headers=headers,
            base_url=args.base_url,
            explicit_round_ids=args.round_id,
            status=args.status,
            max_rounds=args.max_rounds,
        )
        if not rounds:
            print("No rounds selected.")
            return 0

        rounds_processed = 0
        analysis_files_fetched = 0

        for item in rounds:
            round_id = str(item["id"])
            details = get_round_details(round_id, headers=headers, base_url=args.base_url)
            rd = round_dir(logs_root, round_id)

            ensure_round_manifest(logs_root, round_id, details)
            write_json(rd / "round-details.json", details)

            run_path = run_log_path(logs_root, round_id)
            append_run_log(
                run_path,
                round_id=round_id,
                run_type="fetch_historical_rounds:start",
                payload={"status": details.get("status"), "round_number": details.get("round_number")},
            )

            seeds_count = int(details.get("seeds_count") or len(details.get("initial_states", [])))
            fetched_this_round = 0

            for seed_index in range(seeds_count):
                target = analysis_seed_path(logs_root, round_id, seed_index)
                if target.exists() and not args.refresh:
                    continue

                try:
                    analysis = get_analysis(
                        round_id=round_id,
                        seed_index=seed_index,
                        headers=headers,
                        base_url=args.base_url,
                    )
                    write_json(target, analysis)
                    fetched_this_round += 1
                    analysis_files_fetched += 1
                    print(
                        f"round={details.get('round_number')} ({round_id}) seed={seed_index} "
                        f"fetched -> {target}"
                    )
                except ApiError as exc:
                    print(
                        f"round={details.get('round_number')} ({round_id}) seed={seed_index} "
                        f"analysis unavailable: {exc}"
                    )
                time.sleep(args.sleep_seconds)

            append_run_log(
                run_path,
                round_id=round_id,
                run_type="fetch_historical_rounds:end",
                payload={"fetched_analysis_files": fetched_this_round, "seeds_count": seeds_count},
            )
            rounds_processed += 1

        print(
            f"Done. rounds_processed={rounds_processed} "
            f"analysis_files_fetched={analysis_files_fetched}"
        )
        return 0
    except ApiError as exc:
        print(f"API error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
