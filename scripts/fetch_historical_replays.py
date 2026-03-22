#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib import error, request


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "https://api.ainm.no"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


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


def fetch_replay(
    round_id: str,
    seed_index: int,
    headers: dict[str, str],
    *,
    base_url: str,
    max_retries: int,
) -> dict[str, Any] | None:
    url = f"{base_url.rstrip('/')}/astar-island/replay"
    payload = json.dumps({"round_id": round_id, "seed_index": seed_index}).encode("utf-8")

    for attempt in range(max_retries):
        req = request.Request(url=url, data=payload, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=30.0) as response:
                if response.status == 200:
                    return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            if exc.code == 429:
                backoff = 2**attempt
                logging.warning(
                    "Rate limit hit (429) for round=%s seed=%s, retrying in %ss",
                    round_id,
                    seed_index,
                    backoff,
                )
                time.sleep(backoff)
                continue
            if exc.code == 404:
                logging.info("Replay missing for round=%s seed=%s (404)", round_id, seed_index)
                return None
            if exc.code in (401, 403):
                logging.error("Auth error (%s). Check token/cookie.", exc.code)
                return None

            body = ""
            try:
                body = exc.read().decode("utf-8")
            except Exception:
                pass
            logging.error("HTTP error %s for round=%s seed=%s: %s", exc.code, round_id, seed_index, body)
            return None
        except Exception as exc:  # noqa: BLE001
            logging.error("Unexpected replay fetch error for round=%s seed=%s: %s", round_id, seed_index, exc)
            return None

    logging.warning("Failed replay fetch after retries for round=%s seed=%s", round_id, seed_index)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch replay payloads for completed rounds in logs/")
    parser.add_argument("--logs-root", default=str(REPO_ROOT / "logs"), help="Logs root folder.")
    parser.add_argument("--refresh", action="store_true", help="Refetch replay files even if they exist.")
    parser.add_argument("--sleep-seconds", type=float, default=1.0, help="Sleep between replay requests.")
    parser.add_argument("--max-retries", type=int, default=5, help="Retries on 429.")
    parser.add_argument("--token", default=None, help="Bearer token override.")
    parser.add_argument("--cookie-token", default=None, help="Cookie token override.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API base URL.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    headers = build_headers(args)
    logs_root = Path(args.logs_root)

    if not logs_root.exists():
        logging.error("Logs directory not found: %s", logs_root)
        return 1

    processed_rounds = 0
    fetched_files = 0

    for round_dir in sorted(logs_root.iterdir()):
        if not round_dir.is_dir():
            continue

        manifest_path = round_dir / "round_manifest.json"
        if not manifest_path.exists():
            continue

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logging.warning("Skipping malformed manifest %s: %s", manifest_path, exc)
            continue

        round_id = str(manifest.get("round_id") or round_dir.name)
        status = manifest.get("status")
        if status != "completed":
            logging.info("Skipping round %s (status=%s)", round_id, status)
            continue

        details = manifest.get("round_details", {})
        seeds_count = int(
            details.get("seeds_count")
            or manifest.get("seeds_count")
            or 0
        )
        if seeds_count <= 0:
            logging.info("Skipping round %s (seeds_count missing)", round_id)
            continue

        logging.info("Processing round %s (%s seeds)", round_id, seeds_count)
        processed_rounds += 1

        for seed_index in range(seeds_count):
            replay_path = round_dir / f"replay-seed-{seed_index}.json"
            if replay_path.exists() and not args.refresh:
                continue

            replay_data = fetch_replay(
                round_id=round_id,
                seed_index=seed_index,
                headers=headers,
                base_url=args.base_url,
                max_retries=args.max_retries,
            )
            if replay_data is None:
                continue

            replay_path.write_text(
                json.dumps(replay_data, separators=(",", ":"), ensure_ascii=True),
                encoding="utf-8",
            )
            logging.info("Saved %s", replay_path)
            fetched_files += 1
            time.sleep(args.sleep_seconds)

    logging.info(
        "Done. rounds_processed=%s replay_files_fetched=%s",
        processed_rounds,
        fetched_files,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
