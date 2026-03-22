# Step 12 Findings Snapshot (2026-03-22)

This file records Step 12 delivery for submission hygiene and safe round submission.

## What Changed

- Extended `src/astar_island/submission.py` with round-level safety helpers:
  - `build_safe_round_submission(...)`
  - `missing_seed_indices(...)`
  - `SeedSubmissionPlan` metadata for fallback/source/error tracking
- Added a one-button live pipeline script:
  - `scripts/safe_submit_round.py`

## Deliverable

Implemented a safe submission pipeline that:

- keeps a valid fallback prediction for every seed
- enforces probability floors before submission
- verifies per-cell normalization and tensor shape locally
- submits all seeds and verifies completeness against:
  - `GET /astar-island/my-predictions/{round_id}`
  - `GET /astar-island/my-rounds`
- optionally auto-resubmits checkpoint snapshots during an active round

## Safe Submission Flow

1. Select active round (or explicit `--round-id`).
2. Fetch round details.
3. Optionally gather live observations (`/simulate`) with deterministic 3-phase policy.
4. Build seed predictions (`latent` or `prior` model).
5. Build safe per-seed submission plan:
   - use model output when valid
   - otherwise use mechanics-first fallback prior
6. Submit every seed with retry logic.
7. Verify completeness and resubmit missing seeds.
8. Persist checkpoint summary JSON in `outputs/step12_safe_submission/<round_id>/`.

## Test Coverage Added

Updated `tests/test_submission.py` with Step-12 coverage:

- fallback for missing model prediction
- fallback for invalid model prediction
- missing-seed detection helper

## Notes

- Live API execution is environment/token dependent and was not run in this snapshot.
- The pipeline is designed to fail closed if fallback predictions are invalid.
