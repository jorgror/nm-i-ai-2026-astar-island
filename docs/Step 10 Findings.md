# Step 10 Findings Snapshot (2026-03-22)

This file records Step 10 delivery for a deterministic 3-phase query scheduler.

## What Changed

- Added `src/astar_island/query_policy.py` with:
  - `DeterministicThreePhasePolicyConfig`
  - `DeterministicThreePhaseQueryPolicy`
- Scheduler behavior:
  - Phase 1: broad calibration (`~30%` of budget), cycles seeds and probes settlement/coast/high-importance windows.
  - Phase 2: repeated motif probes (`~40%` of budget), repeats top windows per seed.
  - Phase 3: exploit (`~30%` of budget), uses additional high-importance windows.
- Added policy backtest script: `scripts/evaluate_query_policy.py`.
- Added scheduler tests: `tests/test_query_policy.py`.

## Deliverable

Implemented deterministic scheduler compatible with the offline emulator:

- `DeterministicThreePhaseQueryPolicy.next_query(state) -> ViewportQuery | None`
- deterministic, bounded, backtestable

## Validation

Policy tests cover:

- query budget and viewport bounds
- phase-1 per-seed coverage
- phase-2 repeated windows
- phase-3 per-seed spread
- determinism and scaled-budget behavior

## Backtest Runs

### Command (query budget 50)

```bash
. .venv/bin/activate
PYTHONPATH=src python scripts/evaluate_query_policy.py \
  --logs-root logs \
  --output-dir outputs/step10_query_policy_eval \
  --query-budget 50 \
  --model latent \
  --strict
```

Result:

- mean round score (three-phase): `38.567688`
- mean round score (center-sweep baseline): `38.567688`
- delta: `+0.000000`

### Command (query budget 15)

```bash
. .venv/bin/activate
PYTHONPATH=src python scripts/evaluate_query_policy.py \
  --logs-root logs \
  --output-dir outputs/step10_query_policy_eval_q15 \
  --query-budget 15 \
  --model latent \
  --strict
```

Result:

- mean round score (three-phase): `38.313716`
- mean round score (center-sweep baseline): `38.313716`
- delta: `+0.000000`

## Interpretation

- Step-10 deliverable (deterministic scheduler + backtest harness) is complete.
- In current offline setup, policy differences did not change score versus center-sweep.
- Most likely reason: current latent encoder is dominated by global aggregated query statistics, which reduces sensitivity to where queries are taken.
- This is expected to change once Step 11 blending and richer per-location latent features are added.

## Artifacts

- `outputs/step10_query_policy_eval/seed_results.csv`
- `outputs/step10_query_policy_eval/round_results.csv`
- `outputs/step10_query_policy_eval/summary.json`
- `outputs/step10_query_policy_eval/run_summary.json`
- `outputs/step10_query_policy_eval_q15/seed_results.csv`
- `outputs/step10_query_policy_eval_q15/round_results.csv`
- `outputs/step10_query_policy_eval_q15/summary.json`
- `outputs/step10_query_policy_eval_q15/run_summary.json`

## Test Status

- `39 passed` (`PYTHONPATH=src python -m pytest -q`)
