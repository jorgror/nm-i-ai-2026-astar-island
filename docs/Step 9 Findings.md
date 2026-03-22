# Step 9 Findings Snapshot (2026-03-22)

This file records Step 9 implementation and first evaluation pass for round-latent inference.

## What Changed

- Added `src/astar_island/round_latent.py` with:
  - `RoundLatentEncoder`: infers a compact latent vector from live query observations.
  - `RoundLatentConditionalModel`: predicts per-seed tensors conditioned on shared round latent.
- Added `scripts/evaluate_round_latent.py` for side-by-side evaluation:
  - no-query prior baseline
  - queried prior baseline (same query schedule)
  - queried latent-conditioned model
- Added `tests/test_round_latent.py`:
  - ruin-heavy observation signal increases latent ruin bias
  - cross-seed conditioning is active (seed-1 observations shift seed-0 predictions)
  - model runs through `run_offline_round(...)` end-to-end

## Interface Deliverable

Implemented model interface compatible with offline emulator:

- `predict(round_state, seed_initial_state, seed_index) -> HxWx6`

This allows observations from one seed to influence predictions for other seeds in the same round.

## Evaluation Run

Command:

```bash
. .venv/bin/activate
PYTHONPATH=src python scripts/evaluate_round_latent.py \
  --logs-root logs \
  --output-dir outputs/step9_round_latent_eval \
  --query-budget 50 \
  --strict
```

Dataset and coverage:

- rounds: `19`
- seeds: `95`
- replay coverage: `95/95` seeds (complete)
- query schedule: deterministic 3-phase, fixed budget `50`

## Results (Mean Over 19 Rounds)

- prior (no-query): `34.345018`
- prior (with queries): `34.345018`
- latent (with same queries): `38.588990`
- latent gain vs queried prior: `+4.243972`
- latent gain vs no-query prior: `+4.243972`

Round-level consistency:

- positive delta rounds: `19`
- negative delta rounds: `0`
- median round delta (latent vs queried prior): `+2.400242`

Seed-level consistency:

- positive delta seeds: `95/95`
- median seed delta (latent vs queried prior): `+2.416278`

Largest and smallest round gains (latent vs queried prior):

- smallest 3: `+1.0931`, `+1.5450`, `+1.7718`
- largest 3: `+9.2933`, `+11.3283`, `+14.4050`

Interpretation:

- As expected, queried prior == no-query prior because it does not ingest observations.
- The latent layer converts query observations into round-level adjustments that improve all seeds.
- First-pass heuristic latent conditioning is materially positive across all held-out historical rounds.

## Artifacts

- `outputs/step9_round_latent_eval/seed_results.csv`
- `outputs/step9_round_latent_eval/round_results.csv`
- `outputs/step9_round_latent_eval/summary.json`
- `outputs/step9_round_latent_eval/run_summary.json`

## Test Status

- `33 passed` (`PYTHONPATH=src python -m pytest -q`)
