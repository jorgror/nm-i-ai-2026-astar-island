# Step 14 Findings Snapshot (2026-03-22)

This file records Step 14 delivery for a learned world-model upgrade.

## What Changed

- Added `src/astar_island/world_model.py`:
  - trains a Baseline-B world model from historical rounds
  - exposes `BaselineBWorldModelPredictor` for latent conditioning
- Added `scripts/evaluate_step14_world_model.py`:
  - leave-one-round-out evaluation
  - compares `prior-latent-blend` vs `world-latent-blend`
- Updated `scripts/safe_submit_round.py`:
  - new `--model latent-b` mode (default)
  - trains world model once at startup and uses it as latent base predictor
  - falls back to `latent` if world-model training fails

## Evaluation Setup

- Dataset: 22 completed rounds from `logs/`
- Split strategy: leave-one-round-out
- Query policy: deterministic three-phase policy
- Query budget: 50
- Latent blend: enabled in both compared setups

Command used:

```bash
. .venv/bin/activate
PYTHONPATH=src python scripts/evaluate_step14_world_model.py \
  --logs-root logs \
  --output-dir outputs/step14_world_model_eval \
  --query-budget 50 \
  --strict
```

## Results (22 rounds)

- Mean round score (`prior-latent-blend`): `44.715368`
- Mean round score (`world-latent-blend`): `65.860930`
- Mean round delta: `+21.145562`
- Median round delta: `+22.767436`
- Positive round deltas: `21 / 22`
- Negative round deltas: `1 / 22`

Weighted KL:

- Mean seed weighted KL (`prior-latent-blend`): `0.283058`
- Mean seed weighted KL (`world-latent-blend`): `0.142653`

## Artifacts

- `outputs/step14_world_model_eval/seed_results.csv`
- `outputs/step14_world_model_eval/round_results.csv`
- `outputs/step14_world_model_eval/summary.json`
- `outputs/step14_world_model_eval/run_summary.json`
