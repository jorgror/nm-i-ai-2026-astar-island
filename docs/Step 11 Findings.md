# Step 11 Findings Snapshot (2026-03-22)

This file records Step 11 delivery for empirical observation blending.

## What Changed

- Extended `src/astar_island/round_latent.py` with per-cell blending:
  - empirical posterior from observed query cells
  - confidence-weighted blend with model posterior
  - optional neighbor smoothing of empirical posteriors
  - probability floor and renormalization retained
- Added `scripts/evaluate_blending.py` to compare:
  - latent model with blending disabled
  - latent model with blending enabled
- Added blending-focused unit tests in `tests/test_round_latent.py`.

## Blending Rule (Delivered)

For each seed and cell:

- If unobserved: keep model posterior.
- If observed: estimate empirical class probabilities from query observations.
- Blend:
  - `final = (1 - w) * model + w * empirical`
  - `w` grows with number of observations and repeated observations
  - `w` is boosted on high-importance cells and dampened on static terrain
- Optional smoothing:
  - empirical posterior can be mixed with neighboring observed-cell empirical posteriors.

This is the required per-cell blending rule with confidence weights.

## Validation

Added/updated tests verify:

- observed cells shift toward empirical classes
- repeated observations increase blend strength
- unobserved cells remain model-driven when latent shifts are disabled

Test status:

- `42 passed` (`PYTHONPATH=src python -m pytest -q`)

## Evaluation Run (Query Budget 50)

Command:

```bash
. .venv/bin/activate
PYTHONPATH=src python scripts/evaluate_blending.py \
  --logs-root logs \
  --output-dir outputs/step11_blending_eval \
  --query-budget 50 \
  --strict
```

Results (19 rounds, 95 seeds):

- mean round score, latent no blend: `38.567688`
- mean round score, latent + blend: `43.530868`
- mean round delta: `+4.963180`
- positive round deltas: `19/19`
- median round delta: `+4.873050`

Round delta spread (blend vs no blend):

- smallest 3: `+0.4098`, `+1.0767`, `+1.1875`
- largest 3: `+10.5617`, `+12.4537`, `+14.4022`

## Additional Run (Query Budget 15)

Command:

```bash
. .venv/bin/activate
PYTHONPATH=src python scripts/evaluate_blending.py \
  --logs-root logs \
  --output-dir outputs/step11_blending_eval_q15 \
  --query-budget 15 \
  --strict
```

Results:

- mean round score, latent no blend: `38.313716`
- mean round score, latent + blend: `42.256105`
- mean round delta: `+3.942389`
- positive round deltas: `19/19`

## Artifacts

- `outputs/step11_blending_eval/seed_results.csv`
- `outputs/step11_blending_eval/round_results.csv`
- `outputs/step11_blending_eval/summary.json`
- `outputs/step11_blending_eval/run_summary.json`
- `outputs/step11_blending_eval_q15/seed_results.csv`
- `outputs/step11_blending_eval_q15/round_results.csv`
- `outputs/step11_blending_eval_q15/summary.json`
- `outputs/step11_blending_eval_q15/run_summary.json`

## Notes

- `scripts/evaluate_round_latent.py` now explicitly disables blending to keep Step 9 evaluation isolated and comparable over time.
