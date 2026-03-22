# Step 8 Findings Snapshot (2026-03-22)

This file persists Step 8 implementation status for official-objective training.

## What Changed

- Baseline B and Baseline C now optimize an entropy-weighted surrogate objective.
- Per-cell gradient weight is derived from ground-truth entropy:
  - `weight = max(min_entropy_weight, entropy ** entropy_weight_power)`
- Training reports now include epoch-level objective metrics:
  - mean training weighted KL
  - mean training round-score surrogate (`100 * exp(-3 * weighted_kl)`)
- Script outputs (`summary.json`, `run_summary.json`) include:
  - `mean_training_weighted_kl_by_epoch`
  - `mean_training_round_score_by_epoch`

## Smoke Run Artifacts

Baseline B smoke run:
- `outputs/step8_smoke_b/summary.json`
- `outputs/step8_smoke_b/run_summary.json`

Baseline C smoke run:
- `outputs/step8_smoke_c/summary.json`
- `outputs/step8_smoke_c/run_summary.json`

## Smoke Run Results

Config (both): `epochs=1`, `samples_per_epoch=300`, `max_cells_per_seed=200`.

Baseline B:
- Mean seed score: `50.249270`
- Mean training weighted KL by epoch: `[0.205491]`
- Mean training round score by epoch: `[54.023166]`

Baseline C:
- Mean seed score: `49.735187`
- Mean training weighted KL by epoch: `[0.204257]`
- Mean training round score by epoch: `[54.238576]`

## Full Sweep (8 Runs)

Sweep artifacts:

- Ranked CSV: `outputs/step8_sweep/sweep_results.csv`
- Ranked JSON: `outputs/step8_sweep/sweep_results.json`
- Baseline B run dirs: `outputs/step8_sweep_b/b1..b4`
- Baseline C run dirs: `outputs/step8_sweep_c/c1..c4`

Top ranked runs by held-out mean round score:

1. `B/b4`: `70.933301`
2. `C/c1`: `70.286626`
3. `B/b2`: `70.230980`
4. `C/c2`: `70.208144`

Best Baseline B config (`b4`):

- `learning_rate=0.06`
- `epochs=4`
- `samples_per_epoch=20000`
- `max_cells_per_seed=1000`
- `entropy_weight_power=0.8`
- `min_entropy_weight=0.02`

Best Baseline C config (`c1`):

- `patch_radius=1`
- `learning_rate=0.04`
- `epochs=3`
- `samples_per_epoch=12000`
- `max_cells_per_seed=900`
- `entropy_weight_power=1.0`
- `min_entropy_weight=0.02`

Observation:

- Lower training weighted KL did not always produce the highest held-out round score.
- In this sweep, moderate entropy weighting (`power=0.8..1.0`) beat the more aggressive `power=1.3` settings on held-out score.

## Step 8 Conclusion

- Step 8 deliverable is implemented:
  - training loop aligned with official entropy-weighted objective
  - training metrics reported as weighted KL and round score (not accuracy)
- Recommended current defaults from the full sweep:
  - Baseline B: use `b4` settings
  - Baseline C: use `c1` settings

## Production Pass With New Defaults

Defaults were updated in code and scripts to match the selected configs:

- Baseline B defaults = `b4`
- Baseline C defaults = `c1`

Production artifacts:

- Baseline B: `outputs/step8_production_b/`
- Baseline C: `outputs/step8_production_c/`

Final production metrics:

- Baseline B mean round score: `70.933301`
- Baseline C mean round score: `70.286626`
- C vs B mean round delta: `-0.646675`
