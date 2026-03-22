# Rolling Profile Selection Findings (2026-03-22)

This file records anti-overfit profile selection using rolling holdouts:

- train rounds with `round_number < k`
- test on round `k`
- evaluated for `k = 12..22`

## Command

```bash
. .venv/bin/activate
PYTHONPATH=src python scripts/evaluate_rolling_profile_selection.py \
  --logs-root logs \
  --output-dir outputs/rolling_profile_selection \
  --query-budget 50 \
  --min-test-round-number 12 \
  --strict
```

## Best Stable Profile

Profile id:

- `w_more_cells__l_balanced__p_default`

Rolling metrics (11 holdouts):

- mean round score: `69.125687`
- median round score: `67.831755`
- p25 round score: `64.773841`
- mean seed weighted KL: `0.126582`

## Baseline Comparison

Reference profile:

- `w_default__l_default__p_default`

Reference rolling metrics (11 holdouts):

- mean round score: `64.872381`
- median round score: `65.463566`
- p25 round score: `58.644635`
- mean seed weighted KL: `0.148667`

Absolute gains (best - reference):

- mean round score: `+4.253306`
- median round score: `+2.368188`
- p25 round score: `+6.129206`
- mean seed weighted KL: `-0.022085`

## Selected Production Settings

World config:

- `learning_rate=0.07`
- `epochs=4`
- `samples_per_epoch=20000`
- `max_cells_per_seed=1600`
- `entropy_weight_power=0.8`
- `min_entropy_weight=0.02`
- `probability_floor=1e-4`
- `random_seed=7`

Latent overrides:

- `empirical_prior_strength=0.55`
- `observation_confidence_scale=2.2`
- `repeated_observation_bonus=0.12`
- `max_observed_blend_weight=0.9`
- `dynamic_blend_boost=0.35`

Policy:

- default deterministic three-phase policy (`p_default`)

## Artifacts

- `outputs/rolling_profile_selection/holdout_results.csv`
- `outputs/rolling_profile_selection/profile_summary.csv`
- `outputs/rolling_profile_selection/summary.json`
- `outputs/rolling_profile_selection/run_summary.json`
- `outputs/rolling_profile_selection/best_profile.json`
