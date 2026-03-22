# Step 14 Holdout Findings (Train 1-21, Test 22)

This file records a targeted holdout scenario where training is restricted to rounds 1-21 and evaluation is done on round 22.

## Command

```bash
. .venv/bin/activate
PYTHONPATH=src python scripts/evaluate_holdout_train_to_test.py \
  --logs-root logs \
  --output-dir outputs/holdout_train21_test22 \
  --train-max-round-number 21 \
  --test-round-number 22 \
  --query-budget 50 \
  --strict
```

## Results

- `prior_latent_blend`: `45.561777`
- `step14_default_world_latent_blend`: `63.765220`
- `step14_tuned_world_latent_blend`: `69.465134`

Deltas:

- tuned vs default: `+5.699914`
- tuned vs prior: `+23.903356`
- default vs prior: `+18.203442`

## Tuned Stack Used

World model (`BaselineBConfig`):

- `learning_rate=0.07`
- `epochs=4`
- `samples_per_epoch=20000`
- `max_cells_per_seed=1600`
- `entropy_weight_power=0.8`
- `min_entropy_weight=0.02`
- `probability_floor=1e-4`
- `random_seed=7`

Latent config overrides (`RoundLatentConfig`):

- `enable_observation_blend=True`
- `empirical_prior_strength=0.7`
- `observation_confidence_scale=2.4`
- `repeated_observation_bonus=0.1`
- `max_observed_blend_weight=0.9`
- `dynamic_blend_boost=0.3`

Policy config overrides (`DeterministicThreePhasePolicyConfig`):

- `phase1_target=10`
- `phase2_target=25`
- `phase3_target=15`
- `top_windows_per_seed=6`
- `min_center_distance=6.0`
- `default_window=15`

## Artifacts

- `outputs/holdout_train21_test22/scenario_summary.csv`
- `outputs/holdout_train21_test22/seed_scenario_results.csv`
- `outputs/holdout_train21_test22/summary.json`
- `outputs/holdout_train21_test22/run_summary.json`
