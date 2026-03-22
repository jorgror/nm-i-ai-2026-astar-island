# Step 7 Findings Snapshot (2026-03-22)

This file persists Step 7 findings for:
- Baseline A (mechanics-first prior) as reference
- Baseline B (feature-based multinomial logistic regression)
- Baseline C (small spatial local-patch softmax model)

## Evaluation Protocol

- Dataset: `logs/` with 19 rounds and 5 seeds per round (95 seed evaluations total).
- Split strategy: leave-one-round-out (LOO).
- Metric focus: official round score / seed score from entropy-weighted KL scoring.

## Baseline A Reference

- Baseline A implementation: `src/astar_island/priors.py`
- Mean seed/round score in Step 7 runs: `34.345018`

## Baseline B Findings

Run outputs:
- `outputs/step7_baseline_b/summary.json`
- `outputs/step7_baseline_b/loo_seed_results.csv`
- `outputs/step7_baseline_b/loo_round_results.csv`

Config used:
- `learning_rate=0.05`
- `epochs=3`
- `samples_per_epoch=12000`
- `max_cells_per_seed=900`
- `random_seed=7`

Result:
- Mean seed/round score Baseline B: `65.675248`
- Gain vs Baseline A: `+31.330230`

## Baseline C Findings

Run outputs:
- `outputs/step7_baseline_c/summary.json`
- `outputs/step7_baseline_c/loo_seed_results.csv`
- `outputs/step7_baseline_c/loo_round_results.csv`

Best config found in sweep:
- `patch_radius=1` (3x3 local patch)
- `learning_rate=0.04`
- `epochs=3`
- `samples_per_epoch=12000`
- `max_cells_per_seed=900`
- `random_seed=7`

Result:
- Mean seed/round score Baseline C: `66.047448`
- Gain vs Baseline A: `+31.702430`
- Gain vs Baseline B: `+0.372200`

## Step 7 Conclusion

- Step 7 is complete: both required baselines (B and C) are implemented and evaluated with LOO.
- Baseline C currently edges Baseline B on held-out mean round score.
