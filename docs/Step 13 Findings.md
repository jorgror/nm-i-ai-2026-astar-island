# Step 13 Findings Snapshot (2026-03-22)

This file records Step 13 delivery for leave-one-round-out ablation reporting.

## What Changed

- Added `scripts/evaluate_step13_ablations.py` to run and summarize ablations in one pass.
- Generated unified Step 13 artifacts in:
  - `outputs/step13_ablations/round_scenario_results.csv`
  - `outputs/step13_ablations/scenario_summary.csv`
  - `outputs/step13_ablations/summary.json`
  - `outputs/step13_ablations/step13_summary.md`
  - `outputs/step13_ablations/run_summary.json`

## Evaluated Setups

- no-query baseline: `no_query_prior`
- fixed-query policy: `fixed_query_latent_blend` (center-sweep + latent+blend)
- adaptive-query policy: `adaptive_query_latent_blend` (three-phase + latent+blend)
- no latent vs latent:
  - `adaptive_query_prior` vs `adaptive_query_latent_no_blend`
- no blending vs blending:
  - `adaptive_query_latent_no_blend` vs `adaptive_query_latent_blend`
- feature model vs small CNN:
  - Baseline B vs Baseline C from production LOO seed CSVs

## Metrics Tracked

- weighted KL (mean seed weighted KL)
- round score (mean round score)
- calibration error on dynamic cells (dynamic-cell ECE)
- value per query vs no-query baseline

Dynamic-cell calibration metric:

- Expected accuracy per dynamic cell is `ground_truth[predicted_argmax]`.
- ECE is confidence-binned error over dynamic cells (`entropy(ground_truth) > 1e-6`).

## Results (19 rounds, query budget 50)

Scenario means:

- `no_query_prior`:
  - mean round score: `34.345018`
  - mean weighted KL: `0.392625`
  - dynamic ECE: `0.142670`
  - value/query: `0.000000`
- `fixed_query_latent_blend`:
  - mean round score: `38.095179`
  - mean weighted KL: `0.341923`
  - dynamic ECE: `0.145263`
  - value/query: `0.075003`
- `adaptive_query_prior`:
  - mean round score: `34.345018`
  - mean weighted KL: `0.392625`
  - dynamic ECE: `0.142670`
  - value/query: `0.000000`
- `adaptive_query_latent_no_blend`:
  - mean round score: `38.567688`
  - mean weighted KL: `0.343189`
  - dynamic ECE: `0.142473`
  - value/query: `0.084453`
- `adaptive_query_latent_blend`:
  - mean round score: `43.530868`
  - mean weighted KL: `0.292963`
  - dynamic ECE: `0.122972`
  - value/query: `0.183717`

Key mean round-score deltas:

- adaptive vs fixed policy (latent+blend): `+5.435689`
- latent vs prior (adaptive, no blend): `+4.222670`
- blend vs no blend (adaptive latent): `+4.963180`

Feature vs CNN (from production LOO outputs):

- Baseline B mean round score: `70.933301`
- Baseline C mean round score: `70.286626`
- C - B mean round delta: `-0.646675`

## Step 13 Deliverable Status

- Step 13 summary sheet is delivered via:
  - `outputs/step13_ablations/step13_summary.md`
  - `outputs/step13_ablations/summary.json`
