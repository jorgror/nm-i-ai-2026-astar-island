# Astar Island Local Tooling (Steps 1-13)

This repository includes local Python tooling for the early execution phases from `MASTERPLAN.md`:

- parse round and analysis JSON into canonical models
- compute official entropy-weighted KL score locally
- validate and serialize prediction submissions
- visualize initial maps and prediction/ground-truth tensors
- reproduce per-seed and per-round historical scores
- build round-centric datasets and leave-one-round-out splits
- run replay-backed offline round emulation
- generate mechanics-first priors and dynamic-cell importance maps
- compute round fingerprints + clustering artifacts for round archetype analysis
- train with entropy-weighted objective and report epoch-level weighted KL / round score
- infer a shared round latent from query observations and condition all seed predictions on it
- run a deterministic three-phase query scheduler for offline/live probing
- blend empirical observations with model predictions on observed cells
- run a safe one-button submission pipeline with fallback + validation + completeness checks
- run leave-one-round-out ablation summaries with dynamic-cell calibration and value/query reporting

## Setup (`.venv`)

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

If `pip` is not available in your current shell, activate `.venv` and run only the commands that work in your environment.

## Historical Score Reproduction

You can fetch the required historical files directly:

```bash
. .venv/bin/activate
python scripts/fetch_historical_rounds.py --status completed --max-rounds 5
```

This creates files like:

- `logs/<round_id>/round-details.json`
- `logs/<round_id>/analysis-seed-0.json` ... `analysis-seed-4.json`

Optional replay fetch (if endpoint access is available for your team/token):

```bash
. .venv/bin/activate
python scripts/fetch_historical_replays.py
```

Expected input files:

- round detail JSON from `GET /astar-island/rounds/{round_id}`
- analysis JSON files from `GET /astar-island/analysis/{round_id}/{seed_index}`
- name analysis files so they contain seed index, e.g.:
  - `seed_0.json`
  - `seed_1.json`
  - ...
  - `seed_4.json`

Run:

```bash
. .venv/bin/activate
python scripts/reproduce_historical_round.py \
  --round-json logs/<round_id>/round-details.json \
  --analysis-dir logs/<round_id> \
  --my-round-json data/my_rounds.json \
  --output-dir outputs/reproduce_round_7 \
  --strict
```

Outputs:

- printed per-seed summary with weighted KL and score diffs
- `outputs/.../summary.json`
- per-seed SVGs:
  - `initial_grid.svg`
  - `ground_truth_argmax.svg`
  - `ground_truth_entropy.svg`
  - `prediction_argmax.svg`

## Module Overview

- `src/astar_island/parsing.py`: canonical JSON parsing
- `src/astar_island/scoring.py`: official score implementation
- `src/astar_island/submission.py`: validator + serializer + probability floor
- `src/astar_island/visualization.py`: SVG visualizers
- `src/astar_island/round_data.py`: round-centric dataset loader + leave-one-round-out splits
- `src/astar_island/offline_emulator.py`: replay-backed `run_offline_round(policy, model, round_id)`
- `src/astar_island/priors.py`: mechanics-first baseline priors + dynamic importance map
- `src/astar_island/importance.py`: feature-based per-seed dynamic-cell importance heatmaps
- `src/astar_island/archetypes.py`: round fingerprint extraction, k-means clustering, PCA scatter + elbow plots
- `src/astar_island/baseline_b.py`: feature-based non-neural baseline (multinomial logistic regression) + LOO evaluation
- `src/astar_island/baseline_c.py`: small spatial baseline (local patch softmax) + LOO evaluation
- `src/astar_island/round_latent.py`: round-latent encoder + latent-conditioned model + empirical blending (`predict(round_state, seed_initial_state, seed_index)`)
- `src/astar_island/query_policy.py`: deterministic three-phase query scheduler for offline/live probing

## Step 6: Round Archetype Analysis

Generate the step-6 deliverables (round fingerprint table + clustering plots):

```bash
. .venv/bin/activate
PYTHONPATH=src python scripts/analyze_round_archetypes.py \
  --logs-root logs \
  --output-dir outputs/step6_round_archetypes \
  --feature-set compact \
  --k 4 \
  --max-k 8
```

Available feature sets: `default`, `compact`, `dynamics`.  
You can also override with an explicit list:

```bash
. .venv/bin/activate
PYTHONPATH=src python scripts/analyze_round_archetypes.py \
  --features settlement_survival_rate,settlement_growth_rate,port_retention_rate,port_creation_rate,ruin_frequency,average_spread_radius,takeover_conflict_rate,owner_change_rate,owner_consolidation,dominant_owner_share \
  --k 4 \
  --max-k 8
```

Outputs:

- `outputs/step6_round_archetypes/round_fingerprint_table.csv`
- `outputs/step6_round_archetypes/round_fingerprints.json`
- `outputs/step6_round_archetypes/clustering_summary.json`
- `outputs/step6_round_archetypes/cluster_scatter.svg`
- `outputs/step6_round_archetypes/cluster_elbow.svg`
- `outputs/step6_round_archetypes/report_summary.json`

## Step 7: Baseline B (Feature Model)

Run leave-one-round-out evaluation for the feature-based non-neural baseline and compare against mechanics-first prior baseline A:

```bash
. .venv/bin/activate
PYTHONPATH=src python scripts/evaluate_baseline_b.py \
  --logs-root logs \
  --output-dir outputs/step7_baseline_b \
  --learning-rate 0.05 \
  --epochs 5 \
  --entropy-weight-power 1.0 \
  --min-entropy-weight 0.02 \
  --samples-per-epoch 25000 \
  --max-cells-per-seed 1000
```

Outputs:

- `outputs/step7_baseline_b/loo_seed_results.csv`
- `outputs/step7_baseline_b/loo_round_results.csv`
- `outputs/step7_baseline_b/summary.json`
- `outputs/step7_baseline_b/run_summary.json`

## Step 7: Baseline C (Small Spatial Model)

Run leave-one-round-out evaluation for the small spatial model (3x3 local neighborhood by default):

```bash
. .venv/bin/activate
PYTHONPATH=src python scripts/evaluate_baseline_c.py \
  --logs-root logs \
  --output-dir outputs/step7_baseline_c \
  --patch-radius 1 \
  --learning-rate 0.04 \
  --epochs 5 \
  --entropy-weight-power 1.0 \
  --min-entropy-weight 0.02 \
  --samples-per-epoch 25000 \
  --max-cells-per-seed 1000
```

Optional direct C-vs-B comparison using Baseline B result CSVs:

```bash
. .venv/bin/activate
PYTHONPATH=src python scripts/evaluate_baseline_c.py \
  --baseline-b-seed-csv outputs/step7_baseline_b/loo_seed_results.csv \
  --baseline-b-round-csv outputs/step7_baseline_b/loo_round_results.csv
```

Outputs:

- `outputs/step7_baseline_c/loo_seed_results.csv`
- `outputs/step7_baseline_c/loo_round_results.csv`
- `outputs/step7_baseline_c/summary.json`
- `outputs/step7_baseline_c/run_summary.json`

## Step 8: Official-Objective Training Metrics

Baseline B/C training now applies entropy-weighted gradients (high-entropy cells get higher influence)
and reports epoch-level:

- `mean_training_weighted_kl_by_epoch`
- `mean_training_round_score_by_epoch`

Current default settings are tuned from the step-8 sweep:

- Baseline B defaults: `b4` (`lr=0.06`, `epochs=4`, `samples_per_epoch=20000`, `entropy_weight_power=0.8`)
- Baseline C defaults: `c1` (`patch_radius=1`, `lr=0.04`, `epochs=3`, `samples_per_epoch=12000`)

## Step 9: Round-Latent Evaluation

Run step-9 evaluation (no-query prior vs queried prior vs queried latent):

```bash
. .venv/bin/activate
PYTHONPATH=src python scripts/evaluate_round_latent.py \
  --logs-root logs \
  --output-dir outputs/step9_round_latent_eval \
  --query-budget 50 \
  --strict
```

Outputs:

- `outputs/step9_round_latent_eval/seed_results.csv`
- `outputs/step9_round_latent_eval/round_results.csv`
- `outputs/step9_round_latent_eval/summary.json`
- `outputs/step9_round_latent_eval/run_summary.json`

## Step 10: Query Policy Backtest

Run step-10 policy backtest (three-phase scheduler vs simple center-sweep baseline):

```bash
. .venv/bin/activate
PYTHONPATH=src python scripts/evaluate_query_policy.py \
  --logs-root logs \
  --output-dir outputs/step10_query_policy_eval \
  --query-budget 50 \
  --model latent \
  --strict
```

Outputs:

- `outputs/step10_query_policy_eval/seed_results.csv`
- `outputs/step10_query_policy_eval/round_results.csv`
- `outputs/step10_query_policy_eval/summary.json`
- `outputs/step10_query_policy_eval/run_summary.json`

## Step 11: Empirical Blending Evaluation

Run step-11 evaluation (latent model without blending vs with blending):

```bash
. .venv/bin/activate
PYTHONPATH=src python scripts/evaluate_blending.py \
  --logs-root logs \
  --output-dir outputs/step11_blending_eval \
  --query-budget 50 \
  --strict
```

Outputs:

- `outputs/step11_blending_eval/seed_results.csv`
- `outputs/step11_blending_eval/round_results.csv`
- `outputs/step11_blending_eval/summary.json`
- `outputs/step11_blending_eval/run_summary.json`

## Step 12: Safe Submission Pipeline

Run the one-command safe submit pipeline for the active round:

```bash
. .venv/bin/activate
PYTHONPATH=src python scripts/safe_submit_round.py \
  --model latent \
  --query-budget 50 \
  --probability-floor 0.01
```

What it guarantees:

- always constructs valid predictions for every seed index (`0..seeds_count-1`)
- falls back to mechanics-first priors if model output is missing/invalid
- applies probability floor and renormalization before submit
- validates shape and per-cell sums locally before API calls
- verifies all seeds are submitted via `/my-predictions/{round_id}` and `/my-rounds`

Optional checkpoint mode (auto-resubmit during active round):

```bash
. .venv/bin/activate
PYTHONPATH=src python scripts/safe_submit_round.py \
  --checkpoint-seconds 300 \
  --max-checkpoints 6
```

## Step 13: Ablation Summary Sheet

Run the consolidated ablation pass:

```bash
. .venv/bin/activate
PYTHONPATH=src python scripts/evaluate_step13_ablations.py \
  --logs-root logs \
  --output-dir outputs/step13_ablations \
  --query-budget 50 \
  --strict
```

Outputs:

- `outputs/step13_ablations/round_scenario_results.csv`
- `outputs/step13_ablations/scenario_summary.csv`
- `outputs/step13_ablations/summary.json`
- `outputs/step13_ablations/step13_summary.md`
- `outputs/step13_ablations/run_summary.json`

Quick smoke commands:

```bash
. .venv/bin/activate
PYTHONPATH=src python scripts/evaluate_baseline_b.py \
  --logs-root logs \
  --output-dir outputs/step8_smoke_b \
  --epochs 1 \
  --samples-per-epoch 300 \
  --max-cells-per-seed 200
```

```bash
. .venv/bin/activate
PYTHONPATH=src python scripts/evaluate_baseline_c.py \
  --logs-root logs \
  --output-dir outputs/step8_smoke_c \
  --epochs 1 \
  --samples-per-epoch 300 \
  --max-cells-per-seed 200
```

## Tests

```bash
. .venv/bin/activate
PYTHONPATH=src python -m pytest -q
```

If `pytest` is unavailable, use:

```bash
. .venv/bin/activate
PYTHONPATH=src python -m unittest discover -s tests -v
```
