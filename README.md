# Astar Island Local Tooling (Steps 1-6)

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
