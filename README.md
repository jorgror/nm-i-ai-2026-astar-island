# Astar Island Local Tooling (Step 1)

This repository now includes a local Python package to cover step 1 from `MASTERPLAN.md`:

- parse round and analysis JSON into canonical models
- compute official entropy-weighted KL score locally
- validate and serialize prediction submissions
- visualize initial maps and prediction/ground-truth tensors
- reproduce per-seed and per-round historical scores

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
