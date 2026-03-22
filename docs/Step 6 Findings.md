
# Step 6 Findings Snapshot (2026-03-22)

This file persists only Step 6 findings (round fingerprints and round archetype clustering).

## Step 6 Deliverables

Initial run output directory:
- `outputs/step6_round_archetypes/`

Tuned run output directory:
- `outputs/step6_round_archetypes_tuned/`

Key artifacts:
- Fingerprint table: `round_fingerprint_table.csv`
- Fingerprints json: `round_fingerprints.json`
- Cluster summary: `clustering_summary.json`
- Scatter plot: `cluster_scatter.svg`
- Elbow plot: `cluster_elbow.svg`
- Interpretation notes: `cluster_interpretation.md`

## Tuned Step 6 Clustering Configuration

- Feature set: `compact`
- Features used:
  - `settlement_survival_rate`
  - `settlement_growth_rate`
  - `port_retention_rate`
  - `port_creation_rate`
  - `ruin_frequency`
  - `average_spread_radius`
  - `takeover_conflict_rate`
  - `owner_change_rate`
  - `owner_consolidation`
  - `dominant_owner_share`
- `k = 4`
- Cluster sizes: `4, 5, 7, 3` (balanced relative to the initial run)

## Tuned Step 6 Archetype Summary

- Cluster 1 (`rounds 6, 11, 14, 17, 18`):
  - Hyper-growth, port-strong regime.
- Cluster 2 (`rounds 2, 4, 8, 9, 13, 15, 20`):
  - Conflict-heavy competitive growth.
- Cluster 0 (`rounds 5, 7, 12, 16`):
  - Controlled growth, lower churn.
- Cluster 3 (`rounds 3, 10, 19`):
  - Collapse-heavy / winner-take-most.
