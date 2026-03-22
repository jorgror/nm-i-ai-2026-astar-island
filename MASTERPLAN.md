1. Reproduce the game locally before doing any modeling

Build a local package that can:

parse round data into a canonical format,
compute the official score exactly,
validate and serialize submissions,
visualize initial maps, prediction tensors, and post-round analysis.

This sounds basic, but it is step zero. The official score uses entropy-weighted KL, static low-entropy cells barely matter, zero probabilities are dangerous, and missing a seed gives 0 for that seed. Your local evaluator should match that exactly.

Deliverable: a script that can take one historical round and reproduce your historical per-seed and per-round scores exactly.

2. Organize the data by round, not by seed

Create one record per round containing:

the 5 initial seed grids,
the 5 initial settlement layouts,
the 5 ground-truth tensors from /analysis,
any replay payloads you captured,
any metadata you can derive.

Do not split train/validation by seed. Split by round only. Since the hidden parameters are shared across all 5 seeds in a round, seed-level splitting will leak the round dynamics into validation and make your offline results look much better than reality.

Deliverable: leave-one-round-out validation loader.

3. Build an offline “live round emulator”

This is probably the highest-value engineering task.

You want a local harness that mimics the live loop:

reveal the initial states for 5 seeds,
let your policy choose up to 50 viewport queries,
return viewport observations from stored replay data if you have them,
force the model to output 5 full probability tensors,
score them against historical ground truth.

If your replay logs are good enough, this becomes your testbed for query policy and online latent inference. If replay coverage is partial, still use it to emulate the parts you do have; even a rough harness is much better than evaluating only on final-state prediction.

Deliverable: run_offline_round(policy, model, round_id).

4. Hard-code the mechanics you already know

Do not make the model relearn obvious rules from 20 rounds.

Encode these as priors or constraints:

mountains are static,
ocean/plains/empty all map to prediction class 0,
forests are mostly static,
the main dynamic classes are settlement, port, and ruin,
simulation runs for 50 years,
each live observation is only a viewport, max 15×15, under a total 50-query budget.

This should immediately give you a strong fallback predictor and a dynamic mask of cells likely to matter.

Deliverable: a baseline prior tensor generator from initial map only.

5. Create a “dynamic-cell importance map”

Because the score weights high-entropy cells more, you should explicitly predict where uncertainty lives.

I would create a model or heuristic heatmap for likely dynamic cells using features like:

distance to initial settlements,
whether the initial settlement has a port,
coastal proximity,
forest adjacency,
mountain barriers and chokepoints,
local settlement density,
land connectivity / fjord structure.

This map will drive both prediction and query selection.

Deliverable: per-seed 40×40 “importance” heatmap.

6. Do exploratory analysis to identify round archetypes

Before training any serious model, quantify how rounds differ.

For each historical round, compute summaries like:

settlement survival rate,
port retention / creation rate,
ruin frequency,
forest reclaim frequency,
average spread radius,
takeover/conflict rate,
relationship between settlement stats and survival.

The simulation responses expose settlement internals like population, food, wealth, defense, port status, alive, and owner_id in the queried viewport. Those are much closer to the latent rules than terrain alone.

The goal is to discover a few coarse round modes, for example:

aggressive growth,
fragile/collapse-heavy,
trade/port-dominant,
forest-reclaim-heavy.

With only 20 rounds, a handful of interpretable archetypes can outperform a flexible but undertrained latent model.

Deliverable: round fingerprint table + clustering plots.

7. Start with three baselines, in this order
   Baseline A: mechanics-first fallback

Mostly deterministic prior from initial grid:

mountains → mountain with high confidence,
forests → forest with high confidence,
class-0 terrain → mostly class 0,
near settlements/coasts → spread some mass to settlement/port/ruin.

This is your safety net.

Baseline B: motif frequency model

Build a feature-based per-cell model from historical rounds using local context windows and hand-crafted features. I would start with something like:

multinomial logistic regression,
gradient-boosted trees,
or a tiny MLP.
Baseline C: small spatial model

Only after A and B are stable, try a small CNN/U-Net-like model over the initial map plus engineered channels.

With only 20 rounds, I would not start with a large end-to-end deep model. The dataset is too small at the round level, and overfitting will be subtle.

Deliverable: one robust non-neural baseline and one small spatial baseline.

8. Train to the official objective, not generic accuracy

Do not optimize argmax class accuracy. It is the wrong metric.

Train with a loss that mirrors the score:

entropy-weighted per-cell cross-entropy / KL surrogate,
stronger focus on high-entropy cells,
calibration over sharpness,
probability floor before normalization.

Because the official score is entropy-weighted KL, a model that is slightly blurrier but calibrated will beat a sharp model that puts near-zero mass on real outcomes. The official docs explicitly warn against zero probabilities and recommend a floor before renormalization.

Deliverable: training loop reporting weighted KL and round score, not just accuracy.

9. Build a round-latent inference layer

This is the key model idea.

Given a small number of observations from the active round, infer a compact latent vector summarizing the hidden rules for that round. Then condition all 5 seed predictions on that latent.

Inputs to the latent encoder should include:

queried grid cells,
settlement stats from those queries,
changes in port / ruin / survival patterns,
which motifs were probed,
repeated samples from the same window if available.

The point is to let information from seed 1 improve predictions on seeds 0, 2, 3, and 4, because the hidden rules are shared inside the round.

Deliverable: model interface like predict(round_state, seed_initial_state) -> HxWx6.

10. Build the query policy in three phases

I would not let the first version be fully learned. Start with a hand-designed policy, then optimize it offline.

Phase 1: broad calibration

Use the first 10–15 queries to probe each seed at least a little:

settlement-dense areas,
coast/port zones,
chokepoints,
areas where the model expects high entropy.
Phase 2: repeated probes on informative motifs

Spend the middle 20–25 queries on windows that help infer round behavior:

same motif across different seeds,
repeated windows to estimate local stochastic marginals,
areas where settlement stats reveal growth vs collapse.
Phase 3: exploit uncertainty

Use the final 10–15 queries where:

model uncertainty is high,
model disagreement is high,
the dynamic-cell importance map is high.

Default to 15×15 unless you discover a reason not to. The cost is per query, not per cell, and the official API allows 5–15 for width and height.

Deliverable: a deterministic query scheduler you can backtest.

11. Blend empirical observations with model predictions

For cells you have queried multiple times, you can estimate local class probabilities directly from observations. For unqueried cells, rely on the conditioned model.

So your final tensor should be a blend:

empirical local posterior where queried often,
model-based prior/posterior elsewhere,
smoothed with neighboring context,
floored and renormalized.

This is important because live queries are already samples from the stochastic process you are trying to predict. If you have repeated evidence about a window, use it.

Deliverable: per-cell blending rule with confidence weights.

12. Build a strong “submission hygiene” layer

This is boring but worth real score.

Your live system should always:

keep a valid fallback prediction for all 5 seeds,
auto-submit checkpoints during the round,
enforce probability floors,
verify sums to 1,
verify all 5 seeds are submitted.

Since missing a seed gives 0, and zero-probability classes can catastrophically hurt KL, reliability matters almost as much as modeling.

Deliverable: one-button safe submission pipeline.

13. Evaluate everything with leave-one-round-out ablations

For each held-out round, compare:

no-query baseline,
fixed-query policy,
adaptive query policy,
no latent model vs latent model,
no blending vs blending,
feature model vs small CNN.

Track:

weighted KL,
round score,
calibration error on dynamic cells,
value per query.

With only 20 rounds, you need disciplined ablations. Otherwise you will end up optimizing noise.

Deliverable: one summary sheet showing what actually moves held-out round score.

14. Only after that, consider a learned world model

This is a stretch goal, not phase 1.

Possible later directions:

small simulator emulator,
latent-variable sequence model using replay frames,
query policy learned by imitation or bandit optimization,
amortized Bayesian inference over round parameters.

But I would only go there after the simpler stack is strong, because with 20 rounds the risk is spending a lot of time building a beautiful model that is less reliable than a careful hybrid baseline.

What I would prioritize first

If I were running the team, I’d prioritize in this order:

First: steps 1–4
You need local scoring, clean round-based data, offline emulator, and mechanics-first priors.

Second: steps 5–8
That gets you a real baseline that is aligned to the score.

Third: steps 9–13
That is where the competitive edge likely comes from: round-latent inference, query policy, and robust blending.

Last: step 14
Only if the earlier stack is already stable.

The most likely first competitive system is:

mechanics-first fallback,
feature-based per-cell predictor trained with entropy-weighted loss,
round-level latent inferred from query history,
deterministic 3-phase query policy,
empirical/model blending,
strong submission hygiene.

That is much more realistic than trying to solve the full hidden simulator from scratch with only 20 historical rounds.

I can turn this into a concrete project backlog with owners, data schemas, and milestone order next.
