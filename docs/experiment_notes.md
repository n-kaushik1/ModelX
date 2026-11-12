# Experiment notes (for the final report)

All numbers are local estimates, not leaderboard scores. Environment: Python 3.11.9 on Windows (16 cores) with
the exact leaderboard versions (numpy 1.26.4, pandas 2.3.1, scikit-learn 1.7.1, lightgbm 4.6.0, xgboost 3.1.1,
catboost 1.2.10), as created by `scripts/setup_env.*`. Raw outputs of every run are in `experiments/logs/`, and
the scripts are in `experiments/`.

Protocol: three stratified 80/20 splits per problem (random_state 42, 1, 2), with the leaderboard metric on the
20% part. Each model is saved and reloaded with joblib before predicting, as the leaderboard does, and the
metadata is in the leaderboard format (`problems.metadata_for`).

## Baseline vs improved framework

| Problem (metric) | baseline | improved v1 | **improved v2 (submitted)** |
|---|---|---|---|
| P1 ROC-AUC ↑ | **0.8476** | 0.8463 | 0.8458 |
| P2 log loss ↓ | 0.5116 | 0.5106 | **0.5099** |
| P3 normalized cost ↓ | 0.6837 | 0.6857 | **0.6824** |

Per-split v2 − baseline: P1 −0.0018 / −0.0024 / −0.0012; P2 −0.0023 / −0.0012 / −0.0016 (better on all 3);
P3 −0.003 / −0.020 / +0.019.

- v1 = baseline + XGBoost (+ conditional CatBoost), with the time budget renormalised across families.
- v2 = v1 + absolute time shares (an added family no longer reduces LightGBM's search time) + a separate
  random stream per family (the search is reproducible and does not depend on which libraries are installed).

## Supporting experiments

**Model families (5-fold OOF on all development data, default parameters)**

| | P1 AUC | P2 log loss | P3 best norm. cost |
|---|---|---|---|
| LightGBM | 0.8493 | 0.5119 | 0.7421 |
| XGBoost | 0.8503 | 0.5133 | 0.7373 |
| CatBoost | 0.8462 | 0.5218 | 0.7524 |
| LightGBM + XGBoost | 0.8507 | 0.5114 | 0.7295 |

CatBoost took about 45 s per fit (vs. under 1 s for LightGBM) and was worse on all three problems. It is now used
only when a categorical has more than 30 levels, where its ordered target statistics should help. None of the
public problems has such a column.

**Threshold for P3 (10 repeated 80/20 splits, LightGBM + Platt calibration)**: holdout normalized cost

| strategy | mean (sd) |
|---|---|
| learned on OOF (bootstrap median + smoothed cost curve), *current* | 0.747 (0.036) |
| Bayes threshold c_fp/(c_fp+c_fn) = 1/11 | 0.749 (0.042) |
| logit-average of the two | 0.748 (0.039) |
| hindsight-best threshold on the holdout (not attainable) | 0.721 |

The learned threshold is as good as the cost-implied value and lands near it (0.09–0.13). The method was kept.

**Feature selection (shadow-feature / Boruta-style, nested inside the CV folds)**: neutral on P1 (AUC
0.8514 vs 0.8512) and harmful on P2 (log loss 0.5156 vs 0.5109) and P3 (cost 0.770 vs 0.729), because it also
drops useful features. Rejected; feature selection is left to tree regularisation.

## Observations

- P1: logistic regression only reaches about 0.72 AUC vs about 0.85 for boosting, which points to strong
  non-linearity or interactions. LightGBM and XGBoost are tied there, so the choice between them on OOF scores
  is mostly noise.
- P2: blending LightGBM + XGBoost (+ HGB) with Platt calibration gave a small but consistent log-loss gain.
- P3: ExtraTrees is competitive (often in the blend). The decision threshold is learned at about 0.09–0.11,
  close to the Bayes value 1/11.

## Version 3 (`submission/sub2/`): complexity ladder, logit blending, GAM candidate

Environment for all v3 numbers: Python 3.11.18 on a 4-vCPU Linux VM (Claude Code cloud) with the exact leaderboard
versions. Scripts are in `experiments/sub2/`, raw outputs in `experiments/logs/exp{1,2,3}_p{1,2,3}.log` and
`eval_sub2_p{1,2,3}.log`.

### What the data told us (5-fold OOF on all development rows, early stopping per fold)

| model | P1 AUC ↑ | P2 log loss ↓ | P3 log loss ↓ | P3 best norm. cost ↓ |
|---|---|---|---|---|
| LightGBM, depth 1 (stumps = additive model) | **0.8545** | 0.5126 | 0.2010 | 0.739 |
| LightGBM, depth 2 | 0.8534 | **0.5104** | 0.1991 | 0.723 |
| LightGBM, depth 3 | 0.8535 | 0.5110 | 0.1991 | 0.724 |
| LightGBM, 15 leaves (old default region) | 0.8518 | 0.5117 | 0.1991 | 0.730 |
| LightGBM, 31 leaves | 0.8500 | 0.5126 | 0.2000 | 0.732 |
| XGBoost, depth 3 | 0.8533 | 0.5106 | 0.1987 | 0.720 |
| CatBoost, depth 3 / 4 | 0.8533 / 0.8530 | 0.5105 / 0.5111 | **0.1977 / 0.1977** | 0.723 / **0.719** |
| spline-logistic GAM (C = 1 / 0.3 / 0.1) | 0.8458 | 0.5155 | 0.2025 | 0.742 |
| Optuna TPE, 40 trials, LightGBM depth 1-5 | 0.8547 (depth 1) | 0.5104 (depth 2) | 0.1983 (depth 4) | – |

- All three problems are **mostly additive with weak low-order interactions**: depth-1 boosting is the best
  single model on P1 and depth 2-4 wins on P2/P3. Deep trees (31 leaves) lose 0.004 AUC on P1. The old search
  space (4-48 leaves, 100-700 trees, no early stopping) could not express this well.
- A 40-trial Optuna search ends exactly where the hand-made depth ladder ends, so a cheap ladder plus a short
  random refinement captures the available gain.
- Class weights hurt (P3 cost 0.747 at weight 2, early stopping collapses at weight 4+); kept off for cost /
  log-loss objectives.
- Denoising features for near-duplicate sensor pairs (|corr| > 0.9: mean/difference of z-scores) were neutral or
  slightly negative on all problems: rejected.
- Three-seed bagging of one model: +0.0005 AUC on P1, +0.0008 on P3; kept for the final refit only.
- Blends (mean of logits): P1 0.8549-0.8552, P2 0.5093-0.5097, P3 log loss 0.1973-0.1978, cost 0.711-0.718.
  A nested LR stack on the logits was not better than greedy selection, so greedy selection was kept.

### Changes in v3 relative to v2

1. **Complexity ladder + early stopping.** Every boosting family first fits one configuration per tree depth
   (1, 2, 3, 4, 6; CatBoost 3, 4, 6) with the number of trees chosen by early stopping inside each fold, then
   spends the rest of its time share on random refinement around the best depth. The tree count is no longer a
   random search dimension.
2. **Unbiased re-scoring.** The best configuration of each family is re-run on fresh folds with the tree count
   fixed (no early-stopping optimism) before blending, calibration and threshold selection.
3. **Logit-space greedy blending.** Members are averaged on the logit scale (Caruana selection with replacement,
   up to 15 picks). For threshold metrics the selection criterion is log loss instead of the noisy
   best-threshold cost; the threshold is still learned on the blended, calibrated OOF probabilities.
4. **New candidate: spline-logistic GAM** (cubic B-splines of the standardised numerics + one-hot categoricals)
   replaces the plain logistic regression when there are numeric columns. It is a strong additive baseline and a
   diverse blend member; CatBoost is now always a candidate (with a capped time share), HGB only when LightGBM is
   missing, RandomForest dropped (ExtraTrees kept for diversity).
5. **Refit scaling.** The number of trees is scaled up (×(1 + 0.5/(k-1))) when refitting on all rows; three
   seeds per boosting member below 40k rows.
6. Degenerate cases (< 4 minority rows, all candidates failing) fall back to the prior safely; `predict_proba`
   can never return NaN.

### Local results v3 vs v2 (3 stratified 80/20 splits, seeds 42 / 1 / 2, leaderboard metric on the 20 %)

| Problem (metric) | baseline | improved v2 | **sub2 = v3** | per-split v3 − v2 |
|---|---|---|---|---|
| P1 ROC-AUC ↑ | 0.8476 | 0.8458 | **0.8489** | +0.0043 / +0.0040 / +0.0010 (better on all 3) |
| P2 log loss ↓ | 0.5116 | 0.5099 | **0.5088** | −0.0016 / −0.0017 / −0.0001 (better on all 3) |
| P3 normalized cost ↓ | 0.6837 | **0.6824** | 0.6863 | +0.002 / +0.011 / −0.001 (noise: split sd ≈ 0.036) |

Per split v3: P1 0.8523 / 0.8441 / 0.8501; P2 0.4992 / 0.5091 / 0.5182; P3 0.6967 / 0.6722 / 0.6900 (thresholds
0.095 / 0.100 / 0.107). Fit time 260-280 s per split on 2 pinned cores. The v2 numbers are from
`eval_improved_v2_pinned_env.log` (16 cores); because tuning is time-boxed the comparison slightly favours v2.

Blends chosen by v3: P1 LightGBM + CatBoost (+ XGBoost); P2 LightGBM + CatBoost + GAM (+ XGBoost, ExtraTrees);
P3 CatBoost alone or CatBoost + LightGBM + XGBoost (+ GAM, ExtraTrees). The GAM received weight on every P2 split.

**P3 threshold transfer test** (`experiments/sub2/exp4_p3.py`, `exp4_p3.log`): 100 random half splits of the
5-fold OOF predictions; Platt calibration and threshold chosen on one half, normalized cost measured on the other.

| model | log loss | learned threshold | Bayes 1/11 on calibrated p | oracle |
|---|---|---|---|---|
| CatBoost depth 4 | 0.1977 | 0.734 | 0.732 | 0.716 |
| XGBoost depth 3 | 0.1987 | 0.733 | 0.730 | 0.716 |
| ExtraTrees min_leaf 10 | 0.2011 | 0.732 | 0.732 | 0.718 |
| blend CatBoost 4 + XGBoost 3 + CatBoost 3 (greedy on log loss) | 0.1973 | **0.726** | 0.729 | 0.710 |
| blend + ExtraTrees | 0.1976 | 0.723 | 0.720 | 0.709 |

Blends transfer about 0.01 better than any single model, the three threshold rules are equivalent (within 0.003),
and the log-loss ranking of models agrees with their cost ranking, which supports selecting blend members for the
cost problem by log loss. The learned threshold was kept.


## Version 4 (`submission/sub2/`, final): measured structure as model constraints

Scripts `experiments/sub2/exp5.py` (learning curves, additive shapes, interaction detection, constraints) and
`exp6.py` (combined constraints, XGBoost / CatBoost variants, blends); logs `exp5_p*.log`, `exp6_p*.log`.

### What the deeper analysis showed (5-fold OOF, all development rows)

- **Learning curves** (25 / 50 / 75 / 100 % of the training folds): P1 AUC 0.841 → 0.850 → 0.853 → 0.8545,
  P2 log loss 0.519 → 0.513 → 0.512 → 0.510, P3 still improving at 100 %. The models are still variance-limited,
  so anything that removes variance without adding bias should pay.
- **Additive shapes** (partial dependence of a depth-1 model): the strongest effects are hinges and U-shapes
  (P1 `cycle_time`, 46 % of gain, U-shaped; `process_temperature` high at both ends; P3 `traffic_change_1h`
  U-shaped, `throughput_utilization` a sharp hinge). Many weaker effects are cleanly monotone (6-9 features per
  problem with |Spearman| > 0.9 along the partial dependence). Several columns carry no gain at all (ambient
  humidity, batch weight, storage days, shift, machine type; P2 legal structure; P3 vendor, technology, weekday).
- **Interactions** are sparse and concentrated: the top pairs from a depth-2 model (`material_purity_index` ×
  `cycle_time` on P1, `operating_margin` × `debt_service_ratio` on P2, `throughput_utilization` ×
  `traffic_change_1h` on P3) carry most of the pairwise gain.

### Constrained boosting (LightGBM, depth 2 unless stated)

| configuration | P1 AUC ↑ | P2 log loss ↓ | P3 log loss ↓ | P3 best cost ↓ |
|---|---|---|---|---|
| unconstrained depth 2 (v3 best rung) | 0.8549 | 0.5103 | 0.1977 | 0.705 |
| additive only (no interactions) | 0.8542 | 0.5120 | 0.2003 | 0.732 |
| additive + top-8 measured pairs | 0.8557 | 0.5099 | 0.1971 | 0.704 |
| + monotone directions, top-4 / 8 / 12 / 16 / 24 pairs | **0.8560** / 0.8555 / 0.8558 / 0.8557 / 0.8556 | 0.5095 / 0.5097 / **0.5092** / 0.5093 / 0.5094 | 0.1978 / 0.1973 / 0.1969 / 0.1968 / **0.1966** | 0.706 / 0.700 / 0.703 / 0.707 / 0.700 |
| XGBoost depth 2/3 with the same constraints | 0.8542 / 0.8538 | 0.5101 / 0.5104 | 0.1991 / 0.1983 | 0.721 / 0.717 |
| CatBoost depth 4 + monotone | 0.8527 | 0.5113 | 0.1977 | 0.716 |
| blend v3-style (LightGBM d2 + XGBoost d3 + CatBoost d4) | 0.8544 | 0.5098 | 0.1977 | 0.714 |
| blend with the constrained members | **0.8556** | **0.5093** | **0.1970** | 0.708 |

A single constrained LightGBM now matches or beats the whole v3 blend on every problem, and blends built around
it gain a little more. The number of pairs that helps most differs by problem (4 on P1, 12-16 on P2, 24 on P3),
which is why the framework searches it.

### Changes in v4

1. `_detect_structure` (nested inside every fold and refit, cached per training subset): a quick depth-2 model
   ranks feature pairs by shared split gain; a quick depth-1 model gives each numeric feature's additive shape,
   and features whose shape is nearly monotone (|Spearman| > 0.9, at most 2 sign changes) get a direction.
2. New hyper-parameters `inter_k` (how many measured pairs the trees may combine, 0 = unconstrained) and `mono`
   (impose the measured directions). The LightGBM / XGBoost ladders get structured rungs (depth 2 with 8 and 24
   pairs, depth 3 with 16), and the random refinement samples `inter_k` in {4 … 32} and `mono` on/off.
   HistGradientBoosting (fallback) uses `interaction_cst` / `monotonic_cst`; CatBoost only monotone directions.
3. For each boosting family the best structured **and** the best unconstrained configuration are re-scored on
   fresh folds and offered to the greedy blend (they make different errors).
4. Time shares: LightGBM 0.38, XGBoost 0.22, CatBoost 0.22, GAM 0.07, ExtraTrees 0.06 of `time_budget` (240 s).

Nothing in this is dataset-specific: the constraints are measured from the training rows the model is fitted on,
so they adapt to each private problem (and vanish when a problem has no monotone effects or no sparse
interaction structure).

### Local results v4 (3 stratified 80/20 splits, seeds 42 / 1 / 2)

| Problem (metric) | baseline | v2 (improved) | v3 | **v4 (sub2, final)** | v4 per split |
|---|---|---|---|---|---|
| P1 ROC-AUC ↑ | 0.8476 | 0.8458 | 0.8489 | **0.8492** | 0.8526 / 0.8440 / 0.8510 |
| P2 log loss ↓ | 0.5116 | 0.5099 | 0.5088 | **0.5087** | 0.4996 / 0.5087 / 0.5177 |
| P3 normalized cost ↓ | 0.6837 | 0.6824 | 0.6863 | **0.6802** | 0.6794 / 0.6717 / 0.6894 (thresholds 0.107 / 0.097 / 0.108) |

v4 vs v2 per split: P1 +0.0045 / +0.0039 / +0.0018 (better on all 3); P2 −0.0012 / −0.0020 / −0.0005 (better on
all 3); P3 −0.015 / +0.010 / −0.002. Fit time 300-340 s per split on 2 pinned cores. On the holdout splits
(2-3k rows) v4 and v3 are within noise of each other; the case for v4 rests on the 5-fold full-data comparison
above (+0.001 AUC, −0.001 log loss, −0.0007 P3 log loss for the structured members and their blends), which is
measured on 10-15k rows and is consistent across all three problems.

Members chosen by v4: P1 structured LightGBM + CatBoost (+ structured XGBoost, GAM); P2 structured LightGBM + GAM
+ XGBoost (+ CatBoost, ExtraTrees); P3 structured LightGBM / XGBoost + CatBoost. The structured LightGBM received
weight on 8 of 9 splits.


### Final deep pass (`experiments/sub2/exp7.py`, logs `exp7_p*.log`): what did *not* help

Base B = structured LightGBM (depth 2, 8 pairs, monotone), 5-fold OOF, nested structure detection.

| variant | P1 AUC | P2 log loss | P3 log loss | verdict |
|---|---|---|---|---|
| B | 0.8555 | 0.5096 | 0.1977 | reference |
| B, 5 seeds | 0.8556 | 0.5097 | 0.1976 | noise |
| one-hot instead of native categoricals | 0.8546 | 0.5104 | 0.1994 | worse everywhere |
| max_bin 1023 / 63 | 0.8552 / 0.8539 | 0.5104 / 0.5103 | 0.1976 / 0.1980 | worse / noise |
| extra_trees | 0.8530 | 0.5097 | 0.1968 | inconsistent |
| drop zero-gain features (nested) | 0.8556 | 0.5099 | 0.1976 | noise |
| min_child 50, lr 0.015, path_smooth 10 | ±0.0003 | ±0.0006 | ±0.0002 | noise |
| no subsampling (colsample = subsample = 1) | 0.8544 | 0.5106 | 0.1983 | worse |
| piecewise-linear trees (`linear_tree`), depth 1 / 2 | 0.8551 / 0.8552 | 0.5117 / 0.5099 | 0.2006 / 0.1976 | not better |
| blends of B with these variants | 0.8557-0.8559 | 0.5093 | 0.1973 | +0.0003 at most |

Every remaining lever moves the OOF metrics by less than the noise of the public holdout (about 0.008 AUC and
0.036 normalized cost on a 2-3k-row split), and none is consistent across the three problems, so none was
adopted: chasing them would be fitting the development data. v4 is the final framework.

**Private-problem stress test** (`experiments/sub2/stress_test.py`, log `stress_test_sub2_v4.log`): tiny data
(300 rows), 1 % positives with a 1:20 cost objective, 150 features, 300-level categoricals with string labels and
`positive_label="yes"`, 70k rows (tuning subsample path), empty metadata, and an all-categorical table all fit,
pickle, reload and predict correctly with unseen categories, an all-NaN column and an extra column at prediction
time.

## Version 5 (final): repeated-CV screening and the hinge-basis GAM

`experiments/sub2/exp8.py` (logs `exp8_p*.log`) re-screened candidates with **repeated 5-fold CV over three fold
seeds**, structure detection nested. The spread between fold seeds alone is about 0.0005 AUC (P1) and 0.0005-0.0007
log loss (P2/P3), so single-fold-seed differences below that are noise.

| candidate (vs. structured LightGBM reference) | P1 AUC | P2 log loss | P3 log loss | verdict |
|---|---|---|---|---|
| reference B (depth 2, 8 pairs, monotone) | 0.8540 | 0.5099 | 0.1978 | |
| `inter_k` 16 / 4 | 0.8541 / 0.8540 | 0.5096 / 0.5095 | 0.1975 / 0.1990 | searched already |
| `num_leaves` 3 | 0.8542 | 0.5100 | 0.1974 | noise |
| depth 3, 6 leaves, 16 pairs | 0.8540 | 0.5095 | 0.1977 | noise |
| DART boosting | 0.8515 | 0.5141 | 0.2010 | worse, 25x slower |
| min_sum_hessian 5, cat_l2 30 | ±0.0001 | ±0.0000 | ±0.0001 | noise |
| Optuna, 45 trials on the structured space, fold seed 0 | 0.8552 on seed 0 → **0.8543** on seeds 0-2 | 0.5082 → **0.5093** | 0.1961 → **0.1973** | gain does not transfer: tuning-fold overfit |
| calibration none / Platt / beta / isotonic (P2, P3) | – | 0.5099 / 0.5098 / 0.5098 / 0.5116 | 0.1978 / 0.1978 / 0.1978 / 0.1988 | Platt kept (nested check) |
| **hinge-basis GAM** (linear + max(x−knot,0) at 7 quantile knots), alone | 0.8492 | 0.5110 | 0.2004 | vs cubic-spline GAM 0.8459 / 0.5137 / 0.2030 |
| blend B + hinge GAM (3:1) | 0.8544 | **0.5089** (−0.0009 on every fold seed) | 0.1977 | adopted as a candidate |

The Optuna row is the important negative result: a tuned configuration that looks 0.0017 log loss better on the
folds it was tuned on is only 0.0005 better on fresh folds. Chasing CV gains below the fold-seed spread is
fitting the folds. The hinge GAM is the one change that is consistent: as a single model it beats the spline GAM
on all three problems (through the framework: P1 AUC 0.849 vs 0.846, P2 log loss 0.5102 vs 0.5137, P3 0.2001 vs
0.2030, `experiments/sub2/gamcheck.py`), and the greedy blend gives it weight where it helps (P2).

### Change in v5

- `SplineLogit` uses a piecewise-linear hinge basis by default (`basis="hinge"`, 7 quantile knots per numeric
  column with more than 10 distinct values); cubic splines remain available (`basis="spline"`) and are sampled in
  the refinement. GAM ladder: C = 0.05 and 0.2 with hinges.

Everything else is v4. The boosting members, blending, calibration and threshold logic are unchanged.


## Submission v4 = framework version 6 (in progress)

Details, tables and raw logs: `submission/v4/V4_NOTES.md`, `experiments/v4/`. Environment: Windows, 16 cores,
the exact leaderboard versions. Protocol: 5-fold CV repeated over 3 fold seeds (structure detection nested),
plus repeated outer CV for the final-predictor question.

- **No feature-engineering signal.** An OpenFE-style screen (residual booster on base-model OOF logits) of about
  7,000 candidates (pairwise ratios / products / sums / differences, group-relative values, category crosses,
  count encodings) found nothing above the noise floor; on P3 a pure-noise column scored as well as the best
  candidates. `revenue_per_employee` is exactly `annual_revenue / employee_count`, sensor pairs are noisy copies,
  and several effects are step functions, which is why trees win.
- **Single models have plateaued.** LightGBM variants (lr 0.01, subsample 0.5, 3 leaves, 3 seeds, ...), EBM-style
  stage-wise GA2M, and GAM variants all land within about ±0.0005 of the structured LightGBM reference.
- **k-fold bagging beats the refit.** Averaging the 5 inner fold models beat v5's refit on all rows (trees x1.125,
  3 seeds) on all three problems: P1 AUC 0.8551 vs 0.8542, P2 log loss 0.5100 vs 0.5103, P3 log loss 0.1981 vs
  0.1983 (cost at 1/11: 0.720 vs 0.724).
- **Library ensembles.** Caruana selection over a library of configurations beats the reference model by
  +0.0002 AUC (P1), -0.0017 log loss (P2) and -0.0005 log loss (P3) on held-out rows; a logistic stack is better
  still on P2 but worse on P1 and P3, so v4 chooses between them by nested CV.
- **Monotone CatBoost** (depth 3, lr 0.03) is the most-weighted blend member on P1 and P3.

Outer 5-fold CV of the whole framework (fold seed 0, paired): v4 beats v5/sub2 on all three problems (P1 AUC
0.85495 vs 0.85459, better on all 5 folds; P2 log loss 0.50824 vs 0.50857; P3 normalized cost 0.7180 vs 0.7192,
log loss 0.19667 vs 0.19696). Details in `submission/v5/V5_NOTES.md`.



## Submission v5 = framework version 7 (final)

Details, tables and raw logs: `submission/v5/V5_NOTES.md`, `experiments/v5/`. Protocol: paired outer 5-fold CV
of the whole framework (fit on 4/5, joblib save/reload, predict the raw 1/5), two fold seeds; offline
re-combination of the saved fold libraries for combiner / calibration / threshold questions.

- **Combiner: bagged ensemble selection** (greedy Caruana selection on 25 bootstrap resamples of the OOF rows,
  weights averaged). On the outer-CV holdout (two fold seeds, mean) it was first or second on all three problems
  (P1 AUC 0.85460, P2 log loss 0.50847, P3 log loss 0.19729), where every alternative was last somewhere: a single
  Caruana pass (0.85458 / 0.50865 / 0.19732), a logistic stack (0.85464 / 0.50863 / 0.19742, with negative member
  weights), equal weights + Platt (0.85447 / **0.50831** / 0.19745: best on P2, last on P1 and P3). Choosing the combiner by
  nested CV did not transfer (inner-CV gaps of 0.001 AUC vanished on the holdout).
- **Bag design**: 5-fold bagging (models on 80 %) beats 10-fold (90 %) on P1 (0.85502 vs 0.85469); extra
  partitions (5-fold × 2 or × 3) add < 0.0001. Kept at 5.
- **CatBoost variants** (depth 2, Langevin boosting, l2 20): none consistently better than the monotone depth-3
  rung. **P3 threshold rules** (learned, Bayes 1/11, bootstrap mean / median): all within 0.006 normalized cost,
  far below the 0.015 noise of a pooled 15k-row estimate; the learned rule was kept.
- **Robustness for the private problems**: identifier-named columns are only dropped when they are mostly unique
  (a `machine_id` with a few levels is a categorical); date / time columns become numeric features (days, month,
  day of week, hour) instead of being dropped; timedeltas become seconds.

Final paired outer CV, mean of two fold seeds: v5 P1 AUC 0.85461, P2 log loss 0.50851, P3 normalized cost 0.7182
(log loss 0.19728) vs v4 0.85460 / 0.50857 / 0.7198 (0.19729). v5 ties v4 on the public problems and is kept for
its robustness. The per-fold P3 cost of the same model ranges from 0.65 to 0.78, so leaderboard differences of a
few hundredths on P3 (and of a few thousandths AUC / log loss on P1 / P2) are luck, not signal.


## Submission v6 = framework version 8 (final)

Details and raw logs: `submission/v6/V6_NOTES.md`, `experiments/v6/`, `experiments/v5/logs/exp_drop_*.log`.

- **Noisy copies.** Each public problem has columns that are a copy of another column plus noise, in the same
  units (`temperature_sensor_2`, `material_density_estimate`, `utilization_3m_average`,
  `account_balance_last_month`, `utilization_15m_average`). A covariance decomposition shows the original is
  noise-free (var = cov) and the copy carries 4–7 % extra variance. Dropping the copies improved a reference
  booster on all three problems and every fold seed (P1 AUC 0.85402 → 0.85433, P2 log loss 0.50985 → 0.50918, P3
  0.19775 → 0.19725); dropping the originals instead cost up to 0.017 AUC. The framework now detects copies from
  the covariance structure alone (no target, no names): exact on the public data and on 31 of 32 synthetic
  problems, never dropping an original. Twin readings (both noisy) are averaged instead.
- **Rejected: pruning derived columns by CV** (e.g. `revenue_per_employee` = revenue / employees). Chosen with
  hindsight it looked like −0.0008 log loss on P2; nested in the folds it made P2 worse (0.50875 → 0.51006).
- **Threshold.** On 30 synthetic cost problems the cost-derived Bayes threshold c_fp/(c_fp+c_fn) on calibrated
  probabilities beat the empirical OOF threshold by 0.0017 normalized cost (worse on 19 of 30 for the empirical
  rule); on the public P3 outer-CV predictions it was better for every framework version. The framework uses it
  unless the empirical threshold wins out of sample in ≥ 95 % of bootstrap rounds.
- **Generalisation suite.** 16 synthetic problems in the style of the public data, each scored on 20k fresh rows
  against the true probabilities. Mean regret v5 → v6: AUC 0.01484 → 0.01449, log loss 0.00900 → 0.00875, cost
  0.0357 → 0.0328.

Final paired outer CV (mean of two fold seeds): v6 P1 AUC 0.85474, P2 log loss 0.50828, P3 normalized cost 0.7183
vs v5 0.85461 / 0.50851 / 0.7182.


## Submission v7 = framework version 9 (final)

Leaderboard check of the v5 build: P1 0.8617, P2 0.5034, P3 0.6800 (leaders 0.8621 / 0.5023 / 0.6636). Details:
`submission/v7/V7_NOTES.md`, `experiments/v7/`.

- **Honest model zoo.** 42 candidates on two 5-fold partitions per problem, each fold model early-stopped on an
  inner split of its own training part; ensemble value measured on held-out halves (40 random splits).
- **Pair-term GAM.** The framework's hinge GAM plus the products of the top 2 measured numeric pairs is the best
  single P2 model (log loss 0.50848 vs 0.51009 without pairs and 0.51040 for the best tree) and improves the
  ensemble by 0.0008. The gain is from the product terms (standardising the design alone is worse). P2's top pair,
  `operating_margin × debt_service_ratio`, acts as a bilinear interaction that trees approximate poorly.
- **Learning rate 0.01** for the structured LightGBM: best single P1 model (AUC 0.85397 vs 0.85330 at 0.03) and
  +0.00035 AUC in the ensemble; lr 0.005 is worse.
- **Rejected:** linear-tree LightGBM (helps P1 slightly, raises P3 decision cost by 0.003 and doubled the regret
  on an extremely imbalanced synthetic problem), class weighting, EBM-style models, extra CatBoost variants,
  random forests, a GAM-then-boost hybrid, and "use every model" (worse than a focused library).

Final paired outer CV (mean of two fold seeds): v7 P1 AUC 0.85481, P2 log loss 0.50752, P3 normalized cost 0.7145
vs v6 0.85474 / 0.50828 / 0.7183 and v5 0.85461 / 0.50851 / 0.7182.


## Submission v8 = framework version 10 (final)

Details: `submission/v8/V8_NOTES.md`. A diagnostic GAM with switchable bases showed what the GAM still missed on P1
/ P3: not steps or AND-rules (both over-fit), but **category × numeric interactions** (P3: `throughput_utilization
× node_type`, `firmware_age_months × region_type`; P1: `cycle_time × operator_experience_band`). A GAM with a
separate slope per category level for the measured pairs, plus numeric products, reached P3 log loss 0.1982 (plain
GAM 0.2004, trees 0.1972) and helped P3 in the ensemble study (log loss −0.0004, cost −0.003). In the whole
framework: P3 log loss 0.19707 vs 0.19722 (both fold seeds better), ties on P1 / P2 / P3 cost; on the synthetic
suite the log-loss regret falls from 0.00879 to 0.00824 (the numeric × category problem: 0.0164 → 0.0137).
Rejected in the same round: XGBoost at lr 0.01, LightGBM at lr 0.005, averaging whole framework versions.


## Submission v11 = framework version 13 (final)

Details: `submission/v10/V10_NOTES.md` (sparse-GAM study), `submission/v11/V11_NOTES.md`.

- **Shapes are piecewise linear with a few kinks.** An L1-penalised GAM on hinges at 15 interior quantiles per
  column (trend filtering) beats the smooth L2 GAM by 0.0034 AUC on P1; step bases hurt even under L1.
- **Interactions are kinked too.** Adding `max(a − k, 0)·b`-type terms for the measured pairs (and
  category-specific hinges) makes the sparse GAM the best single P2 model (0.5075 vs 0.5099 for the best trees)
  and the best single P3 model at C 0.05 (0.1970 vs 0.1972).
- **Hybrid for P1.** On P1 the trees beat the GAM only on `cycle_time`'s very steep U-walls; a booster fitted on
  the sparse GAM's residual adds +0.0003 AUC to the ensemble.
- Whole framework, paired outer CV (2 fold seeds): v11 P1 AUC 0.85503, P2 log loss 0.50659, P3 cost 0.7098 /
  log loss 0.19654 vs v8 0.85484 / 0.50757 / 0.7143 / 0.19707; synthetic log-loss regret 0.00824 → 0.00746.
- Rejected: steps, AND-rules, hinge × hinge, tail knots, squared walls, relaxed lasso, > 6 kinked pairs.


## Submission v12 = framework version 14 (final)

Details: `submission/v12/V12_NOTES.md`. A second GAM-then-boost hybrid, built on the strongest sparse GAM (6 kinked
pairs, C 0.05), is the best single model on P2 and P3; in the whole framework it improves P2 (0.50636 vs 0.50659
for v11, both fold seeds) and P3 cost (0.7087 vs 0.7098), and v12 has the lowest synthetic regret of all versions
for every objective type. Rejected in this round: refitting GAM members on all rows (better alone, tie in the
framework), FAST interaction ranking (worse pairs than the booster-gain ranking), value-spaced / dense knots,
hybrid booster variants (depth 3, lr 0.02, CatBoost residual), RuleFit (much worse).
