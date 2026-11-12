# Submission v11 (framework version 13): status, changes and evidence

> **Status: final framework.** `submission.zip` in this folder is the build to upload (see §5).

v11 = v8 + the **sparse kinked-interaction GAM** (v10, `submission/v10/V10_NOTES.md` has the full study) + a
**GAM-then-boost hybrid** family. Versions 9 and 10 were intermediate steps of the same line of work.

## 1. The path from v8 to v11 (honest zoo study, then the whole framework)

| step | finding | ensemble value in the zoo study |
|---|---|---|
| v9: L1 sparse hinge GAM | shapes are piecewise linear with a few sharp kinks (L1 on 15 hinges per column beats the smooth L2 GAM by 0.0034 AUC on P1; steps hurt) | P1 +0.00012, P2 −0.00017 |
| v10: kinked interactions | `max(a − k, 0)·b` terms for measured pairs and category-specific hinges: best single P2 model (0.50746), best GAM on P3 | P2 −0.0010, P3 −0.0002 log loss |
| v10c: 6 kinked pairs at C 0.05 | best single P3 model of any kind (0.19703) | P3 −0.0002 log loss, −0.004 cost |
| v11: GAM-then-boost hybrid | the sparse GAM's logits are the starting score of a small booster (lr 0.01, depth 2), which then models only what the GAM misses — on P1, the steep walls of `cycle_time`'s U-shape | P1 +0.00026 AUC |

Diagnosis behind the hybrid: a regression tree on the logit difference "best trees − best GAM" on P1 splits only on
`cycle_time`; beyond the GAM's outer knots the U-shape's walls are steeper than a linear extrapolation. Tail knots
and squared walls inside the GAM over-fit; letting a booster model the residual works.

Rejected along the way: step / AND-rule / hinge × hinge bases, tail knots, squared walls, a relaxed lasso, more
than 6 kinked pairs, XGBoost at lr 0.01, LightGBM at lr 0.005, averaging whole framework versions, linear-tree
LightGBM, class weighting.

## 2. Whole-framework comparison (paired outer 5-fold CV, mean of fold seeds 0 and 1)

| | P1 AUC ↑ | P2 log loss ↓ | P3 normalized cost ↓ | P3 log loss ↓ |
|---|---|---|---|---|
| framework 7 (v5, on the leaderboard) | 0.85461 | 0.50851 | 0.7182 | 0.19728 |
| framework 10 (v8) | 0.85484 | 0.50757 | 0.7143 | 0.19707 |
| v10b (sparse GAM, 4 and 6 kinked pairs) | 0.85498 | 0.50660 | 0.7114 | 0.19658 |
| v10c (+ 6 kinked pairs at C 0.05) | 0.85497 | 0.50674 | 0.7101 | 0.19655 |
| **v11 (+ hybrid)** | **0.85503** | **0.50659** | **0.7098** | **0.19654** |

Per seed (v11): P1 0.85560 / 0.85446; P2 0.50634 / 0.50684; P3 cost 0.7115 / 0.7081, log loss 0.19593 / 0.19715.
Against the leaderboard build (v5): P1 +0.0004, P2 −0.0019, P3 cost −0.008 in the same protocol.

The combiner was re-checked on the v10b fold libraries: bagged Caruana + Platt is still best or tied-best on all
three problems (stack and equal weights are not better), and the Bayes threshold again beats the learned one on
P3 (0.7126 vs 0.7131, 0.7081 vs 0.7142).

## 3. Code changes (v8 → v11)

- `SplineLogit(penalty="l1", pair_hinge=…, cat_hinge=…)`: standardised design, `liblinear` L1, interior hinge
  knots (1/16 … 15/16; extreme-quantile hinges add noise under L1), kinked interaction terms at the 20/40/60/80 %
  quantiles. GAM ladder: 11 configurations (no pairs, products, category slopes, three sparse kinked rungs);
  library quota 4 GAMs including the best of each penalty; GAM time share 15 % of the budget.
- New family `hyb` (`GamBoost`, `_PairEncoder`): sparse kinked GAM (C 0.1, 15 knots, 4 pairs, 2 category slopes)
  + LightGBM on its residual (init score), trees by early stopping; one configuration, library quota 1.
- `_gam_terms`: shared extraction of the measured numeric pairs / category slopes for the GAM and the hybrid.

## 4. Generalisation checks

**Synthetic suite** (16 problems, regret vs the Bayes-optimal score; lower is better):

| | AUC problems (6) | log-loss problems (6) | cost problems (4) |
|---|---|---|---|
| framework 10 (v8) | 0.01416 | 0.00824 | 0.0235 |
| v10c | 0.01421 | **0.00744** | **0.0233** |
| **v11** | 0.01418 | 0.00746 | 0.0239 |

v11 ties v8 / v10c on AUC and cost problems (differences far below one problem's threshold noise) and keeps the
large log-loss gain of the sparse GAM (numeric × category problem 0.0137 → 0.0107, linear-ish 0.0071 → 0.0059).

**Stress test** (`experiments/v11/logs/stress_test_v11.log`): all 12 cases pass.

## 5. Final build

`submission/v11/submission.zip` (12.7 MB): official `make_submission.py ../../data` on the Linux server (exact
leaderboard versions, 30 dedicated cores), log `experiments/v11/logs/make_submission_v11.log`; self-check passed on
Linux and macOS; the zip's `solution.py` is byte-identical to this folder's.

| | search | blend weights | Platt | OOF score (v8 build / v5 build) | fit |
|---|---|---|---|---|---|
| P1 | GAM 11, hybrid 1, LightGBM 32, XGBoost 18, CatBoost 4, ExtraTrees 3 | structured LightGBM 0.55, hybrid 0.17, monotone CatBoost 0.10, sparse GAM 0.09, structured XGBoost 0.08 | yes | AUC 0.85548 (0.85515 / 0.85489) | 388 s |
| P2 | same | **hybrid 0.51**, CatBoost 0.23, sparse GAM 0.19, structured LightGBM 0.06 | no | log loss 0.50624 (0.50744 / 0.50862) | 304 s |
| P3 | same | **sparse GAM 0.54**, XGBoost 0.22, LightGBM 0.13, CatBoost 0.06, hybrid 0.04 | yes | log loss 0.19538 (0.19605 / 0.19651) | 417 s |

P3 threshold 1/11; OOF cost 427.6 per 1000 cases (v8 build 430.3, v5 build 436.1).
