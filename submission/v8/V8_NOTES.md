# Submission v8 (framework version 10): status, changes and evidence

> **Status: final framework.** `submission.zip` in this folder is the build to upload (see §5).

v8 = v7 + **category × numeric slope terms in the GAM** (a varying-coefficient GAM). Scripts: `experiments/v7/zoo.py`
(`gam_rich_f`), `blend_zoo.py`; logs `experiments/v7/logs/grich_local*.out`, `blend_grich*_p*.log`.

## 1. Diagnosis: what does the GAM miss on P1 / P3?

After v7 the GAM was the best single model on P2 but still far behind the trees on P1 (AUC 0.847 vs 0.854) and P3
(log loss 0.2004 vs 0.1972). A diagnostic GAM with switchable bases (same honest zoo protocol: two 5-fold
partitions, inner early stopping) tested which kind of structure closes the gap:

| GAM basis added to the hinge GAM | P1 AUC ↑ | P2 log loss ↓ | P3 log loss ↓ |
|---|---|---|---|
| none (framework hinge GAM) | 0.84734 | 0.51011 | 0.20041 |
| step indicators at 10 / 20 quantiles | 0.84174 / 0.84207 | 0.52032 / 0.52729 | 0.20620 / 0.21095 |
| AND-rule terms 1[a>q]·1[b>q] for 4 pairs | 0.84509 | 0.51028 | 0.20109 |
| products of 2 numeric pairs | 0.84722 | **0.50849** | 0.20071 |
| **category × numeric slopes, 1 / 2 pairs** | **0.84806** / 0.84772 | 0.51029 | **0.19886 / 0.19888** |
| 2 category slopes + 4 / 6 numeric products | 0.84740 / 0.84721 | 0.50880 / 0.50884 | **0.19838 / 0.19817** |

P3's measured category × numeric pairs are `throughput_utilization × node_type` and `firmware_age_months ×
region_type`: the effect of utilisation differs by node type. P1's top one is `cycle_time ×
operator_experience_band`. Step and AND-rule bases over-fit in this form (no gain).

## 2. Ensemble value (honest half-split study, change vs the framework-like library)

| library | P1 AUC | P2 log loss | P3 log loss | P3 cost @ 1/11 |
|---|---|---|---|---|
| v7-like (pair GAM + LightGBM lr 0.01) | +0.00034 | −0.00083 | 0.00000 | +0.0002 (se 0.0003) |
| v7-like + GAM (2 category slopes, 4 products) | +0.00033 | −0.00084 | **−0.00044** (se 0.00003) | **−0.0032** (se 0.0008) |
| v7-like + GAM (2 category slopes, 6 products) | +0.00031 | −0.00084 | −0.00050 | −0.0027 (se 0.0009) |
| v7-like + GAM (1 category slope) | **+0.00041** | −0.00083 | −0.00018 | −0.0030 (se 0.0009) |

The first modelling change that improves P3's decision cost beyond noise.

## 3. What v8 changes (on top of v7)

`SplineLogit` takes `cat_slopes`: for a measured category × numeric pair, the numeric column times every one-hot
column of the category (unseen / rare / missing levels use the encoder's own `__rare__` / `__na__` columns).
Interaction terms (products and slopes) are standardised on the training rows. GAM ladder: no pairs (C 0.05 /
0.2), 2 and 4 products, 4 products + 2 slopes, 2 products + 1 slope; two refinements sample products (0–6),
slopes (0–4), C, knots and basis.

## 4. Final comparison (whole framework)

**Public problems**, paired outer 5-fold CV, mean of fold seeds 0 and 1 (tag `v8a`):

| | P1 AUC ↑ | P2 log loss ↓ | P3 normalized cost ↓ | P3 log loss ↓ |
|---|---|---|---|---|
| framework 7 (v5, on the leaderboard) | 0.85461 | 0.50851 | 0.7182 | 0.19728 |
| framework 9 (v7) | 0.85481 | **0.50752** | 0.7145 | 0.19722 |
| **framework 10 (v8)** | **0.85484** | 0.50757 | **0.7143** | **0.19707** |

v8 ties v7 on P1 / P2 / P3 cost and improves P3 log loss on both seeds (−0.00013 / −0.00018). The category-slope
GAM takes 14–46 % of P3's blend on every fold. The zoo study predicted a larger P3 gain; the framework's real
library (tuned CatBoost in particular) already captured part of the category × numeric structure.

**Synthetic suite** (mean regret, lower is better):

| | AUC problems (6) | log-loss problems (6) | cost problems (4) |
|---|---|---|---|
| framework 9 (v7) | **0.01416** | 0.00879 | **0.0223** |
| **framework 10 (v8)** | **0.01416** | **0.00824** | 0.0235 |

The log-loss gain comes from the problem with numeric × category effects (regret 0.0164 → 0.0137), exactly the
structure the new terms represent; the cost difference is one run's threshold noise (P3-like 0.025 vs 0.031,
cost5 0.036 vs 0.035).

**Also tested and rejected in this round** (honest blend study on top of the v8-like library): XGBoost at learning
rate 0.01 (±0.00002 everywhere), LightGBM at learning rate 0.005 (±0.00001), averaging whole framework versions
(v7 + v6 averaged is worse than v7 alone on every problem: the versions are too correlated).

## 5. Final build

`submission/v8/submission.zip` (11.6 MB): official `make_submission.py ../../data` on the Linux server (exact
leaderboard versions, 20 dedicated cores), log `experiments/v8/logs/make_submission_v8.log`; self-check passed on
Linux and macOS; the zip's `solution.py` is byte-identical to this folder's.

| | search (configs) | blend weights | Platt | OOF score (v7 build) | fit |
|---|---|---|---|---|---|
| P1 | GAM 8, LightGBM 32, XGBoost 18, CatBoost 4, ExtraTrees 3 | structured LightGBM 0.65, monotone CatBoost 0.16, structured XGBoost 0.13, interaction GAM 0.06 | yes | AUC 0.85515 (0.85510) | 298 s |
| P2 | same | interaction GAM 0.49, CatBoost 0.27, structured LightGBM 0.18, XGBoost 0.06 | no | log loss 0.50744 (0.50745) | 236 s |
| P3 | same | **interaction GAM 0.32**, XGBoost 0.28, CatBoost 0.29, LightGBM 0.11 | yes | log loss **0.19605** (0.19649) | 254 s |

P3 threshold 1/11; OOF cost 430.3 per 1000 cases (v7 build 432.3).

**Stress test** (`experiments/v8/logs/stress_test_v8.log`): all 12 cases pass.
