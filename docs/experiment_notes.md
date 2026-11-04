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
