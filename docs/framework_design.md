# Framework design

`submission/sub2/solution.py` (version 5; `submission/improved/` is version 2, `experiments/variants/solution_v3.py` version 3) contains one reusable framework, the class `StudentSolution` with
`fit(X, y, metadata)`, `predict_proba(X, metadata)` and `get_threshold()`. There is no per-dataset logic: every
decision comes from measured properties of the training data or from the objective given in `metadata`.

## Pipeline inside `fit`

| Step | What it does and how it decides |
|---|---|
| **Objective** | Read from `metadata`. Leaderboard format: `objective` (`roc_auc` / `log_loss` / `decision_cost`), `requires_threshold`, `cost_false_positive`, `cost_false_negative`, `positive_label`. Other spellings are also understood (`metric`, `primary_metric`, nested `{"costs": {"fp": 1, "fn": 10}}`, F1, accuracy, average precision, Brier). If `requires_threshold` is true, a decision metric is used. With no metadata it defaults to log loss and threshold 0.5. |
| **Identifiers** | Dropped when named in metadata (`id_columns`), when named like `id` / `*_id` / `uuid` / `record_id`, or when they are text columns that are at least 90 % unique. Constant and all-missing columns are dropped. |
| **Types** | Numeric-looking strings become numeric; text and bool columns become categorical. Missing values, unseen categories and rare categories (< 0.5 %, or beyond the top 30 levels) are handled by the encoder. Missing or extra columns at predict time are tolerated, and column order does not matter. |
| **Encoders** | *native*: NaN kept + integer category codes (LightGBM, XGBoost, HGB). *imputed*: median + missing flags + one-hot (forests). *linear*: also log1p for skewed columns, standardise, clip ±5 sd (logistic regression). *frame*: numerics with NaN + string categories (CatBoost). |
| **Candidates** | Prior baseline, hinge-basis GAM (logistic regression on linear + piecewise-linear hinge terms of the numerics, one-hot categoricals; plain logistic regression when there are no numerics or more than 60 of them), LightGBM, XGBoost, CatBoost, ExtraTrees. HistGradientBoosting is used only when LightGBM is missing. |
| **Validation** | Stratified 5-fold out-of-fold (OOF) predictions; encoders are refit inside every fold, so nothing leaks. The fold count shrinks automatically for tiny minority classes. Data over 60k rows is subsampled to 40k rows for tuning only. |
| **Tuning** | Two stages per family within an absolute share of `time_budget` (default 200 s). Stage 1 is a *complexity ladder*: one configuration per tree depth (1, 2, 3, 4, 6; CatBoost 3, 4, 6) with the number of trees found by early stopping inside each fold (patience 150 / 100). Depth 1 is an additive model, so the ladder measures how much interaction the data contains. Stage 2 spends the remaining share on random draws around the best depth (learning rate, leaves, min leaf, subsample, colsample, L1/L2). Each family uses its own random stream. For decision metrics (cost/F1/accuracy) tuning uses log loss. Class weighting is only searched when the data is imbalanced **and** the metric is ranking- or label-based. |
| **Structure** | Before a boosting model is fitted with `inter_k > 0` or `mono`, the training rows it will see are profiled (cached per subset, so each fold is profiled once): a quick depth-2 model ranks feature pairs by the split gain they share, and a quick depth-1 model gives every numeric feature's additive shape; a feature whose shape is nearly monotone (\|Spearman\| > 0.9, at most 2 sign changes) gets a direction. The model is then fitted with LightGBM / XGBoost / HGB `interaction_constraints` (each feature alone plus the top `inter_k` pairs) and `monotone_constraints` (CatBoost: monotone only). The ladder tries `inter_k` 8 and 24 at depth 2 and 16 at depth 3; the refinement samples 4-32 pairs and monotone on/off. On data without sparse interactions or monotone effects the constraints simply fall away. |
| **Selection** | The best config of each family (for boosting families: the best structured **and** the best unconstrained one) is re-scored on *fresh* folds with a fixed number of trees (removes the early-stopping optimism), then greedy (Caruana) ensemble selection with replacement runs on the **logit** average of the members. The criterion is the metric itself for ranking metrics (AUC, average precision) and log loss otherwise, including cost problems, because the best-threshold cost on a finite sample is far noisier than a proper scoring rule. A single model wins if blending does not help. |
| **Calibration** | A Platt (sigmoid) map fitted on OOF predictions, kept only if nested CV lowers log loss. It is monotone, so AUC is unchanged. |
| **Threshold** | For cost metrics: minimise `FP·c_fp + FN·c_fn` on calibrated OOF probabilities, averaging the optimum of a smoothed cost curve with the median over 200 bootstrap resamples. The Bayes value `c_fp/(c_fp+c_fn)` (1/11 for P3) is only reported as a sanity check, not used. Other metrics return 0.5. |
| **Final fit** | The selected members are refit on all data (3 seeds for boosting / ExtraTrees members when there are fewer than 40k rows, otherwise 1; always 1 for the GAM). The tree count found by early stopping on (k-1)/k of the data is scaled by 1 + 0.5/(k-1). Members are averaged in logit space, the calibrator is applied, probabilities are clipped to [1e-4, 1-1e-4] and any non-finite value is replaced by the prior. |

## What `predict_proba` does

It applies the stored cleaner and encoders to the raw features, takes the weighted average of the members'
logits, applies the calibrator if one was kept, and clips the result. Nothing is fitted at prediction
time, which is what the leaderboard requires (it only loads the pickled object and calls `predict_proba`).

## Things the framework does *not* do (on purpose)

- **Feature selection:** a nested shadow-feature test was neutral on P1 and hurt P2 and P3 (see
  `experiment_notes.md`).
- **Resampling (SMOTE etc.):** distorts probabilities. Class weights are used instead, and only where the metric
  is ranking- or label-based.
- **Hard-coded thresholds or dataset fingerprints:** forbidden by the project rules (Section 8 of the brief).

## Why the complexity ladder (v3)

On all three public problems depth-1 or depth-2 boosting with many trees beats 15-31-leaf trees (see
`experiment_notes.md`), and a 40-trial Optuna search converges to the same place. Starting every boosting family
at depth 1 and letting the data ask for more interaction is therefore both faster and better, and it is a
measured property of the data rather than a dataset-specific rule, so it transfers to the private problems.
