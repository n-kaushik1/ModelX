# Framework design

`submission/v7/solution.py` (framework version 9; `submission/v6/` is version 8, `submission/v5/` version 7, `submission/v4/` version 6) contains one reusable framework, the class `StudentSolution` with
`fit(X, y, metadata)`, `predict_proba(X, metadata)` and `get_threshold()`. There is no per-dataset logic: every
decision comes from measured properties of the training data or from the objective given in `metadata`.

## Pipeline inside `fit`

| Step | What it does and how it decides |
|---|---|
| **Objective** | Read from `metadata`. Leaderboard format: `objective` (`roc_auc` / `log_loss` / `decision_cost`), `requires_threshold`, `cost_false_positive`, `cost_false_negative`, `positive_label`. Other spellings are also understood (`metric`, `primary_metric`, nested `{"costs": {"fp": 1, "fn": 10}}`, F1, accuracy, average precision, Brier). If `requires_threshold` is true, a decision metric is used. With no metadata it defaults to log loss and threshold 0.5. |
| **Identifiers** | Dropped when named in metadata (`id_columns`), when named like `id` / `*_id` / `uuid` / `record_id` **and** at least 50 % of their values are distinct (a low-cardinality `machine_id` stays a categorical), or when they are text columns that are at least 90 % unique. Constant and all-missing columns are dropped. |
| **Types** | Numeric-looking strings become numeric; date/time columns (datetime dtype, or text that parses as full dates) become days since 1970, month, day of week and hour; timedeltas become seconds; other text and bool columns become categorical. Missing values, unseen categories and rare categories (< 0.5 %, or beyond the top 30 levels) are handled by the encoder. Missing or extra columns at predict time are tolerated, and column order does not matter. |
| **Redundancy** | Numeric pairs with \|corr\| ≥ 0.9 in the same units (means within 0.1 sd, sd ratio 0.8–1.25) are decomposed as measurement error: if `b = a + e`, var(a) = cov(a, b), so the clean column's noise share (var − cov)/var is ≈ 0 and the copy's is positive. A noisy copy (one share ≤ ⅓ of the other) is dropped; twin readings (both shares ≥ 0.008 and similar) are kept and their noise-weighted average is added. No target and no column names are used. |
| **Encoders** | *native*: NaN kept + integer category codes (LightGBM, XGBoost, HGB). *imputed*: median + missing flags + one-hot (forests). *linear*: also log1p for skewed columns, standardise, clip ±5 sd (logistic regression). *frame*: numerics with NaN + string categories (CatBoost). |
| **Candidates** | Prior baseline, hinge-basis GAM (logistic regression on linear + piecewise-linear hinge terms of the numerics, optionally plus the products of the top 2–6 measured numeric pairs (bilinear interaction terms), one-hot categoricals; plain logistic regression when there are no numerics or more than 60 of them), LightGBM, XGBoost, CatBoost, ExtraTrees. HistGradientBoosting is used only when LightGBM is missing. |
| **Validation** | Stratified 5-fold out-of-fold (OOF) predictions; encoders are refit inside every fold, so nothing leaks. The fold count shrinks automatically for tiny minority classes. Data over 60k rows is subsampled to 40k rows for tuning only. |
| **Tuning** | Two stages per family with a fixed number of configurations (LightGBM 32, XGBoost 18, CatBoost 4, ExtraTrees 3; the GAM's 2 ladder rungs always run and its random refinements only within its small time share), so the boosting search does not depend on the machine; `time_budget` (300 s) is otherwise only a safety net for slow machines. Stage 1 is a *complexity ladder*: one configuration per tree depth (1, 2, 3, 4, 6; CatBoost 3, 4, 6) with the number of trees found by early stopping inside each fold (patience 150 / 100). Depth 1 is an additive model, so the ladder measures how much interaction the data contains. Stage 2 spends the remaining share on random draws around the best depth (learning rate, leaves, min leaf, subsample, colsample, L1/L2). Each family uses its own random stream. For decision metrics (cost/F1/accuracy) tuning uses log loss. Class weighting is only searched when the data is imbalanced **and** the metric is ranking- or label-based. |
| **Structure** | Before a boosting model is fitted with `inter_k > 0` or `mono`, the training rows it will see are profiled (cached per subset, so each fold is profiled once): a quick depth-2 model ranks feature pairs by the split gain they share, and a quick depth-1 model gives every numeric feature's additive shape; a feature whose shape is nearly monotone (\|Spearman\| > 0.9, at most 2 sign changes) gets a direction. The model is then fitted with LightGBM / XGBoost / HGB `interaction_constraints` (each feature alone plus the top `inter_k` pairs) and `monotone_constraints` (CatBoost: monotone only). The ladder tries `inter_k` 8 and 24 at depth 2 and 16 at depth 3; the refinement samples 4-32 pairs and monotone on/off. On data without sparse interactions or monotone effects the constraints simply fall away. |
| **Library** | The top configurations of each family (LightGBM 3: best, best of the other structured/unconstrained kind, runner-up; XGBoost 2; CatBoost 2; GAM 2; ExtraTrees 1) are re-scored on *fresh* folds with a fixed number of trees (removes the early-stopping optimism). Their k fold models are kept, and their OOF predictions drive the combiner, the calibration and the threshold. |
| **Combiner** | *Bagged ensemble selection* (Caruana et al. 2006): greedy forward selection with replacement on the **logit** OOF predictions, repeated on 25 bootstrap resamples of the rows, weights averaged, weights below 0.02 dropped. The criterion is the metric itself for ranking metrics (AUC, average precision) and log loss otherwise, including cost problems, because the best-threshold cost on a finite sample is far noisier than a proper scoring rule. On the outer-CV holdout it was first or second on all three public problems, while a single selection pass, a logistic stack and equal weights were each last on at least one. |
| **Calibration** | A Platt (sigmoid) map fitted on OOF predictions, kept only if nested CV lowers log loss. It is monotone, so AUC is unchanged. |
| **Threshold** | For cost metrics: the decision-theoretic threshold c_fp / (c_fp + c_fn) on the calibrated probabilities. The empirical OOF cost minimiser (smoothed cost curve + bootstrap median) replaces it only if it wins out of sample (fitted in-bag, scored out-of-bag) in ≥ 95 % of 400 bootstrap rounds. On 30 synthetic cost problems and on the public P3 outer-CV predictions the Bayes threshold was better than the empirical one. Other decision metrics (F1, accuracy, balanced accuracy) use the empirical rule; ranking and probability metrics return 0.5. |
| **Final predictor** | *k-fold bagging*: each selected boosting / GAM member contributes its k fold models (logits averaged); in repeated outer CV this beat a refit on all rows on all three public problems, and it keeps the deployed predictor consistent with the OOF predictions the calibration and threshold were fitted on. ExtraTrees (already a bagged ensemble) is refit once on all rows; when tuning ran on a subsample (> 60k rows) every member is refit on all rows with the tree count scaled up. Members are averaged in logit space, the calibrator is applied, probabilities are clipped to [1e-4, 1-1e-4] and any non-finite value is replaced by the prior. |

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
