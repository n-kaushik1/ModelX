# Framework design

`submission/improved/solution.py` contains one reusable framework, the class `StudentSolution` with
`fit(X, y, metadata)`, `predict_proba(X, metadata)` and `get_threshold()`. There is no per-dataset logic: every
decision comes from measured properties of the training data or from the objective given in `metadata`.

## Pipeline inside `fit`

| Step | What it does and how it decides |
|---|---|
| **Objective** | Read from `metadata`. Leaderboard format: `objective` (`roc_auc` / `log_loss` / `decision_cost`), `requires_threshold`, `cost_false_positive`, `cost_false_negative`, `positive_label`. Other spellings are also understood (`metric`, `primary_metric`, nested `{"costs": {"fp": 1, "fn": 10}}`, F1, accuracy, average precision, Brier). If `requires_threshold` is true, a decision metric is used. With no metadata it defaults to log loss and threshold 0.5. |
| **Identifiers** | Dropped when named in metadata (`id_columns`), when named like `id` / `*_id` / `uuid` / `record_id`, or when they are text columns that are at least 90 % unique. Constant and all-missing columns are dropped. |
| **Types** | Numeric-looking strings become numeric; text and bool columns become categorical. Missing values, unseen categories and rare categories (< 0.5 %, or beyond the top 30 levels) are handled by the encoder. Missing or extra columns at predict time are tolerated, and column order does not matter. |
| **Encoders** | *native*: NaN kept + integer category codes (LightGBM, XGBoost, HGB). *imputed*: median + missing flags + one-hot (forests). *linear*: also log1p for skewed columns, standardise, clip ±5 sd (logistic regression). *frame*: numerics with NaN + string categories (CatBoost). |
| **Candidates** | Prior baseline, logistic regression, LightGBM, XGBoost, HistGradientBoosting, ExtraTrees, RandomForest. CatBoost is added **only if a categorical has more than 30 levels**: it was about 50× slower and not better on low-cardinality data. |
| **Validation** | Stratified 5-fold out-of-fold (OOF) predictions; encoders are refit inside every fold, so nothing leaks. The fold count shrinks automatically for tiny minority classes. Data over 60k rows is subsampled to 40k rows for tuning only. |
| **Tuning** | A default config plus a time-boxed random search per family (`time_budget`, default 150 s). Each family has a fixed share of that budget; the shares are absolute, so adding a family does not starve the others, and unused time rolls over to the next family. Each family uses its own random stream (seeded from `random_state` + family name), so the search does not change when a library is missing. For decision metrics (cost/F1/accuracy) tuning uses log loss, a proper score that does not depend on a threshold. Class weighting is only searched when the data is imbalanced **and** the metric is ranking- or label-based, never for log loss or cost, where it would damage calibration. |
| **Selection** | The best config of each family is re-scored on *fresh* folds, then greedy (Caruana) ensemble selection runs on the actual metric. A single model wins if blending does not help. |
| **Calibration** | A Platt (sigmoid) map fitted on OOF predictions, kept only if nested CV lowers log loss. It is monotone, so AUC is unchanged. |
| **Threshold** | For cost metrics: minimise `FP·c_fp + FN·c_fn` on calibrated OOF probabilities, averaging the optimum of a smoothed cost curve with the median over 200 bootstrap resamples. The Bayes value `c_fp/(c_fp+c_fn)` (1/11 for P3) is only reported as a sanity check, not used. Other metrics return 0.5. |
| **Final fit** | The selected members are refit on all data (3 seeds for stochastic learners when there are fewer than 40k rows, otherwise 1; always 1 for logistic regression / CatBoost). Probabilities are clipped to [1e-4, 1-1e-4] and any non-finite value is replaced by the prior. |

## What `predict_proba` does

It applies the stored cleaner and encoders to the raw features, takes the weighted average of the members'
probabilities, applies the calibrator if one was kept, and clips the result. Nothing is fitted at prediction
time, which is what the leaderboard requires (it only loads the pickled object and calls `predict_proba`).

## Things the framework does *not* do (on purpose)

- **Feature selection:** a nested shadow-feature test was neutral on P1 and hurt P2 and P3 (see
  `experiment_notes.md`).
- **Resampling (SMOTE etc.):** distorts probabilities. Class weights are used instead, and only where the metric
  is ranking- or label-based.
- **Hard-coded thresholds or dataset fingerprints:** forbidden by the project rules (Section 8 of the brief).
