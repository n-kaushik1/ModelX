# Submission v13 (framework version 15): status, changes and evidence

> **Status: candidate, not yet built.** v12 (`submission/v12/`) stays the upload until the v13 zip is built and
> checked (§4).

v13 = v12 + changes that only act on data unlike the three public problems, chosen from a benchmark of the
framework against a strong conventional pipeline on 48 synthetic problems of 16 types (§2). On the public problems
v13 runs exactly the v12 code path: the same fold of P2 gives identical predictions to v12 with change 3 (§3).

| # | Change | Evidence |
|---|---|---|
| 1 | **Target encoding of high-cardinality categoricals.** A categorical with more levels than the one-hot limit (25) also gets a smoothed target encoding (logit of an m-estimate, m = 20) in every encoder except CatBoost's (which has its own ordered target statistics). On the training rows it is computed out of fold (5 folds inside the rows the model is fitted on), for new rows from all training rows; the columns are appended last, so the indices used by the measured interaction and monotone constraints do not move. Before, such a column reached LightGBM / XGBoost as an arbitrary integer code of its 30 most frequent levels and the GAM as that code plus a frequency, so only CatBoost could use it. | Single models on a 288-level problem (4k rows): AUC GAM 0.698 → 0.746, hybrid 0.707 → 0.744, LightGBM 0.698 → 0.724, XGBoost 0.695 → 0.723, ExtraTrees 0.670 → 0.713 (Bayes 0.786). Whole framework, three draws of the 150-level type (20k rows, AUC regret vs the true probabilities): **0.01335 → 0.00658, 0.01258 → 0.00877, 0.01483 → 0.00651**. Same fit time (side by side, 2 threads: 1,131 s vs 1,100 s). No effect on the public problems (no categorical there has more than 6 levels). |
| 2 | **Wide data keeps the GAM.** Above 60 numeric columns v12 replaced the GAM and the hybrid by a plain logistic regression. v13 fits them on the 60 numerics with the largest additive gain in the measured structure (a quick depth-1 booster on the same training rows, so the choice is nested in every fold); the hybrid's booster still sees every column. | Seven wide synthetic problems (80-120 numerics, 10-40 informative), regret v12 → v13: log loss (5) 0.01146 → 0.01128, 0.01387 → 0.01064, 0.01037 → 0.00914, 0.02281 → 0.02004, 0.02191 → 0.01621 (mean −0.0026, se 0.0008); AUC (2) 0.02753 → 0.02609, 0.03067 → 0.01925. Better on 7 of 7; fit time about +25 % on such data only. Stress case "150 features" passes; model 10.6 MB. |
| 3 | **Reproducible sparse GAM.** The L1 (`liblinear`) logistic regression is given `random_state=0`; before, it shuffled its coordinate order differently on every fit, so two fits on the same data differed (max 5.6e-4 on P2 for the sparse GAM, 3.9e-3 for the hybrid built on it). | Two v13 fits on the same data are now identical; v13 and v12 + this fix are identical on P2. |

## 1. Is anything left on the public problems?

Residual analysis of v12's outer-CV held-out predictions (every row predicted by a framework that never saw it;
`experiments/v13/resid_diag.py`, logs `experiments/v13/logs/resid_v12b_p*.log`):

- calibration: reliability chi-square p = 0.19-0.95; a nested Platt step on the held-out logits does not help;
- misfit per feature (20 quantile bins or levels) and per numeric pair (5 x 5 cells): nothing significant after
  correcting for the number of tests (smallest adjusted p 0.13, a checkerboard without structure);
- residual boosters (LightGBM with the framework's logit as the starting score, own 5-fold CV, depth 1-6): early
  stopping keeps 1-2 trees and the held-out log loss gets slightly worse on every problem.

The public problems are at their ceiling for 10-15k rows. A threshold-tailored proper scoring rule for the P3
blend (beta family centred on 1/11, Buja et al. 2005) looked like −0.0055 cost on the outer folds but was
−0.0006 ± 0.0006 over 60 repeated half-splits and +0.00007 ± 0.00021 expected-cost regret on 14 synthetic cost
problems scored against the true probabilities: rejected.

## 2. Benchmark against a strong conventional pipeline

`experiments/v13/competitor.py`: Optuna-tuned LightGBM (40 TPE trials, 5-fold CV with early stopping on the
objective), the 5 fold models averaged, Platt calibration and, for costs, the out-of-fold cost-minimising
threshold. 48 synthetic problems = 16 types x 3 random draws, each scored on a 20k-row test set against the true
probabilities (`experiments/v13/cmp.py`).

| objective | problems | v12 regret | conventional pipeline | difference (se) |
|---|---|---|---|---|
| ROC-AUC | 18 | **0.0156** | 0.0194 | 0.0038 (0.0008) |
| log loss | 18 | **0.0076** | 0.0117 | 0.0041 (0.0008) |
| decision cost | 12 | **0.0278** | 0.0429 | 0.0151 (0.0031) |

v12 wins 45 of 48. The three losses: two draws of the high-cardinality type (change 1) and one draw of the 80k-row
type (v12 wins the other two clearly).

Eight further types (2 draws each; regret v12 / conventional pipeline / v13 where v13 differs):

| type | v12 | conventional | v13 |
|---|---|---|---|
| three-way and XOR interactions (AUC) | **0.0297, 0.0329** | 0.0358, 0.0359 | = v12 |
| 800 rows, 40 numerics (log loss) | **0.0476, 0.0515** | 0.0586, 0.0742 | = v12 |
| 300-level categorical with signal (log loss) | 0.0148, 0.0254 | 0.0115, 0.0170 | **0.0086, 0.0118** |
| 3 % positives (log loss) | **0.0042, 0.0040** | 0.0048, 0.0044 | = v12 |
| cost 1:50, 2 % positives | **0.0114, 0.0498** | 0.0223, 0.0575 | = v12 |
| 12 categoricals with category x category effects (AUC) | **0.0190, 0.0246** | 0.0239, 0.0333 | = v12 |
| near-separable (log loss) | **0.0190, 0.0189** | 0.0286, 0.0244 | = v12 |
| very noisy labels (AUC) | **0.0086, 0.0134** | 0.0128, 0.0166 | = v12 |

Over all 64 problems v12 wins 59 and v13 63; the remaining loss is the one 80k-row draw.

## 3. Checks

- Identity on the public problems: one P2 outer fold, v12 + change 3 vs v13, every model family: identical.
- Review probes (12 edge cases) and the private-problem stress test (12 cases): all pass
  (`experiments/v13/logs/probe_v13.log`, `stress_test_v13.log`).

Tested and rejected in this round:
- **Threshold-tailored blend selection for decision costs** (§1): no gain once measured properly.
- **Large data: library, blend, calibration and threshold on all rows** (instead of the 40k-row tuning
  subsample) up to 120k rows: regret 0.00235 → 0.00229, 0.00240 → 0.00219, 0.00279 → 0.00257 on three 80k-row
  problems (−0.00016, se 0.00004), but the fit took about twice as long (about 50 min at 2 threads) and the model
  5-11 MB instead of 0.5-1.6 MB. With the private retraining time unknown, not worth it.

## 4. Build
