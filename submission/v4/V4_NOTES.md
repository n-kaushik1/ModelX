# Submission v4 (framework version 6) — status, changes and evidence

> **Status: work in progress (not yet a submission).** `solution.py` here is the current v4 development
> version. It has passed a smoke test, but the decisive check (outer 5-fold CV of v4 vs. v5 on all three
> problems) is still running, and `submission.zip` has not been built yet. Do not upload anything from this
> folder until this note says so.

Environment for every number below: Python 3.11.9, the exact leaderboard versions (`.venv311`: numpy 1.26.4,
pandas 2.3.1, scikit-learn 1.7.1, lightgbm 4.6.0, xgboost 3.1.1, catboost 1.2.10), Windows, 16 cores.
Raw outputs are in `experiments/v4/logs/`, scripts in `experiments/v4/` (paths from the repo root).

Naming: framework versions 3–5 live in `submission/sub2/`; this folder is **submission v4 = framework
version 6**. Below, "v4" means this new framework and "v5" the previous one (`sub2`, copied as
`experiments/v4/sol_v5.py` so the scripts are self-contained).

---

## 1. What v4 changes (and why)

| # | Change | Evidence |
|---|---|---|
| 1 | **Final predictor = k-fold bagging.** The k fold models fitted when the library is re-scored are kept and their logits averaged; the refit on all rows (trees ×1.125, 3 seeds) is gone. Forests (ExtraTrees) are still refit once, since they are bagged already (this also keeps the model file small). Very large data (tuning on a subsample) still uses the refit path. | §3.3: better than the v5 refit on all 3 problems in repeated outer CV. Also keeps the deployed predictor consistent with the OOF predictions that calibration and the threshold are fitted on. |
| 2 | **Library ensemble.** Instead of the best 1–2 configs per family, the top configs are re-scored and offered to the blend: LightGBM 3 (best, best of the other structured/unconstrained kind, runner-up), XGBoost 2, CatBoost 2, GAM 2, ExtraTrees 1. Caruana selection runs up to 40 steps (was 15). | §3.4: Caruana over a library beats the best single model on every problem. |
| 3 | **Combiner chosen by nested CV.** Caruana blend (+ Platt) vs. a logistic stack on the member logits are both scored on held-out OOF rows; the stack is used only if it wins by > 1e-4. | §3.4: the stack is best on P2 and worse on P1 / P3, so it must be a measured choice, not a fixed rule. |
| 4 | **Monotone CatBoost rung** (depth 3, lr 0.03, monotone directions from the nested structure detection); refinement samples `mono` for CatBoost too. | §3.2/§3.4: the most-weighted blend member on P1 and P3, tied-best CatBoost on P2. |
| 5 | **Deterministic search.** A fixed number of configurations per family (GAM 4, LightGBM 32, XGBoost 18, CatBoost 4, ExtraTrees 3); `time_budget` (300 s) is only a safety net for slow machines. v5's time-boxed search gave different models on different machines. | Reproducibility. |
| 6 | ExtraTrees: 300 trees, larger leaves (5 / 15). | Model size (a 5-fold ET bag made a 54 MB model). |

Unchanged from v5: cleaning/typing, encoders, structure detection (interaction pairs + monotone directions),
complexity ladder, hinge GAM, Platt calibration with nested check, P3 threshold rule (smoothed cost curve +
bootstrap median on calibrated OOF).

## 2. What was tried and rejected

| Idea | Result | Verdict |
|---|---|---|
| **Automated feature engineering** (OpenFE-style screen: a residual booster on top of base-model OOF logits for every candidate; `fe_screen.py`) — pairwise ratios, products, sums/differences of z-scores, group-relative values (x − mean(x \| category)), category crosses, count encodings: 1 493 (P1), 2 861 (P2), 2 868 (P3) candidates | Best log-loss gains 0.0009 / 0.0005 / 0.0006, i.e. what the maximum of thousands of noise draws gives; on P3 a pure-noise column scored +0.0003. | Rejected: the trees already capture the structure. |
| Data findings behind this | `revenue_per_employee` = `annual_revenue / employee_count` exactly; sensor pairs are noisy copies (slope 1.0, R² 0.91–0.98); several effects are step functions (e.g. P1 failure rate jumps at both ends of `process_temperature`). | Explains why trees beat smooth models. |
| EBM-style GA²M (mains-only cyclic boosting, then pairs on the residual) | best variant P1 0.8542 / P2 0.5106 / P3 0.1977 vs reference 0.8540 / 0.5099 / 0.1978 | Not better. |
| GAM variants (two-sided hinges, 12–20 knots, pair products, pair hinges) | all worse than the v5 hinge GAM | Rejected. |
| LightGBM tweaks: lr 0.01, subsample 0.5, min_child 60, colsample 0.5, 3 seeds, 3 leaves | all within ±0.0005 of the reference (fold-seed noise) | Plateau. |

## 3. Experiments (5-fold CV repeated over 3 fold seeds unless stated)

### 3.1 Reference
Structured LightGBM (depth 2, 8 measured pairs, monotone) = "B": P1 AUC 0.85402, P2 log loss 0.50985,
P3 log loss 0.19775 (cost at 1/11 after Platt 0.7212).

### 3.2 Single models (selected rows of `exp_models_p*.log`)

| model | P1 AUC ↑ | P2 log loss ↓ | P3 log loss ↓ |
|---|---|---|---|
| B (LightGBM d2, structured) | 0.85402 | 0.50985 | 0.19775 |
| B, subsample 0.5 | 0.85437 | **0.50941** | 0.19781 |
| B, 3 leaves | 0.85424 | 0.50998 | **0.19736** |
| EBM-style, 16 pairs | 0.85421 | 0.51059 | 0.19811 |
| XGBoost d2 structured | 0.85352 | 0.51022 | 0.19767 |
| CatBoost d4 | 0.85262 | 0.51042 | 0.19754 |
| **CatBoost d3 lr 0.03 monotone** | 0.85347 | 0.51037 | 0.19755 |
| hinge GAM (v5), C 0.05 | 0.84733 | 0.51076 | 0.20020 |
| ExtraTrees min_leaf 15 | 0.83871 | 0.52067 | 0.20103 |

### 3.3 Final predictor: refit vs. k-fold bagging (`exp_bagging.py`)
Outer 5-fold × 2 repeats; inside each outer-train an inner 5-fold gives the tree count and the fold models.

| predictor | P1 AUC ↑ | P2 log loss ↓ | P3 log loss ↓ | P3 cost @1/11 ↓ | calib. slope |
|---|---|---|---|---|---|
| refit, 1 seed | 0.85416 | 0.51046 | 0.19821 | 0.7262 | 1.00–1.035 |
| refit, 3 seeds (v5) | 0.85424 | 0.51031 | 0.19825 | 0.7244 | 1.00–1.035 |
| **5-fold bag** | **0.85505** | **0.51003** | **0.19811** | **0.7202** | 1.04–1.07 |
| 5-fold bag × 3 seeds | 0.85513 | 0.51011 | 0.19810 | 0.7204 | 1.05–1.07 |

Bagging wins on every problem and on both repeats. Its predictions are slightly under-confident (slope > 1);
whether that needs a correction for P2's log loss is being checked on the outer-CV holdout (§4).

### 3.4 Ensembles (`blend_study.py`, 34 library members incl. CatBoost)
Honest protocol: weights chosen on a random half of the rows (fold-seed-0 OOF), scored on the other half
(fold-seed-1 OOF), 40 repeats.

| strategy | P1 AUC ↑ | P2 log loss ↓ | P3 log loss ↓ |
|---|---|---|---|
| reference B | 0.85252 | 0.51047 | 0.19725 |
| Caruana over the library | 0.85276 | 0.50882 | **0.19680** |
| Caruana + Platt | 0.85276 | 0.50892 | 0.19687 |
| logistic stack (C = 1) | 0.85214 | **0.50847** | 0.19787 |
| equal weights, all members | 0.85281 | 0.50904 | 0.19691 |

Most-weighted members: P1 CatBoost-monotone, EBM-style, hinge GAM; P2 LightGBM d3-structured, CatBoost d4,
GAMs; P3 CatBoost-monotone, LightGBM d3-structured, EBM-style.

## 4. Still to do

1. Robustness stress test of v4 (tiny / 1 %-positive / 70k-row / all-categorical / odd-column data, empty or
   missing metadata, unseen categories, reload): running.
2. **Outer 5-fold CV, v4 vs. v5, all three problems** (fit on 80 % → joblib save/load → score the raw 20 %,
   pooled over all rows; `eval_outer.py`): running.
3. Offline holdout analysis of the saved library predictions (`lib_study.py`): combiner choice, calibration of
   bagged predictions (P2), P3 threshold rule.
4. Build `submission.zip` with the official `make_submission.py`, self-check, reload test, final numbers here.
