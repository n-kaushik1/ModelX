# Final report: draft content

Draft material for the brief final report (Section 14 of the brief). Copy it into the official template and
shorten as needed. Fill in the two bracketed leaderboard items yourself.

---

## Common framework (one `StudentSolution` for every problem)

1. **Profile and clean** the data: drop identifiers (named like `*_id` **and** mostly unique), constant and empty
   columns; coerce numeric-looking text; turn dates into numeric parts; detect numeric and categorical columns;
   measure missingness, imbalance and cardinality.
2. **Encode per model family**: native NaN + category codes for gradient boosting, raw strings for CatBoost, and
   median imputation + missing flags + log transform + scaling + one-hot for the GAM.
3. **Measure the structure** (nested in every fold): a quick depth-2 booster ranks interacting feature pairs, and
   a quick additive booster gives each feature's shape and monotone direction. Boosters are searched with and
   without these as `interaction_constraints` / `monotone_constraints`.
4. **Compare and tune families** (LightGBM, XGBoost, CatBoost, a hinge-basis GAM, ExtraTrees) by stratified 5-fold
   OOF predictions on the problem's metric (log loss when a threshold is needed). Each family runs a
   **complexity ladder** (depth 1, 2, 3, 4, 6, trees by early stopping), then random refinement around the best
   rung.
5. **Library and k-fold bagging**: the top configurations are refit on fresh folds with a fixed tree count; their
   fold models form the final predictor (their logits are averaged).
6. **Bagged ensemble selection** in logit space (Caruana selection on bootstrap resamples), then **Platt
   calibration**, kept only if nested CV shows a log-loss gain.
7. **Threshold** (decision metrics only): minimise the stated cost on calibrated OOF predictions (smoothed cost
   curve + bootstrap median).

**Validation strategy.** Inside `fit`: stratified 5-fold OOF for tuning, fresh folds for the library, and
nested folds for calibration and the threshold. For framework decisions: **paired outer 5-fold CV of the whole
framework** (fit on 4/5, joblib save and reload, predict the raw 1/5, exactly as the leaderboard does),
repeated over fold seeds. The leaderboard was used only as an external check.

## Per problem

| | P1 Manufacturing (ROC-AUC) | P2 SME credit (log loss) | P3 Telecom (FP + 10·FN) |
|---|---|---|---|
| Data | 10,000 rows; 20 numeric, 7 categorical; 30.5 % positives; 2.5–10 % missing in 6 columns | 12,000 rows; 30 numeric, 3 categorical; 27 % positives; 1–4 % missing in 5 columns | 15,000 rows; 29 numeric, 6 categorical; **6 % positives**; 2–5 % missing in 5 columns |
| Key characteristics | Strongly non-linear but **additive**: hinge and U-shaped effects (cycle time, process temperature), a few interacting pairs; noisy duplicate sensors; several columns with no signal. Logistic regression reaches only ~0.72 AUC vs ~0.855 for boosting. | Mostly additive and **monotone** effects (well suited to a GAM); a derived ratio column (`revenue_per_employee` = revenue / employees); probability quality matters, not ranking. | Imbalanced; sharp hinge (throughput utilisation) and U-shape (traffic change); one dominant interaction pair; cost-sensitive decision. |
| Preprocessing | Native NaN handling, category codes; no feature engineering (a screen of ~1,500 generated features found nothing above noise) | same; the GAM uses log transforms and hinge bases | same |
| Models compared | LR / spline GAM / hinge GAM, LightGBM, XGBoost, CatBoost, ExtraTrees, RF, HGB, EBM-style GA²M, DART | same | same |
| Final model | Bag of fold models: structured LightGBM (depth 2, measured pairs, monotone) 0.58 + monotone CatBoost 0.25 + small GAM / LightGBM / CatBoost weights; Platt | GAM 0.35 + structured LightGBM 0.23 + XGBoost 0.21 + CatBoost 0.17; no Platt (nested check) | structured LightGBM 0.34 + CatBoost 0.33 + XGBoost 0.23 + ExtraTrees 0.06 + LightGBM 0.04; Platt |
| Tuning decisions | Depth 1–2 trees win (deep trees lose 0.004 AUC); search depth first; early stopping for the tree count | Depth 2–3; the GAM is a strong blend member | Selection by log loss, not by the noisy best-threshold cost |
| Outer-CV score (final framework) | AUC 0.8546 (2 fold seeds) | log loss 0.5085 | normalized cost 0.718 |

**P3 threshold.** Learned from the development data only: the cost `FP + 10·FN` is minimised on calibrated OOF
predictions, averaging a smoothed cost curve's optimum with the median of 200 bootstrap optima. The learned
threshold (0.12 on the full data; 0.08–0.12 across folds) lands near the Bayes value 1/11 = 0.091. On held-out
folds both give the same cost, which confirms the probabilities are well calibrated.

**Key observations**
- Measured structure transfers: constraining boosters to the measured feature pairs and monotone directions
  improved all three problems in 5-fold CV (P1 +0.001 AUC, P2 −0.001 log loss, P3 −0.0007 log loss).
- Averaging the k fold models beat refitting on all rows on every problem (P1 0.8551 vs 0.8542 AUC). Averaging
  models that each saw 80 % of the rows also beat 10-fold bags (models on 90 %): variance, not bias, limits these
  models.
- Gains from tuning below the fold-seed spread (≈0.0005) do not transfer. A 45-trial Optuna search that looked
  0.0017 log loss better on its own folds was only about 0.0005 better on fresh folds. The same holds for choosing the
  combiner by nested CV.
- The P3 cost on a 3,000-row fold ranges from 0.65 to 0.78 for the *same* model, so small leaderboard
  differences on P3 are luck.

## Reflection

- **How the framework adapts.** Everything is driven by measured properties or by `metadata`. The ladder decides
  the depth (additive vs. interacting data), structure detection decides the constraints, OOF comparison decides
  the families, and bagged selection decides the weights (e.g. the GAM gets a third of P2 and nearly nothing on
  P3). The metric decides the tuning criterion and whether a threshold is learned, and the cost ratio sets it.
- **What should generalise.** Proper-scoring-rule selection, nested structure detection (it switches itself off
  when there are no monotone effects or sparse interactions), k-fold bagging, and calibration behind a nested
  check. The cleaner and stress test cover unseen categories, missing / extra / shuffled columns, identifier-like
  and date columns, tiny data, 1 % positives, 150 features, 300-level categoricals and 70k-row data.
- **Changes based on leaderboard feedback.** [fill in: e.g. "none; the leaderboard confirmed the local outer-CV
  ranking of versions" or the specific change]. Leaderboard position at submission: [fill in].
