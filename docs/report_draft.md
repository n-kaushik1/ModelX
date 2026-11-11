# Final report: draft content

Draft material for the brief final report (Section 14 of the brief). Copy it into the official template and
shorten as needed. Fill in the two bracketed leaderboard items yourself.

---

## Common framework (one `StudentSolution` for every problem)

1. **Profile and clean** the data: drop identifiers (named like `*_id` **and** mostly unique), constant and empty
   columns; coerce numeric-looking text; turn dates into numeric parts; detect numeric and categorical columns;
   measure missingness, imbalance and cardinality. **Measured redundancy**: a column that is a noisy copy of
   another column in the same units (detected from the covariance structure: for `b = a + noise`,
   var(a) = cov(a, b)) is dropped; two noisy readings of the same quantity are averaged.
2. **Encode per model family**: native NaN + category codes for gradient boosting, raw strings for CatBoost, and
   median imputation + missing flags + log transform + scaling + one-hot for the GAM.
3. **Measure the structure** (nested in every fold): a quick depth-2 booster ranks interacting feature pairs, and
   a quick additive booster gives each feature's shape and monotone direction. Boosters are searched with and
   without these as `interaction_constraints` / `monotone_constraints`.
4. **Compare and tune families** (LightGBM, XGBoost, CatBoost, a hinge-basis GAM optionally with bilinear terms
   for the measured feature pairs and category-dependent slopes, ExtraTrees) by stratified 5-fold
   OOF predictions on the problem's metric (log loss when a threshold is needed). Each family runs a
   **complexity ladder** (depth 1, 2, 3, 4, 6, trees by early stopping), then random refinement around the best
   rung.
5. **Library and k-fold bagging**: the top configurations are refit on fresh folds with a fixed tree count; their
   fold models form the final predictor (their logits are averaged).
6. **Bagged ensemble selection** in logit space (Caruana selection on bootstrap resamples), then **Platt
   calibration**, kept only if nested CV shows a log-loss gain.
7. **Threshold** (decision metrics only): the cost-derived threshold c_fp / (c_fp + c_fn) applied to the
   calibrated probabilities; the empirical OOF cost minimiser replaces it only if it is better out of sample in
   ≥ 95 % of bootstrap rounds.

**Validation strategy.** Inside `fit`: stratified 5-fold OOF for tuning, fresh folds for the library, and
nested folds for calibration and the threshold. For framework decisions: **paired outer 5-fold CV of the whole
framework** (fit on 4/5, joblib save and reload, predict the raw 1/5, exactly as the leaderboard does),
repeated over fold seeds. The leaderboard was used only as an external check.

## Per problem

| | P1 Manufacturing (ROC-AUC) | P2 SME credit (log loss) | P3 Telecom (FP + 10·FN) |
|---|---|---|---|
| Data | 10,000 rows; 20 numeric, 7 categorical; 30.5 % positives; 2.5–10 % missing in 6 columns | 12,000 rows; 30 numeric, 3 categorical; 27 % positives; 1–4 % missing in 5 columns | 15,000 rows; 29 numeric, 6 categorical; **6 % positives**; 2–5 % missing in 5 columns |
| Key characteristics | Strongly non-linear but **additive**: hinge and U-shaped effects (cycle time, process temperature), a few interacting pairs; noisy duplicate sensors; several columns with no signal. Logistic regression reaches only ~0.72 AUC vs ~0.855 for boosting. | Mostly additive and **monotone** effects (well suited to a GAM); a derived ratio column (`revenue_per_employee` = revenue / employees); probability quality matters, not ranking. | Imbalanced; sharp hinge (throughput utilisation) and U-shape (traffic change); one dominant interaction pair; cost-sensitive decision. |
| Preprocessing | Native NaN handling, category codes; noisy copies `temperature_sensor_2`, `material_density_estimate` dropped; no feature engineering (a screen of ~1,500 generated features found nothing above noise) | copies `utilization_3m_average`, `account_balance_last_month` dropped; the GAM uses log transforms and hinge bases | copy `utilization_15m_average` dropped; the two packet-loss readings averaged |
| Models compared | LR / spline GAM / hinge GAM (with and without pair terms), LightGBM (incl. linear trees, low learning rate), XGBoost, CatBoost, ExtraTrees, RF, HGB, EBM-style GA²M, DART, class-weighted boosters | same | same |
| Final model | Bag of fold models: structured LightGBM (depth 2, measured pairs, monotone) 0.65 + monotone CatBoost 0.16 + structured XGBoost 0.13 + interaction GAM 0.06; Platt | **GAM with bilinear terms for the 2 strongest measured pairs 0.49** + CatBoost 0.27 + structured LightGBM 0.18 + XGBoost 0.06 | **GAM with pair products and category-dependent slopes 0.32** + XGBoost 0.28 + CatBoost 0.29 + LightGBM 0.11; Platt; threshold 1/11 |
| Tuning decisions | Depth 1–2 trees win (deep trees lose 0.004 AUC); search depth first; early stopping for the tree count | Depth 2–3; the GAM is a strong blend member | Selection by log loss, not by the noisy best-threshold cost |
| Outer-CV score (final framework) | AUC 0.8548 (2 fold seeds) | log loss 0.5076 | normalized cost 0.714 |

**P3 threshold.** Chosen from the development data only. The probabilities are calibrated on out-of-fold
predictions (Platt), and the threshold that minimises the expected cost `FP + 10·FN` for calibrated
probabilities is 1/11 ≈ 0.091. An empirical alternative, the OOF cost minimiser, is used only if it beats 1/11 out
of sample in ≥ 95 % of bootstrap rounds; on P3 it does not. On held-out folds the cost-derived threshold was better
than the empirically tuned one (normalized cost 0.717 vs 0.722, averaged over three framework versions and two fold seeds), because a tuned
threshold adds variance once the probabilities are calibrated.

**Key observations**
- P2 behaves like an additive model plus a couple of bilinear interactions (top measured pair: operating margin
  × debt-service ratio). A GAM with the products of the two strongest measured pairs became the best single P2
  model (log loss 0.5085 vs 0.5101 without them) and improved the ensemble by 0.0008; trees approximate a product
  only with many splits.
- On P3 the effect of throughput utilisation depends on the node type (a measured category × numeric pair); a
  GAM with category-dependent slopes for such pairs became a major P3 blend member.
- Several columns are noisy copies of others (e.g. `temperature_sensor_2` = `process_temperature` + noise). The
  covariance structure identifies which one is clean; dropping the copies improved all three problems, while
  dropping the originals cost up to 0.017 AUC.
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
- **What should generalise.** Redundancy detection and the cost-derived threshold need no tuning to a dataset,
  and were also checked on synthetic data with known structure. Proper-scoring-rule selection, nested structure detection (it switches itself off
  when there are no monotone effects or sparse interactions), k-fold bagging, and calibration behind a nested
  check. The cleaner and stress test cover unseen categories, missing / extra / shuffled columns, identifier-like
  and date columns, tiny data, 1 % positives, 150 features, 300-level categoricals and 70k-row data.
- **Changes based on leaderboard feedback.** [fill in: e.g. "none; the leaderboard confirmed the local outer-CV
  ranking of versions" or the specific change]. Leaderboard position at submission: [fill in].
