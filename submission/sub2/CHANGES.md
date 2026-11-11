# sub2 (framework version 5) — what changed and why

Upload `sub2.zip` on Codabench (My Submissions), wait for *Finished*, then click the green leaderboard icon.
`sub2.zip` was built with the official `make_submission.py` in the exact leaderboard environment
(Python 3.11.18, numpy 1.26.4, pandas 2.3.1, scikit-learn 1.7.1, lightgbm 4.6.0, xgboost 3.1.1, catboost 1.2.10)
and passed its self-check; `experiments/robust_test.py` also passes (unseen categories, missing / extra /
shuffled columns, fresh-process joblib load).

## Files in this zip

| path | what |
|---|---|
| `submission/sub2/solution.py` | the version-4 framework (`StudentSolution`) |
| `experiments/variants/solution_v3.py`, `sub2_v3.zip` | version 3 (previous build), kept as fallback |
| `submission/sub2/{make_submission,local_test,problems}.py, requirements.txt` | unchanged official kit files |
| `submission/sub2/CHANGES.md` | this file |
| `README.md`, `docs/framework_design.md`, `docs/experiment_notes.md` | updated docs (results, design, experiments) |
| `experiments/sub2/*.py` | experiment scripts behind the v3 decisions (run from the repo root) |
| `experiments/logs/exp{1,2,3}_p*.log, exp4_p3.log, eval_sub2_p*.log, robust_test_sub2.log, make_submission_sub2.log` | raw outputs |

To drop into the repository: unzip at the repo root (it only adds `submission/sub2/`, `experiments/sub2/`, new
logs, and replaces the three docs). `submission/improved/` and `submission/baseline/` are untouched.

## Version 4 on top of version 3: measured structure as model constraints

A deeper analysis (`experiments/sub2/exp5.py`, `exp6.py`) showed that all three problems are additive with
hinge / U-shaped effects, many cleanly monotone features, and a few strong feature pairs. v4 therefore
*measures* this structure on the training rows of every fit (nested in each CV fold, cached per subset):
a quick depth-2 model ranks interacting pairs by shared split gain, a quick depth-1 model gives each numeric
feature's additive shape and a monotone direction where the shape is monotone. LightGBM / XGBoost are then
searched with and without `interaction_constraints` (each feature alone + the top `inter_k` pairs) and
`monotone_constraints`; `inter_k` (4-32) and `mono` are tuned like any other hyper-parameter. The best structured
and the best unconstrained configuration of each boosting family both go to the blend. On 5-fold CV of the full
development data the constrained LightGBM alone beats the whole v3 blend on every problem (P1 AUC 0.8560 vs
0.8549, P2 log loss 0.5092 vs 0.5097, P3 log loss 0.1966 vs 0.1973), and v4 blends gain a little more.

## Framework changes (v2 -> v3), all generic (driven by data properties or metadata)

1. **Complexity ladder with early stopping.** Each boosting family (LightGBM, XGBoost, CatBoost) first fits one
   configuration per tree depth (1, 2, 3, 4, 6; CatBoost 3, 4, 6); the number of trees comes from early stopping
   inside every CV fold instead of being a random search dimension. The remaining time share is spent on random
   refinement around the best depth. Depth 1 is a purely additive model, so the ladder *measures* how much
   interaction the data has. On all three public problems depth 1-2 won: +0.003 AUC on P1 over 15-leaf trees.
2. **Unbiased re-scoring.** The best configuration per family is re-run on fresh folds with the tree count fixed
   before blending / calibration / threshold selection (removes early-stopping optimism).
3. **Greedy blending in logit space** (Caruana selection with replacement). Ranking metrics use the metric itself;
   log loss and cost problems use log loss. A P3 transfer test showed the best-threshold cost is too noisy as a
   selection criterion and that log-loss ranking agrees with cost ranking.
4. **Spline-logistic GAM candidate** (cubic B-splines on standardised numerics + one-hot categoricals) replaces the
   plain logistic regression when 1-60 numeric columns exist. CatBoost is a regular candidate (best single model on
   P3). HistGradientBoosting only when LightGBM is missing; RandomForest removed (ExtraTrees kept).
5. **Refit**: tree counts scaled by 1 + 0.5/(k-1) for the full data; 3 seeds per stochastic member below 40k rows;
   members averaged in logit space, then the (optional, nested-validated) Platt calibrator.
6. **Safety**: CPU count respects `sched_getaffinity` (container limits); a hard cap of 2.5 x `time_budget` on the
   ladder; degenerate data (< 4 minority rows, all candidates failing) falls back to the prior; no NaN output.
7. Rejected after testing (see `docs/experiment_notes.md`): class weights for cost / log loss, near-duplicate
   feature denoising, LR stacking instead of greedy selection, deeper default trees.

## Version 5: hinge-basis GAM (from repeated-CV screening)

`experiments/sub2/exp8.py` re-screened candidates with repeated 5-fold CV over three fold seeds (DART, leaf counts,
pair counts, regularisation, a 45-trial Optuna search on the structured space, calibration variants, a
hinge-basis GAM). Only one change was consistent across problems and above the fold-seed spread: the GAM candidate
now uses a piecewise-linear hinge basis instead of cubic splines (alone: P1 AUC 0.849 vs 0.846, P2 log loss 0.5102
vs 0.5137, P3 0.2001 vs 0.2030; blended with the structured LightGBM it lowers P2 log loss by 0.0009 on every fold
seed). The Optuna search is the key negative result: its best configuration looked 0.0017 log loss better on the
tuning folds and only 0.0005 better on fresh folds, so fold-level tuning gains are not pursued.

## Final deep pass: nothing left that is not noise

`experiments/sub2/exp7.py` tried, on top of v4: 5-seed bagging, one-hot vs native categoricals, max_bin 63/1023,
extra-randomised trees, nested removal of zero-gain features, piecewise-linear trees, lower learning rate, path
smoothing and no subsampling. Every change moved the 5-fold OOF metrics by less than the holdout noise and none
was consistent across the three problems, so none was adopted. A private-problem stress test (tiny data, 1 %
positives, 150 features, 300-level categoricals, string labels, 70k rows, empty metadata, all-categorical table)
passes end to end including pickle / reload.

## Local results (3 stratified 80/20 splits, seeds 42/1/2)

| problem | v2 (improved) | v3 | **v4 (sub2)** |
|---|---|---|---|
| P1 ROC-AUC | 0.8458 | 0.8489 | **0.8492** (better than v2 on all 3 splits) |
| P2 log loss | 0.5099 | 0.5088 | **0.5087** (better than v2 on all 3 splits) |
| P3 normalized cost | 0.6824 | 0.6863 | **0.6802** (better than v2 on 2 of 3; split sd ~0.036) |

Models in `sub2.zip` / `subv3.zip` (v5, trained on all development rows, 3 vCPUs, ~5 min each): P1 structured
LightGBM + structured XGBoost + CatBoost (3 seeds each, Platt-calibrated); P2 hinge GAM + LightGBM + CatBoost;
P3 structured XGBoost + XGBoost + CatBoost, threshold 0.1068 (Bayes value 1/11 = 0.0909). The v4 build is kept
as `subv3_v4.zip` next to it.
