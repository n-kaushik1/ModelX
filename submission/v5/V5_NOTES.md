# Submission v5 (framework version 7): status, changes and evidence

> **Status: final framework.** `submission.zip` in this folder is built with the official `make_submission.py`
> (self-check passed, see §5). Upload it on Codabench (My Submissions), wait for *Finished*, then click the green
> leaderboard icon.

Naming: `submission/sub2/` = framework version 5 (the zip uploaded before), `submission/v4/` = framework
version 6, this folder = framework version 7. Below, "v4" and "v5" refer to these submission folders, and the old
sub2 build is called "sub2".

Environment: Python 3.11 with the exact leaderboard versions (numpy 1.26.4, pandas 2.3.1, scikit-learn 1.7.1,
lightgbm 4.6.0, xgboost 3.1.1, catboost 1.2.10). Runs on a 32-core Linux server (CPU only: nothing in the
framework or the experiments uses a GPU, and the private retraining will not have one) and an Apple M4.
Scripts: `experiments/v5/`, raw logs: `experiments/v5/logs/`.

**Protocol for every framework comparison: outer 5-fold CV of the whole framework** (`outer_fold.py`): fit on
4/5 of the development rows, joblib save + reload, predict the raw held-out 1/5, as the leaderboard does. Scores
are pooled over all rows; for P3 every fold uses the threshold its own model learned. Versions are compared on
the same folds (paired). `outer_agg.py` aggregates, `lib_study.py` re-combines the saved library logits offline.
**Tags in the logs:** `v5` = the old sub2 build (framework 5), `v4` = `submission/v4`, `v5new` = this folder.
Job lists: `jobs_round1.txt` (v4 vs sub2), `jobs_round2.txt` (v5new vs v4).

---

## 1. The pending v4 check: v4 vs sub2 (outer 5-fold CV, fold seed 0)

| | P1 AUC ↑ | P2 log loss ↓ | P3 normalized cost ↓ | P3 log loss ↓ | model size |
|---|---|---|---|---|---|
| sub2 (framework 5, uploaded) | 0.85459 | 0.50857 | 0.7192 | 0.19696 | 0.1–71 MB |
| **v4 (framework 6)** | **0.85495** | **0.50824** | **0.7180** | **0.19667** | 0.6–17 MB |

v4 is better on all three problems (P1 on all 5 folds). Per-fold P1: v4 0.8519 / 0.8443 / 0.8611 / 0.8543 /
0.8643 vs sub2 0.8514 / 0.8438 / 0.8608 / 0.8543 / 0.8638. The k-fold bag also keeps the files small (sub2's
ExtraTrees refits made 58–71 MB models on some folds).

## 2. What v5 changes

| # | Change | Why |
|---|---|---|
| 1 | **Combiner = bagged ensemble selection** (Caruana et al. 2006): greedy selection with replacement repeated on 25 bootstrap resamples of the OOF rows, weights averaged, weights < 0.02 dropped. Replaces "single Caruana pass vs. logistic stack, chosen by nested CV". Platt calibration with the nested check is unchanged. | §3.2: on the outer-CV holdout (2 fold seeds) it is first or second on every problem; every alternative is last on at least one. The logistic stack learned negative weights (e.g. LightGBM −0.42, XGBoost −0.60), which is fragile on unseen data, and the nested choice between combiners was noise-dominated. |
| 2 | **Identifier rule**: a column with an ID-like name (`*_id`, `uuid`, ...) is dropped only if it also behaves like an identifier (≥ 50 % distinct values). | A low-cardinality `machine_id` / `region_id` is a real categorical on a private problem. The public data is unaffected (the kit removes `record_id` before `fit`). |
| 3 | **Date / time columns** (datetime dtype, or text that parses as full dates) become numeric features: days since 1970, month, day of week, hour (each kept only if it varies). Timedelta columns become seconds. Before, they were dropped. | Generic robustness for private problems; band labels such as `"10-49"` are not mistaken for dates. |
| 4 | `bag_splits` parameter (folds of the k-fold bag, default 5 as before). | Tested, see §3.1. |

Everything else is v4: cleaning/typing, encoders, structure detection, complexity ladder, deterministic search,
library, k-fold bagging, Platt, P3 threshold rule.

## 3. Experiments

### 3.1 Bag design (`exp_kfold.py`, reference LightGBM B; outer 5-fold × 2 repeats)

| bag | P1 AUC ↑ | P2 log loss ↓ | P3 log loss ↓ | P3 cost @1/11 ↓ |
|---|---|---|---|---|
| **5-fold (v4)** | **0.85502** | 0.51003 | 0.19804 | 0.7206 |
| 10-fold (models see 90 %) | 0.85469 | 0.51011 | 0.19807 | 0.7221 |
| 5-fold × 2 partitions | 0.85500 | 0.50996 | 0.19801 | 0.7201 |
| 5-fold × 3 partitions | 0.85499 | 0.51002 | 0.19793 | 0.7160 |
| 10-fold × 2 partitions | 0.85477 | 0.51009 | 0.19803 | 0.7242 |

More data per model (10-fold) does not help, and extra partitions add < 0.0001: the 5-fold bag is kept.

### 3.2 Combiners on the outer-CV holdout (`lib_study.py`, v4 libraries, fold seeds 0 and 1)

Each outer fold's library (OOF logits on the 80 %, bagged logits on the held-out 20 %) is re-combined offline;
every combiner is fitted on the 80 % only. Seed 0 / seed 1 / mean (logs `logs/lib_study_v4_p*_s*.log`).

| combiner | P1 AUC ↑ | P2 log loss ↓ | P3 log loss ↓ |
|---|---|---|---|
| best single member (seed 0) | 0.85430 | 0.51032 | 0.19753 |
| Caruana (one pass) + Platt (v4 default) | 0.85495 / 0.85421 / 0.85458 | 0.50837 / 0.50892 / 0.50865 | 0.19665 / 0.19798 / 0.19732 |
| logistic stack, C = 1 | 0.85493 / 0.85434 / **0.85464** | 0.50838 / 0.50888 / 0.50863 | 0.19660 / 0.19824 / 0.19742 |
| equal weights + Platt | 0.85500 / 0.85393 / 0.85447 | 0.50805 / 0.50857 / **0.50831** | 0.19702 / 0.19787 / 0.19745 |
| **bagged Caruana + Platt** | 0.85511 / 0.85409 / 0.85460 | 0.50817 / 0.50876 / 0.50847 | 0.19666 / 0.19792 / **0.19729** |

Seed 0 only: logistic stack C = 0.01 0.85505 / 0.50817 / 0.19681; non-negative stack 0.85500 / 0.50823 / 0.19671;
Caruana initialised with all members 0.85505 / 0.50827 / 0.19667; nested-CV choice among Caruana / equal /
stacks 0.85490–0.85495 / 0.50808–0.50846 / 0.19665–0.19695.

- Bagged selection is first or second on every problem. Equal weights + Platt is best on P2 but last on P1 and
  P3 (and would be hurt by a weak family on a private problem); the stack is last on P3 and learns negative weights.
- Choosing the combiner by nested CV does not work: inner-CV differences of up to 0.001 AUC did not carry over.
- After Platt, the bagged predictions are very slightly under-confident on the holdout (needed slope
  1.013–1.018); the log-loss cost of that is below 0.0001, so no correction was added.

### 3.3 P3 threshold rules (same libraries, Caruana + Platt predictions)

| rule | pooled normalized cost | thresholds per fold |
|---|---|---|
| learned (v4: smoothed curve + bootstrap median) | 0.7147 | 0.088–0.118 |
| Bayes 1/11 on calibrated probabilities | 0.7147 | 0.091 |
| raw OOF optimum | 0.7159 | 0.088–0.122 |
| bootstrap mean | 0.7105 | 0.089–0.104 |

All within noise (sd of a pooled 15k-row cost ≈ 0.015); the v4 rule is kept.

### 3.4 CatBoost variants (`exp_cat.py`, 5-fold CV × 3 fold seeds, monotone directions nested)

| variant | P1 AUC ↑ | P2 log loss ↓ | P3 log loss ↓ |
|---|---|---|---|
| depth 3, lr 0.03, l2 5, monotone (v4 ladder rung) | 0.85346 | 0.51039 | 0.19751 |
| depth 2 | 0.85343 | 0.51121 | 0.19766 |
| Langevin boosting (SGLB) | 0.85325 | 0.51025 | 0.19756 |
| l2 20 | 0.85357 | 0.51037 | 0.19767 |

Nothing is consistently better: the CatBoost rungs are unchanged.

### 3.5 Literature check (November 2026)

The current tabular benchmarks (TabArena) put tuned + ensembled gradient boosting (LightGBM, CatBoost) at the top
of the classical models; everything above them is a neural / foundation model, which the brief does not allow.
Post-hoc ensembling of many tuned configurations, k-fold bagging, and smooth (Platt / beta) rather than binning
calibration are the practices the framework already follows. Recent calibration benchmarks also warn that
Platt / isotonic can hurt strong models, which is why the framework keeps Platt only behind a nested check.

## 4. Final check: v5 vs v4 (vs sub2), whole framework, paired outer 5-fold CV

Two fold seeds (0 and 1), every version on the same folds; 4 threads per fit. (The seed-1 runs shared the
server with the stress test, so the time-budget safety net may have trimmed some refinements, equally for both.)

| | P1 AUC ↑ | P2 log loss ↓ | P3 normalized cost ↓ | P3 log loss ↓ |
|---|---|---|---|---|
| sub2 (seed 0 only) | 0.85459 | 0.50857 | 0.7192 | 0.19696 |
| v4, seed 0 / seed 1 | 0.85495 / 0.85425 | 0.50824 / 0.50890 | 0.7180 / 0.7216 | 0.19667 / 0.19791 |
| **v5**, seed 0 / seed 1 | **0.85510** / 0.85411 | **0.50824** / **0.50878** | **0.7146** / 0.7217 | **0.19663** / 0.19792 |
| v4, mean of both seeds | 0.85460 | 0.50857 | 0.7198 | 0.19729 |
| **v5, mean of both seeds** | **0.85461** | **0.50851** | **0.7182** | **0.19728** |

Per fold (seed 0 / seed 1):
- P1 v5: 0.8520 0.8442 0.8613 0.8546 0.8647 / 0.8580 0.8522 0.8539 0.8526 0.8546
- P1 v4: 0.8519 0.8443 0.8611 0.8543 0.8643 / 0.8578 0.8531 0.8537 0.8526 0.8547
- P2 v5: 0.5179 0.5012 0.5137 0.5061 0.5023 / 0.5072 0.5115 0.5102 0.5120 0.5030
- P2 v4: 0.5180 0.5006 0.5143 0.5061 0.5022 / 0.5072 0.5115 0.5103 0.5119 0.5035
- P3 v5 cost: 0.7194 0.7161 0.7272 0.7088 0.7017 / 0.6533 0.7794 0.7289 0.7442 0.7028
- P3 v4 cost: 0.7072 0.7206 0.7294 0.7210 0.7116 / 0.6522 0.7822 0.7244 0.7442 0.7050

**Reading.** v4 is a clear step over sub2 (better on all three problems, P1 on every fold). v5 and v4 are tied
on the public problems (every difference is below the fold-seed noise of about 0.0005); v5 is kept because it is
at least as good everywhere, it never deploys a negative-weight logistic stack, and its cleaner handles
identifier-named categoricals and date columns that the private problems may contain. P3's per-fold normalized
cost ranges from 0.65 to 0.78 for the *same* model, which is the scale of luck to expect on the public holdout.

**Stress test** (`stress_test.py`, log `logs/stress_test_v5.log`): identifier-named categoricals, dates as text,
band labels, timedeltas, tiny data (120 and 300 rows), 1 % positives with a 1:20 cost, 150 features,
300-level categoricals with string labels, 70k rows (subsample path), empty / None metadata, all-categorical
data, odd columns (constant, all-NaN, infinities, numbers as text, booleans) and an accuracy objective: every
case must fit, pickle, reload and predict valid probabilities with unseen categories, a missing column, an extra
column and shuffled column order.

**Known limitation (large data).** Above 60k rows the search runs on a 40k-row subsample and every selected member
is refit on all rows. In the 70k-row stress case the blend gave ExtraTrees weight 0.27, and that refit forest made
the pickled model 441 MB (uncompressed; the kit saves with `compress=3`) and the fit took 25 min on a shared,
oversubscribed server. It works, but it is slow and large. The public problems (10-15k rows) never take this path:
their model files are 3-9 MB and each fits in about 5 minutes on 16 cores.

## 5. Final build

`submission/v5/submission.zip` (16.1 MB): `make_submission.py ../../data` on the Linux server (Python 3.11.16, exact
leaderboard library versions, 16 dedicated cores), log `experiments/v5/logs/make_submission_v5.log`. The kit's
self-check passed on Linux and macOS. (After the build a code comment in `solution.py` was reworded and the file
in the zip replaced; predictions from the zip are identical to the original build, max difference 7e-17.)

| | search (configs) | blend weights | Platt | OOF score | fit |
|---|---|---|---|---|---|
| P1 | LightGBM 32, XGBoost 17, CatBoost 4, ExtraTrees 3, GAM 2 | LightGBM-structured 0.58, CatBoost-monotone 0.25, LightGBM 0.06, GAM 0.05, CatBoost 0.05 | yes | AUC 0.85489 | 343 s |
| P2 | LightGBM 32, XGBoost 18, CatBoost 4, ExtraTrees 3, GAM 2 | GAM 0.35, LightGBM-structured 0.23, XGBoost 0.21, CatBoost 0.17, LightGBM 0.03 | no (nested check) | log loss 0.50862 | 292 s |
| P3 | LightGBM 32, XGBoost 18, CatBoost 4, ExtraTrees 3, GAM 2 | LightGBM-structured 0.34, CatBoost 0.33, XGBoost 0.17, XGBoost-structured 0.06, ExtraTrees 0.06, LightGBM 0.04 | yes | log loss 0.19651 | 273 s |

P3 threshold: **0.1211** (`p3_threshold.json`). On the out-of-fold predictions the cost at this threshold is 436.1
per 1000 cases vs 436.3 at the Bayes value 1/11 = 0.091 (the cost curve is flat between them), 601 for "alert
nobody" and 940 for "alert everybody", i.e. an OOF normalized cost of about 0.725.
