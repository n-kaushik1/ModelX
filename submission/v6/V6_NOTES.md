# Submission v6 (framework version 8): status, changes and evidence

> **Status: final framework.** `submission.zip` in this folder is the build to upload (see §4).

Naming: `submission/v6/` = framework version 8; `submission/v5/` = framework 7 (the previous final). Scripts:
`experiments/v6/` (synthetic suite, threshold study, pruning test) and `experiments/v5/` (outer CV harness, drop
tests); raw logs in their `logs/` folders.

## 1. What v6 changes

| # | Change | Evidence |
|---|---|---|
| 1 | **Measured redundancy: noisy copies dropped, twin readings averaged.** For numeric pairs with \|corr\| ≥ 0.9 in the same units (equal means within 0.1 sd, sd ratio 0.8–1.25), a measurement-error decomposition: if `b = a + e`, then var(a) = cov(a, b), so the clean column has a noise share (var − cov)/var ≈ 0 and the copy a positive one. One share ≤ ⅓ of the other → the noisier column is dropped; both shares positive and similar → both are kept and their noise-weighted average is added. No target, no names involved. | §2.1–2.3 |
| 2 | **Decision threshold = c_fp / (c_fp + c_fn) on the calibrated probabilities**, replaced by the empirical OOF threshold only if it is better *out of sample*: fitted on the in-bag rows of a bootstrap round and scored against the Bayes threshold on the out-of-bag rows, it must win in ≥ 95 % of 400 rounds (evidence of miscalibration at the decision boundary). | §2.4 |
| 3 | The GAM (and plain logistic regression) always get their 2 random refinements, so the search no longer depends on machine speed. | reproducibility |
| 4 | Forest leaves per tree are bounded (min leaf ≥ rows / 8000): roughly halves a forest refit on 70k rows. No effect below 40k rows. | 70k-row stress case: 441 MB model in framework 7 |
| 5 | **Linear-model fits run with at most 4 BLAS threads** (`threadpoolctl`, a scikit-learn dependency in the pinned requirements), and **every model family gets its own time window** (unused time still rolls over). | §4: the first v6 build ran the GAM with 32 BLAS threads on a shared server; the GAM stage became so slow that the global time box cut P1's boosting search to 4 of 32 LightGBM configurations. With the cap the GAM fits are 3–6× faster even on an idle machine, and a slow family can no longer starve the others. Results are unchanged (P1 / P2 outer folds reproduce exactly). |

## 2. Analysis and experiments

### 2.1 What the data says about duplicated columns

Each public problem has columns that are noisy copies of another column in the same units
(`temperature_sensor_2` ≈ `process_temperature`, `material_density_estimate`, `utilization_3m_average`,
`account_balance_last_month`, `utilization_15m_average`, ...). The covariance decomposition shows the original is
noise-free (share 0.000–0.002) and the copy carries 4–7 % extra variance: the copy holds no information the
original lacks. That is also why averaging the pair (tried in framework 3) did not help. P3's `packet_loss_rate` /
`packet_loss_5m` pair looks like two noisy readings (shares 0.057 / 0.050) and is treated as twins.

### 2.2 Oracle test (`experiments/v5/exp_drop.py`, reference LightGBM, 5-fold CV × 3 fold seeds)

| | P1 AUC ↑ | P2 log loss ↓ | P3 log loss ↓ |
|---|---|---|---|
| all columns | 0.85402 | 0.50985 | 0.19775 |
| copies dropped | **0.85433** | **0.50918** | **0.19725** |
| originals dropped instead (sanity check) | 0.83693 | 0.51226 | 0.19837 |
| distractor columns dropped | 0.85433 | 0.50975 | 0.19892 |

Dropping copies helped on all three problems and every fold seed; dropping distractors was inconsistent (it
changes the measured interaction / monotone constraints), so only copies are handled.

### 2.3 Detector validation

On the public data the detector drops exactly the same-unit copies and keeps every original; on 32 synthetic
problems with known copies and twin pairs it is exact on 31 and never drops an original or a twin (one twin pair is
left alone). Two pitfalls found while building it: (i) a CV comparison of one-column models could not tell an
original from its copy (it dropped originals on P2 / P3), so the decision uses the covariance structure instead;
(ii) computing a column's noise share on a trimmed range of that column biases the share of a noisy column towards
zero, so the shares use all rows.

**Rejected: supervised pruning of derived columns** (`experiments/v6/exp_prune.py`). Columns ≥ 90 % explained by
other columns (e.g. `revenue_per_employee` = revenue / employees) were dropped greedily when a quick CV said so.
Nested inside the folds it made P2 *worse* (0.50875 → 0.51006): it sometimes dropped both `revenue_per_employee`
and `annual_revenue`. The single-model "gain" seen when choosing the columns with hindsight was selection bias.

### 2.4 Threshold rule (`experiments/v6/exp_threshold_sim.py`)

30 synthetic cost problems (cost ratios 1:2, 1:5, 1:10, 1:20, 3:1, 5:1; 3k–20k rows; 3–40 % positives), 5-fold
bagged booster + Platt, 20k-row test sets:

| rule | mean normalized cost |
|---|---|
| empirical (framework 7: smoothed OOF curve + bootstrap median) | 0.8473 |
| **Bayes c_fp/(c_fp+c_fn) on calibrated probabilities** | **0.8456** |
| logit-average of the two | 0.8455 |
| true probabilities, Bayes threshold (unattainable) | 0.8106 |

The empirical rule is worse by 0.0017 (≈ 2 standard errors; worse on 19 of 30). On the public P3 outer-CV
predictions the Bayes threshold also beats the learned one for every framework version (mean of two fold seeds:
v4 0.7152 vs 0.7198, v5 0.7160 vs 0.7182, v6 0.7196 vs 0.7281).

A first version of the threshold guard compared the two thresholds on the same rows the empirical one was fitted
on, so the empirical threshold "won" in every resample (in-sample optimism). The out-of-bag version fixes that: on
calibrated data the empirical rule wins only about 30 % of the rounds and the Bayes threshold is used.

## 3. Final comparison: paired outer CV (public) and synthetic suite (generalisation)

**Public problems**, whole framework, paired outer 5-fold CV, mean of fold seeds 0 and 1 (`experiments/v5/`;
tags `v5new` = framework 7, `v6` = copy detection only, `v6b` = this folder):

| | P1 AUC ↑ | P2 log loss ↓ | P3 normalized cost ↓ | P3 log loss ↓ |
|---|---|---|---|---|
| framework 7 (v5) | 0.85461 | 0.50851 | **0.7182** | **0.19728** |
| + redundancy (v6 tag) | **0.85474** | **0.50828** | 0.7281 | 0.19724 |
| **+ Bayes threshold, deterministic GAM search, bounded forests (final v6)** | **0.85474** | **0.50828** | 0.7183 | 0.19729 |

Per fold, seed 0 / seed 1 (final v6): P1 0.8523 0.8443 0.8616 0.8549 0.8647 / 0.8582 0.8519 0.8535 0.8523 0.8556;
P2 0.5180 0.5005 0.5139 0.5062 0.5017 / 0.5069 0.5111 0.5102 0.5120 0.5023; P3 cost 0.7289 0.7294 0.7228 0.7110
0.7011 / 0.6478 0.7456 0.7517 0.7420 0.7028 (threshold 1/11 on every fold).

P1 improves on 8 of 10 folds and P2 on both seeds; P3's probabilities are unchanged (log loss tied) and its cost
is within noise of framework 7 (the learned threshold made framework 7's cost look 0.002 better than its own
predictions at 1/11, and v6's look 0.009 worse: threshold noise, which the Bayes rule removes).

**Synthetic suite** (`experiments/v6/simgen.py`, `sim_run.py`): 16 problems in the style of the public data
(1.5k–80k rows, 1.4–50 % positives, AUC / log loss / cost objectives with FN:FP from 1:5 to 20:1 and 5:1 the other
way, strong interactions, numeric × category effects, high-cardinality categoricals, 100 noise columns, heavy
missing-not-at-random data, noisy copies and twin readings). Each is fitted once and scored on 20k fresh rows
against the true probabilities (regret = gap to the Bayes-optimal score, lower is better):

| mean regret | AUC problems (6) | log-loss problems (6) | cost problems (4) |
|---|---|---|---|
| framework 7 (v5) | 0.01484 | 0.00900 | 0.0357 |
| v6 tag (redundancy) | 0.01469 | 0.00880 | 0.0357 |
| **final v6** | **0.01449** | **0.00875** | **0.0328** |

**Stress test** (`experiments/v5/stress_test.py`, logs `experiments/v6/logs/stress_test_v6b.log` and `stress_test_v6final.log` for the final code): all 12 cases pass
(identifier-like and date columns, tiny / 1 %-positive / 150-feature / 300-level / 70k-row data, empty metadata,
odd columns, accuracy and F1 objectives); thresholds follow the costs (1:5 → 0.167, 1:20 → 0.048, 3:1 → 0.75).
(In this run the 70k-row blend did not select ExtraTrees, so its 3.5 MB model says nothing about the forest
bound; at 70k rows the bound raises the forest's minimum leaf size from 5 to 9, roughly halving a refit forest
compared with framework 7's 441 MB case. Large tables remain the slow, large path of the framework.)

## 4. Final build

`submission/v6/submission.zip` (16.4 MB): official `make_submission.py ../../data` on the Linux server (Python 3.11.16,
exact leaderboard library versions, 32 cores, otherwise idle), log `experiments/v6/logs/make_submission_v6.log`.
The kit's self-check passed on Linux and macOS; the `solution.py` in the zip is byte-identical to this folder's.

| | search (configs) | blend weights | Platt | OOF score | fit |
|---|---|---|---|---|---|
| P1 | LightGBM 32, XGBoost 17, CatBoost 4, ExtraTrees 3, GAM 4 | LightGBM-structured 0.66, CatBoost-monotone 0.15, XGBoost-structured 0.15, GAM 0.02, CatBoost 0.02 | yes | AUC 0.85504 (framework 7 build: 0.85489) | 336 s |
| P2 | LightGBM 32, XGBoost 18, CatBoost 4, ExtraTrees 3, GAM 4 | GAM 0.35, CatBoost 0.33, LightGBM-structured 0.23, XGBoost 0.10 | yes | log loss 0.50833 (0.50862) | 246 s |
| P3 | LightGBM 32, XGBoost 18, CatBoost 4, ExtraTrees 3, GAM 4 | LightGBM 0.27, XGBoost 0.36, CatBoost 0.34, ExtraTrees 0.03 | yes | log loss 0.19658 (0.19651) | 282 s |

Columns dropped as noisy copies: P1 `temperature_sensor_2`, `material_density_estimate`; P2
`account_balance_last_month`, `utilization_3m_average`; P3 `utilization_15m_average` (and `packet_loss_rate` /
`packet_loss_5m` averaged as twin readings).

P3 threshold **0.0909 = 1/11** (`p3_threshold.json`). The empirical OOF optimum (0.098) beat it out of sample in
61 % of bootstrap rounds, short of the 95 % needed to override; OOF cost at 1/11 is 432.3 per 1000 cases (601 for
"alert nobody", 940 for "alert everybody"), i.e. an OOF normalized cost of about 0.719.

The first v6 build (before the BLAS cap / per-family windows) had run P1's boosting search on only 4 of 32
LightGBM configurations because of the slow GAM stage; it was discarded and rebuilt.
