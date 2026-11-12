# Submission v12 (framework version 14): status, changes and evidence

> **Status: final framework.** `submission.zip` in this folder is the build to upload (see §3).

Two candidate changes on top of v11 (framework 13), each first measured in the honest zoo / ensemble study and
then in the whole framework. **Only change 1 is in the final v12** (this folder, tag `v12a`); change 2 (tag `v12b`)
tied in the whole framework and was dropped.

| # | Change | Evidence |
|---|---|---|
| 1 | **Second GAM-then-boost hybrid**, built on P3's best sparse GAM (6 kinked pairs, C 0.05): the strongest single model on P2 (log loss 0.50683) and P3 (0.19670). Hybrid ladder: 2 rungs, library quota 2. | Zoo ensemble study on top of the v11 library: P2 −0.00025, P3 cost −0.0015, P1 +0.00008. Whole framework (v12a vs v11, 2 fold seeds): P1 0.85508 vs 0.85503, P2 **0.50636** vs 0.50659 (both seeds), P3 cost **0.7087** vs 0.7098, log loss 0.19651 vs 0.19654. |
| 3 | **Default `time_budget` 300 → 600 s.** The search is a fixed number of configurations; the budget only cuts it on a slow machine. | A build on a 10-core laptop CPU with 300 s cut P1's LightGBM search to 12 of 32 configurations (the larger GAM / hybrid stages leave the boosters less time). With 600 s the full search runs there too; nothing changes on a fast machine. This also protects the private retraining on an unknown machine. |
| 4 | **Ladder rungs are never cut by a family's time window**; only the random refinements absorb time pressure (a guard stops a run only after 6 × `time_budget`). | Independent review (`docs/review_v12_2026-11-12.md` §4): at 4 threads the per-family hard window cut ladder rungs, so on P3 the second hybrid (the v12 change) never ran. The validated library members all come from the ladder. |
| 5 | **Numeric text with a few placeholder tokens** ("n/a", "?", "-") is coerced to numeric when ≥ 80 % of the values parse and the rest are ≤ 5 distinct tokens (which become missing values). | Review probe (§5): a numeric column with every 7th value "n/a" became a 2,000-level categorical capped at 30 levels, losing its information. No effect on the public data. |
| 2 (dropped) | GAM members refit once on all rows instead of k-fold bagged; boosters and hybrids keep the bag. | Outer 5-fold × 2 seeds, fixed configurations: sparse GAM refit vs bag P1 AUC 0.85036 vs 0.84969, P2 0.50811 vs 0.50805, P3 **0.19739 vs 0.19794**; the hybrid is better bagged (P2 0.50622 vs 0.50686). A regularised GAM gains more from 25 % more rows than from averaging. **But in the whole framework v12b ties v12a** (P1 0.85506 vs 0.85508, P2 0.50638 vs 0.50636, P3 cost 0.7087 vs 0.7087, log loss 0.19647 vs 0.19651): the blend weights and the calibration are learned on bagged OOF predictions, so the deployed members stay bagged. |

Tested and rejected in this round:
- **FAST interaction ranking** (residual 2-D boosters on top of the additive model, Lou et al. 2013) as the source
  of the GAM's pairs: much worse than the booster-gain ranking (P2 0.50965 vs 0.50746, P3 0.1995 vs 0.1977).
- Knots spaced by value instead of by quantile, dense / tail knots for the most important columns: no gain
  (P1's steep `cycle_time` walls stay with the trees).
- Hybrid variants (depth-3 residual booster, lr 0.02, min leaf 50, CatBoost on the residual): within noise as
  single models and no ensemble value on top of the v12 library (±0.00006).
- RuleFit (leaf indicators of a small boosted ensemble + the sparse GAM basis, one L1 fit): far worse everywhere
  (P2 0.533, P3 0.213) -- rules grown on the same rows carry in-sample optimism that L1 cannot undo.

## 2. Final comparison

**Public problems**, paired outer 5-fold CV, mean of fold seeds 0 and 1:

| | P1 AUC ↑ | P2 log loss ↓ | P3 normalized cost ↓ | P3 log loss ↓ |
|---|---|---|---|---|
| framework 7 (v5, on the leaderboard) | 0.85461 | 0.50851 | 0.7182 | 0.19728 |
| framework 10 (v8) | 0.85484 | 0.50757 | 0.7143 | 0.19707 |
| framework 13 (v11) | 0.85503 | 0.50659 | 0.7098 | 0.19654 |
| **framework 14 (v12)** | **0.85508** | **0.50636** | **0.7087** | **0.19651** |

**Synthetic suite** (mean regret vs the Bayes-optimal score; lower is better):

| | AUC problems (6) | log-loss problems (6) | cost problems (4) |
|---|---|---|---|
| framework 10 (v8) | 0.01416 | 0.00824 | 0.0235 |
| framework 13 (v11) | 0.01418 | 0.00746 | 0.0239 |
| **framework 14 (v12)** | **0.01402** | **0.00742** | **0.0225** |

v12 is the best version on every public problem and on every objective type of the synthetic suite.

## 3. Final build

`submission/v12/submission.zip` (9.1 MB): official `make_submission.py ../../data` on the Linux server (Python
3.11.16, exact leaderboard library versions, 30 dedicated cores), log
`experiments/v12/logs/make_submission_v12_linux.log`. The kit's self-check passed on Linux and macOS; the zip's
`solution.py` is byte-identical to this folder's. **Every search ran in full** on all three problems (GAM 11,
hybrids 2, LightGBM 32, XGBoost 18, CatBoost 4, ExtraTrees 3).

| | blend weights | Platt | OOF score (v11 build) | fit |
|---|---|---|---|---|
| P1 | structured LightGBM 0.42, hybrids 0.39, monotone CatBoost 0.10, structured XGBoost 0.08 | yes | AUC 0.85569 (0.85548) | 459 s |
| P2 | **hybrids 0.63**, CatBoost 0.21, sparse GAM 0.12, structured LightGBM 0.04 | no | log loss 0.50615 (0.50624) | 317 s |
| P3 | **hybrids 0.36, sparse GAMs 0.31**, XGBoost 0.26, CatBoost 0.05, LightGBM 0.03 | no | log loss **0.19529** (0.19538) | 445 s |

P3 threshold 1/11; OOF cost 425.6 per 1000 cases (v11 build 427.6, v5 build 436.1).

Build history: a first build on a 10-core laptop CPU with the old `time_budget` (300 s) cut P1's LightGBM search to
12 / 32 configurations; a second one (600 s, while that laptop was busy) still cut some P1 *refinements* (the
ladder ran in full). Both were discarded; this one ran on a dedicated server. On an idle laptop at 4 threads the
final code runs the full P3 search in 418 s (`experiments/v12/logs/time_p3_nj4_final.log`).

**Stress test** (`experiments/v12/logs/stress_test_v12_final.log`, final code): all 12 cases pass. **Review probes**
(`experiments/v12/logs/probe_v12_final.log`, 12 more cases: labels {1,2} / {−1,+1} / strings / booleans, unique
integer ids, a 2,500-level categorical, numeric columns with "n/a" tokens, integer column names, ...): all pass;
the "n/a" column is now numeric (AUC on that probe 0.834 vs 0.822).
