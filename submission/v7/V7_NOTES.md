# Submission v7 (framework version 9): status, changes and evidence

> **Status: final framework.** `submission.zip` in this folder is the build to upload (see §5).

Starting point: the public leaderboard (v5 build: P1 0.8617, P2 0.5034, P3 0.6800) put us 3rd / 4th / 8th of the
top teams; the leaders were 0.8621 / 0.5023 / 0.6636. P1 and P2 differences of that size are within the holdout's
noise, but P2 (log loss) is where a systematic gain was most plausible, because ensembling diverse models had
helped P2 far more than P1 or P3.

## 1. Method: an honest model zoo (`experiments/v7/zoo.py`, `blend_zoo.py`)

42 candidate models, each fitted on two 5-fold partitions of every problem (v6 cleaning, so noisy copies are
already removed). Every fold model early-stops on an inner 15 % split of its own training part, never on the
fold it predicts (the v4 zoo stopped on the predicted fold, which flatters every member). Ensemble value is
measured honestly: 40 random 50/50 row splits; the framework's combiner (bagged Caruana selection + Platt) is
fitted on half A with fold-seed-0 predictions and scored on half B with fold-seed-1 predictions. Reported:
change versus a framework-like library (11 members: structured / unconstrained LightGBM depth 2–4, XGBoost
depth 2–3, CatBoost monotone depth 3 and depth 4, hinge GAM C 0.05 / 0.2, ExtraTrees).

## 2. Findings

### 2.1 Single models (mean of two fold seeds)

| | P1 AUC ↑ | P2 log loss ↓ | P3 log loss ↓ |
|---|---|---|---|
| structured LightGBM (lr 0.03, framework rung) | 0.85330 | 0.51040 | 0.19740 |
| **structured LightGBM, lr 0.01** | **0.85397** | 0.50985 | 0.19736 |
| hinge GAM (framework) | 0.84736 | 0.51009 | 0.20040 |
| **hinge GAM + 2 measured pair products** | 0.84721 | **0.50848** | 0.20072 |
| hinge GAM + 4 / 6 pair products | 0.84708 / 0.84684 | 0.50857 / 0.50864 | 0.19983 / 0.19968 |
| GAM, all design columns standardised (no pairs) | 0.84757 | 0.51112 | 0.20172 |
| linear-tree LightGBM, lr 0.01 | 0.85381 | 0.50959 | **0.19714** |
| EBM-style (mains, then 8 pairs on the residual) | 0.85273 | 0.51145 | 0.19885 |
| CatBoost monotone depth 3 | 0.85210 | 0.51040 | 0.19804 |
| class-weighted LightGBM (pos. weight 3 / 10) | 0.85169 / 0.76938 | 0.588 / 0.677 | 0.223 / 0.287 |

The P2 gain of the pair GAM comes from the product terms, not from scaling: standardising the design without
pairs makes P2 worse (0.51112). The top measured pairs on P2 are `operating_margin × debt_service_ratio` and
`business_age_years × operating_margin`: P2 behaves like an additive model plus a couple of bilinear terms, which
trees can only approximate with many splits.

### 2.2 Ensemble value (honest half-split study; change vs the framework-like library)

| added to the library | P1 AUC | P2 log loss | P3 log loss | P3 cost @ 1/11 |
|---|---|---|---|---|
| GAM + 2 pair products | −0.00002 | **−0.00082** (se 0.00005) | 0.00000 | +0.0002 |
| GAM + 4 pair products | −0.00001 | **−0.00084** (se 0.00004) | −0.00005 | −0.0001 |
| LightGBM lr 0.01 | **+0.00035** (se 0.00003) | −0.00002 | 0.00000 | +0.0002 |
| linear-tree LightGBM | +0.00022 (se 0.00003) | +0.00003 | 0.00000 | **+0.0018** (se 0.0006) |
| pair GAM + LightGBM lr 0.01 | **+0.00034** | **−0.00083** | 0.00000 | +0.0002 (se 0.0003) |
| ... + linear-tree LightGBM lr 0.01 | +0.00040 | −0.00087 | −0.00002 | **+0.0029** (se 0.0006) |
| EBM-style, CatBoost Langevin / depth 6, RF, HGB, extra-trees LightGBM, class-weighted models | within ±0.00004 | within ±0.00003 | within ±0.00003 | up to +0.0016 |
| all 34 zoo members | +0.00015 | −0.00058 | +0.00017 | +0.0033 |

- **Adopted:** GAM with measured pair products (P2 −0.0008) and the learning-rate-0.01 structured LightGBM (P1
  +0.00035); together neutral on P3.
- **Rejected:** linear-tree LightGBM. Small extra P1/P2 gains, but it raised P3's decision cost by 0.003 at an
  unchanged log loss (it reorders cases around the decision threshold).
- **Rejected:** class weighting (no cost gain once recalibrated), EBM-style models, more CatBoost variants,
  random forests: no ensemble value.
- A bigger library is not better: all 34 members together are worse than the focused library (selection over
  many near-duplicates over-fits even when bagged).

## 3. What v7 changes (on top of v6)

| # | Change |
|---|---|
| 1 | `SplineLogit` takes `pairs`: products of the standardised numeric columns of the top measured pairs (from the same nested structure detection the boosters use). GAM ladder: C 0.05 / 0.2 without pairs, C 0.05 with 2 and with 4 pairs; two random refinements also sample the number of pairs (0–6). The library keeps the best GAM with and the best without pairs (quota 3). |
| 2 | LightGBM ladder gains a structured depth-2 rung at learning rate 0.01; LightGBM may grow up to 8000 trees and its early-stopping patience scales with 0.03 / learning rate. |

## 4. Final comparison (whole framework)

**Public problems**, paired outer 5-fold CV, fold seeds 0 and 1 (tag `v7c` in `experiments/v5/logs/outer.log`):

| | P1 AUC ↑ | P2 log loss ↓ | P3 normalized cost ↓ | P3 log loss ↓ |
|---|---|---|---|---|
| framework 7 (v5, the leaderboard build) | 0.85461 | 0.50851 | 0.7182 | 0.19728 |
| framework 8 (v6) | 0.85474 | 0.50828 | 0.7183 | 0.19729 |
| **framework 9 (v7)** | **0.85481** | **0.50752** | **0.7145** | **0.19722** |

Per seed (v7 vs v6): P1 0.85533 / 0.85428 vs 0.85526 / 0.85421; P2 0.50739 / 0.50765 vs 0.50807 / 0.50849 (better on
9 of 10 folds); P3 cost 0.7159 / 0.7131 vs 0.7186 / 0.7180.

**Synthetic generalisation suite** (16 problems, regret vs the Bayes-optimal score on 20k fresh rows):

| mean regret | AUC problems (6) | log-loss problems (6) | cost problems (4) |
|---|---|---|---|
| framework 7 (v5) | 0.01484 | 0.00900 | 0.0357 |
| framework 8 (v6) | 0.01449 | **0.00875** | 0.0328 |
| **framework 9 (v7)** | **0.01416** | 0.00879 | **0.0223** |

A first v7 candidate that also had a linear-tree LightGBM rung raised the regret on the 1.4 %-positive problem
from 0.016 to 0.036; without that rung v7 is at 0.0161, so the rejection in §2.2 is confirmed on unseen data too.

Rejected after the zoo study as well: a GAM-then-boost hybrid (GAM logits as the booster's starting score). It was
the best single P2 model (0.50830), but added only −0.00007 to the ensemble (≈ 1.4 standard errors).

## 5. Final build

`submission/v7/submission.zip` (18.2 MB): official `make_submission.py ../../data` on the Linux server (Python
3.11.16, exact leaderboard versions, 20 dedicated cores), log `experiments/v7/logs/make_submission_v7.log`. The
kit's self-check passed on Linux and macOS; the `solution.py` in the zip is byte-identical to this folder's.

| | search (configs) | blend weights | Platt | OOF score (v6 build / v5 build) | fit |
|---|---|---|---|---|---|
| P1 | GAM 6, LightGBM 32, XGBoost 18, CatBoost 4, ExtraTrees 3 | structured LightGBM 0.66, monotone CatBoost 0.16, structured XGBoost 0.14, GAM 0.04 | yes | AUC 0.85510 (0.85504 / 0.85489) | 303 s |
| P2 | same | **GAM with pair terms 0.49**, CatBoost 0.27, structured LightGBM 0.18, XGBoost 0.06 | no (nested check) | log loss 0.50745 (0.50833 / 0.50862) | 241 s |
| P3 | same | structured XGBoost 0.23, monotone CatBoost 0.17, CatBoost 0.16, LightGBM 0.17, XGBoost 0.12, GAM with pair terms 0.11, ExtraTrees 0.02 | yes | log loss 0.19649 (0.19658 / 0.19651) | 254 s |

P3 threshold 1/11 = 0.0909; OOF cost 432.3 per 1000 cases (about 0.719 normalized).

**Stress test** (`experiments/v7/logs/stress_test_v7.log`): all 12 cases pass. In the 70k-row case the blend is the
pair-term GAM alone; that synthetic target is additive plus a product of two columns, which this GAM represents
exactly.
