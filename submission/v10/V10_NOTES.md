# Submission v10 (framework version 12): status, changes and evidence

> **Status: DRAFT.** Whole-framework comparison pending.

v10 = v8 + a **sparse GAM with kinked interactions**: an L1-penalised logistic regression on a standardised
basis of hinges at 15 interior quantiles per numeric column, products of the top measured numeric pairs,
category-dependent slopes for the top category × numeric pairs, and *kinked* interaction terms
(`max(a − k, 0)·b`, `a·max(b − k, 0)`, and category-specific hinges) at four quantile knots. (v9, the plain sparse
hinge GAM, was an intermediate step and is superseded.) Scripts: `experiments/v7/zoo.py` (`gam_l1_f`),
`blend_zoo.py`; logs `experiments/v7/logs/l1*_local.out`, `blend_l1_*`, `blend_ph*_*`.

## 1. What the shapes look like

Diagnostic GAMs on the honest zoo protocol (two 5-fold partitions, inner early stopping):

| GAM | P1 AUC ↑ | P2 log loss ↓ | P3 log loss ↓ |
|---|---|---|---|
| framework L2 hinge GAM (7 knots) | 0.84734 | 0.51011 | 0.20041 |
| L1, hinges + steps at 15 quantiles | 0.84869 | 0.51437 | 0.19985 |
| L1, steps only | 0.84193 | 0.51720 | 0.20165 |
| **L1, hinges only** | **0.85075** | 0.50840 | 0.19822 |
| L1 hinges + kinked interactions (4 pairs, 2 category kinks) | 0.85102 | **0.50746** | 0.19773 |
| same with 6 kinked pairs | 0.85057 | 0.50749 | **0.19735** |
| best trees (for reference) | 0.85397 | 0.50985 | 0.19722 |

- The effects are **piecewise linear with a few sharp kinks**, not steps: step bases hurt even under L1, while L1
  on a dense hinge basis (= trend filtering: few kinks survive) gains 0.0034 AUC on P1 over the smooth L2 GAM.
- The interactions are **kinked** too: "above a threshold of A, the slope of B changes". With kinked interaction
  terms the sparse GAM is the best single model on P2 by a wide margin and matches the best trees on P3.
- Knot placement matters for the sparse GAM: hinges at the extreme quantiles (1/30, 29/30) rest on few rows and
  add noise under L1 (P2 0.50803 vs 0.50746 with interior knots 1/16 … 15/16).
- At a lower penalty (C 0.05) the 6-pair kinked GAM is the best single P3 model of any kind (0.19703 vs 0.1972 for
  the best trees).
- Rejected: steps / AND-rule / hinge × hinge bases, tail knots and squared "walls" (all over-fit), a relaxed
  lasso (L2 refit on the selected terms: worse everywhere, the L1 shrinkage is useful), a GAM-then-boost hybrid
  (no gain over the sparse GAM), more than 6 kinked pairs.
- Where trees still win (P1): a regression tree on the tree − GAM logit difference splits only on `cycle_time`,
  whose U-shape has very steep walls beyond the outer knots; tail knots and squared walls did not fix it, so the
  ensemble leaves that to the trees.

## 2. Ensemble value (honest half-split study; change vs the framework-like library)

| library | P1 AUC | P2 log loss | P3 log loss | P3 cost @ 1/11 |
|---|---|---|---|---|
| v8-like (pair GAM, LightGBM lr 0.01, category-slope GAM) | +0.00033 | −0.00084 | −0.00044 | −0.0032 |
| + plain sparse hinge GAM (v9) | +0.00045 | −0.00101 | −0.00045 | −0.0028 |
| + sparse GAM with kinked interactions, 4 pairs | +0.00051 | −0.00188 | −0.00065 | −0.0037 |
| + kinked GAMs with 4 and 6 pairs | +0.00049 | −0.00192 | −0.00082 | −0.0028 |
| **+ the 6-pair kinked GAM at C 0.05 as well (final v10)** | **+0.00048** | **−0.00202** | **−0.00099** | **−0.0072** (se 0.0011) |

## 3. What v10 changes (on top of v8)

`SplineLogit(penalty="l1")`: all design columns standardised, `liblinear` L1 fit, interior hinge knots;
`pair_hinge` / `cat_hinge`: kinked interaction terms for the first measured numeric / category pairs (knots at
the 20/40/60/80 % quantiles). GAM ladder gains two sparse rungs (C 0.1, 15 knots, 4 products + 2 category slopes
with 4 kinked pairs and 2 category kinks; and the same with 6 pairs); the refinement samples sparse variants;
the library keeps up to 4 GAMs, including the best of each penalty. Final ladder: no pairs (C 0.05 / 0.2), 2
and 4 products, 4 products + 2 slopes, 2 products + 1 slope, and three sparse rungs (4 kinked pairs at C 0.1; 6 at
C 0.1 and at C 0.05). The GAM family's time share is raised to 15 % of the budget so the ladder is never cut on a
slow machine (it takes about a minute on P3).
