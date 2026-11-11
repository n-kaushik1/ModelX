# ModelX: a unified ML framework for binary classification

ModelX is a course project for DA5401 (Data Analytics Lab). It is **one reusable framework** that takes any tabular
binary-classification dataset plus a description of how it will be scored. It then cleans the data, compares
several model families, tunes and blends them, calibrates the probabilities and, when costs are given, picks a
decision threshold, all without any dataset-specific code.

It is evaluated on three public problems (manufacturing quality, SME credit delinquency, telecom fault
escalation) and later on three hidden ones.

---

## Current status

| What | Where |
|---|---|
| **Zip to upload to the leaderboard** | [`submission/sub2/sub2.zip`](submission/sub2/sub2.zip) (version 5) |
| Submission v4 (framework version 6): **in progress, not yet a submission** | [`submission/v4/`](submission/v4/); status, changes and evidence in [`V4_NOTES.md`](submission/v4/V4_NOTES.md) |
| Previous zips (v3: `experiments/variants/sub2_v3.zip`, v2) | [`submission/improved/submission.zip`](submission/improved/submission.zip) |
| Fallback / reference zip | [`submission/baseline/submission.zip`](submission/baseline/submission.zip) |
| Framework code | [`submission/sub2/solution.py`](submission/sub2/solution.py) |
| What was tried and why | [`docs/experiment_notes.md`](docs/experiment_notes.md) |
| How the framework works | [`docs/framework_design.md`](docs/framework_design.md) |

All zips were built with the official `make_submission.py` in the exact leaderboard environment and passed its
self-check.

**Local results** (mean of 3 stratified 80/20 splits, seeds 42/1/2; not leaderboard scores):

| Problem | Metric | Baseline | Improved (v2) | **sub2 (v4)** |
|---|---|---|---|---|
| P1 Manufacturing | ROC-AUC (higher is better) | 0.8476 | 0.8458 | **0.8492** |
| P2 Credit | Log loss (lower is better) | 0.5116 | 0.5099 | **0.5087** |
| P3 Telecom | Normalized cost (lower is better) | 0.6837 | 0.6824 | **0.6802** |

Version 4 (`submission/sub2/`) is better than v2 on every split of P1 and P2 and on 2 of 3 splits of P3.
Compared with v2 it:
- tunes every boosting family with a *complexity ladder* (tree depth 1, 2, 3, 4, 6, number of trees by early
  stopping) followed by random refinement around the best depth: all three public problems turn out to be mostly
  additive with hinge / U-shaped effects, where depth-1/2 boosting beats the 15-31-leaf trees v2 searched;
- **measures the structure of the training data and uses it as model constraints**: a quick depth-2 model ranks
  the feature pairs that interact, a quick depth-1 model gives each feature's additive shape and its monotone
  direction where the shape is monotone; the boosting candidates are then searched with and without
  `interaction_constraints` / `monotone_constraints` (nested inside every fold, so nothing leaks);
- re-scores the best configurations on fresh folds with a fixed tree count, then blends them in logit space with
  greedy (Caruana) selection; for the cost problem the members are selected by log loss, which transfers better
  than the noisy best-threshold cost;
- adds a spline-logistic GAM candidate and makes CatBoost a regular candidate;
- respects CPU affinity limits and scales the tree count when refitting on all rows.

Full details and the supporting experiments (model families, Optuna, feature pairs, stacking, threshold transfer)
are in [`docs/experiment_notes.md`](docs/experiment_notes.md).

**Key dates:** leaderboard closes **28 Nov 2026, 23:59 IST**; final report due **29 Nov**.

---

## Repository layout

```
ModelX/
├── README.md                  ← you are here
├── CLAUDE.md                  rules and commands for Claude Code sessions (local or cloud)
├── info.md                    leaderboard page: metrics, submission format, rules, library versions
├── public_data.zip            the three development CSVs (unzipped into data/ by the setup script)
├── starting_kit.zip           the official starting kit, unchanged
├── submission/
│   ├── v4/                    submission v4 = framework version 6, in progress (k-fold bagging, library ensemble; see V4_NOTES.md)
│   │   ├── solution.py, V4_NOTES.md, and the official kit files
│   ├── sub2/                  ← the current submission (versions 3-5: complexity ladder, measured-structure constraints, logit blending, GAM)
│   │   ├── solution.py, sub2.zip (version 5 build), and the official kit files
│   ├── improved/              version 2, kept for comparison
│   │   ├── solution.py        the framework (StudentSolution)
│   │   ├── submission.zip     built zip, ready to upload
│   │   └── make_submission.py, local_test.py, problems.py, requirements.txt   (official kit files)
│   └── baseline/              the first version, kept frozen for comparison (same structure)
├── experiments/               scripts behind the numbers in docs/experiment_notes.md
│   ├── eval_splits.py         repeated 80/20 evaluation with joblib save/load (main comparison tool)
│   ├── v4/                    submission-v4 experiments: feature screen, model variants, bagging, ensembles, outer CV (+ logs/)
│   ├── sub2/                  v3 experiments: model families / depth ladder, feature pairs, Optuna, stacking
│   ├── robust_test.py         unseen categories, missing values, shuffled/missing columns, fresh-process load
│   ├── exp_families.py        LightGBM vs XGBoost vs CatBoost
│   ├── exp_threshold.py       P3 threshold strategies
│   ├── exp_featsel.py         nested shadow-feature selection (rejected)
│   ├── run_local_eval.py      original harness (old metadata format)
│   ├── variants/              earlier versions of solution.py
│   └── logs/                  raw output of every run
├── docs/
│   ├── project_instructions.md   the course brief
│   ├── framework_design.md       pipeline, step by step
│   └── experiment_notes.md       results, decisions, observations (input for the final report)
└── scripts/
    ├── setup_env.sh           environment setup for Linux / macOS / Claude cloud
    └── setup_env.ps1          environment setup for Windows
```

---

## Setup

You need **Python 3.11**. Models only load on the leaderboard if they were saved with its exact library versions
(numpy 1.26.4, pandas 2.3.1, scikit-learn 1.7.1, lightgbm 4.6.0, xgboost 3.1.1, catboost 1.2.10). The setup
script creates `.venv/` with those versions and unzips the data into `data/`.

**Windows (PowerShell), from the repo root**

```powershell
git clone https://github.com/n-kaushik1/ModelX.git
cd ModelX
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
```

**Linux / macOS / Claude Code cloud**

```bash
git clone https://github.com/n-kaushik1/ModelX.git
cd ModelX
bash scripts/setup_env.sh        # uses uv if available (it can download Python 3.11 itself)
```

The last lines it prints should show `python 3.11.x`, the versions above, and the three CSV files in `data/`.

> In the commands below, `.venv/bin/python` is the Linux/macOS path. On Windows use `.venv\Scripts\python.exe`.

---

## Everyday tasks

### 1. Check that the framework still works (about 2 min)

```bash
.venv/bin/python experiments/robust_test.py submission/sub2
```

It fits on synthetic data with awkward properties and checks predictions in a fresh process, as the leaderboard
does. It must end with `ROBUSTNESS TEST PASSED`.

### 2. Estimate scores locally (about 10 min per problem on 16 cores)

```bash
.venv/bin/python experiments/eval_splits.py submission/sub2 data p1 42 1 2
.venv/bin/python experiments/eval_splits.py submission/sub2 data p2 42 1 2
.venv/bin/python experiments/eval_splits.py submission/sub2 data p3 42 1 2
```

Each line of output is one split: the score in the leaderboard metric, the chosen blend, and the per-family CV
scores. Compare against the tables in `docs/experiment_notes.md` using the same seeds. Differences below about
0.002 (AUC or log loss) or 0.01 (normalized cost) are noise.

### 3. Build the zip to upload (about 10 min)

```bash
cd submission/sub2
../../.venv/bin/python make_submission.py ../../data --out sub2.zip
```

This trains on the full development data, writes `sub2.zip`, then reloads it in a fresh process and
predicts, the same check the leaderboard runs. It must end with `self-check passed`. If it warns about a
version mismatch, the environment is wrong; re-run the setup script.

### 4. Upload to the leaderboard

1. On Codabench, open **My Submissions** and upload `submission.zip`. If you built it in the cloud, download it
   from GitHub first: open the file and click *Download raw file*.
2. Wait until it shows **Finished**. If it shows **Failed**, open it and read the ingestion log.
3. Click the green **leaderboard icon** in the *Actions* column. Submissions are **not** added to the
   leaderboard automatically.
4. Check the **Results** tab.

Limits: 3 uploads per day, 30 in total; failed or deleted uploads also count. The submission shown on the
leaderboard at the deadline is your final one, and its `solution.py` is what gets retrained on the private
problems.

---

## How the framework works (short version)

1. **Read the objective** from `metadata`: metric, whether a threshold is needed, FP/FN costs.
2. **Clean and profile**: drop ID-like and constant columns, detect numeric vs categorical columns, and measure
   missingness, imbalance and cardinality.
3. **Encode** in a way that suits each model family: native NaN handling for boosting, imputation + one-hot for
   forests, scaling + log transforms for logistic regression.
4. **Compare model families** with stratified 5-fold out-of-fold predictions: logistic regression, LightGBM,
   XGBoost, HistGradientBoosting, ExtraTrees, RandomForest, and CatBoost when there are high-cardinality
   categoricals.
5. **Tune** each family with a time-boxed random search, scored on the problem's metric (log loss when the metric
   needs a threshold).
6. **Blend** the best models with greedy ensemble selection on fresh folds, then calibrate with Platt scaling if
   that improves log loss.
7. **Pick the threshold** (cost problems only) by minimising the expected cost on out-of-fold predictions,
   stabilised by bootstrapping.
8. **Refit** the selected models on all the data. Everything is stored inside the object, so `predict_proba`
   works on raw features.

Details and the reasoning behind each choice: [`docs/framework_design.md`](docs/framework_design.md).

---

## Rules this project must follow

From the course brief, Sections 8 and 10:

- One framework for all problems, with **no dataset-specific rules** (no feature names, row counts or other
  fingerprints).
- `record_id` is never used as a feature.
- Only the libraries listed in `requirements.txt` (no neural nets, AutoML, internet access or stored predictions).
- The P3 threshold is learned from the development data only.

---

## Continuing with Claude Code in the cloud

`CLAUDE.md` gives a Claude Code session the project rules, setup and commands. A cloud session works on a fresh
clone of this repository:

1. Connect GitHub at [claude.ai/code](https://claude.ai/code) (authorize the Claude GitHub App) and pick this repo.
2. In the first message ask it to run `bash scripts/setup_env.sh`. The default *Trusted* network access allows
   PyPI and GitHub, which is all the setup needs.
3. It works on its own branch. Review the changes on claude.ai/code or GitHub and merge them into `main`.

`.claude/settings.json` turns off Claude's commit and pull-request attribution for this repository.
