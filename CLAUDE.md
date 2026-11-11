# Notes for Claude sessions working on this repo

DA5401 course project: **one reusable binary-classification framework** (`StudentSolution` in `solution.py`).
It is scored on a Codabench leaderboard (P1 ROC-AUC, P2 log loss, P3 normalized decision cost `FP + 10·FN`) and
later retrained, unchanged, on three unseen private problems. The leaderboard closes **28 Nov 2026, 23:59 IST**;
the final report is due 29 Nov. Read `README.md` first, then `docs/experiment_notes.md` for what has been tried.

## Hard rules: never break these

They come from `docs/project_instructions.md` and `info.md`:

- One generic framework. No dataset-specific logic: never branch on feature names, row or column counts, specific
  values or other dataset fingerprints. Every decision must come from measured data properties or from `metadata`.
- `record_id` is never a feature. The leaderboard passes the raw features without the target and without `record_id`.
- Only the libraries in `requirements.txt`, at those exact versions, on Python 3.11. No neural nets, no AutoML, no
  internet access at runtime, no stored predictions.
- The P3 threshold must be chosen from development data only.
- All preprocessing lives inside the fitted `StudentSolution`. It is pickled with joblib, and the leaderboard only
  calls `predict_proba(X, metadata)` and `get_threshold()`, so nothing may be fitted at predict time.

## Setup (once per session: a cloud VM starts from a fresh clone)

```bash
bash scripts/setup_env.sh      # .venv (Python 3.11 + pinned versions) and data/ (CSVs from public_data.zip)
```

`.venv/` and `data/` are gitignored, so they are recreated in every new session.

## Commands (run from the repo root)

The current framework is `submission/v5/` (framework version 7; see `submission/v5/V5_NOTES.md`).

| Task | Command | Time |
|---|---|---|
| Private-problem stress test | `.venv/bin/python experiments/v5/stress_test.py submission/v5` | ~20 min (16+ cores) |
| One outer-CV fold of a framework (fit 4/5, save/reload, score 1/5) | `cd experiments/v5 && ../../.venv/bin/python outer_fold.py ../../submission/v5/solution.py <tag> p1 <n_jobs> <fold_seed> <fold>` | ~5 min (4 threads) |
| Run a queue of folds in parallel | `cd experiments/v5 && ./runq.sh jobs.txt <parallel>` (one `outer_fold.py` argument line per job) | |
| Aggregate outer CV | `cd experiments/v5 && ../../.venv/bin/python outer_agg.py <tag> [<tag> ...]` | seconds |
| Re-combine saved fold libraries offline (combiners, calibration, P3 threshold) | `cd experiments/v5 && ../../.venv/bin/python lib_study.py <tag> <key> [fold_seed]` | ~1 min |
| Build the upload zip | `cd submission/v5 && ../../.venv/bin/python make_submission.py ../../data` | ~10 min |

`make_submission.py` writes `submission.zip` next to itself and runs the leaderboard's own load-and-predict check.
Build it on an otherwise idle machine: `time_budget` is a safety net that can cut the random refinement short
when the CPU is heavily shared. Run long jobs in the background with output redirected to a log file.

## Conventions

- `submission/baseline/` is frozen as the reference; the current framework is `submission/v5/`. For a new variant,
  copy the whole folder: `make_submission.py` zips every `.py` next to it, so keep only `solution.py` (+ helpers)
  and the four kit files there.
- Compare framework versions with paired outer CV (`experiments/v5/outer_fold.py`, same fold seeds for both).
  Fold-seed noise is about 0.0005 AUC / log loss; the public holdout itself is far noisier (about 0.006 AUC and
  0.03 normalized cost on a 2-3k-row split).
- When a change is accepted: rebuild `submission.zip`, add the results to `docs/experiment_notes.md`, put the raw
  output in `experiments/logs/`, and commit the zip so it can be downloaded and uploaded.
- The leaderboard is an external check, not a validation set (3 submissions per day, 30 in total).
- Commits and pull requests must not carry `Co-Authored-By: Claude` or "Generated with Claude Code" lines. This is
  the repo owner's preference, and `.claude/settings.json` already turns attribution off.
