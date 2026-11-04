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

| Task | Command | Time (16 cores) |
|---|---|---|
| Robustness smoke test | `.venv/bin/python experiments/robust_test.py submission/improved` | ~2 min |
| Local estimate, 3 splits | `.venv/bin/python experiments/eval_splits.py submission/improved data p1 42 1 2` (also `p2`, `p3`) | ~10 min per problem |
| Official quick check | `cd submission/improved && ../../.venv/bin/python local_test.py ../../data` | ~10 min |
| Build the upload zip | `cd submission/improved && ../../.venv/bin/python make_submission.py ../../data` | ~10 min |

`make_submission.py` writes `submission.zip` next to itself and runs the leaderboard's own load-and-predict check.
On the 4-vCPU cloud VM expect about twice these times. The cloud's foreground command limit is 10 minutes, so run
long jobs in the background with output redirected to a log file.

## Conventions

- `submission/baseline/` is frozen as the reference; work in `submission/improved/`. For a new variant, copy the
  whole folder: `make_submission.py` zips every `.py` next to it, so keep only `solution.py` (+ helpers) and the
  four kit files there.
- Compare variants with `eval_splits.py` on the same seeds (`42 1 2`) and on the same machine, against
  `docs/experiment_notes.md`. Tuning is time-boxed, so a slower machine tries fewer configurations and gets
  slightly different models. Differences below about 0.002 AUC, 0.002 log loss or 0.01 normalized cost are noise.
- When a change is accepted: rebuild `submission.zip`, add the results to `docs/experiment_notes.md`, put the raw
  output in `experiments/logs/`, and commit the zip so it can be downloaded and uploaded.
- The leaderboard is an external check, not a validation set (3 submissions per day, 30 in total).
- Commits and pull requests must not carry `Co-Authored-By: Claude` or "Generated with Claude Code" lines. This is
  the repo owner's preference, and `.claude/settings.json` already turns attribution off.
