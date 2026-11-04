"""Quick local estimate of your framework's scores on the development data.

    python local_test.py path/to/folder_with_development_csvs

Each development file is split 80/20 (stratified). Your StudentSolution is trained on the 80% part, saved and
reloaded with joblib (as the leaderboard does), and scored on the raw 20% part with the leaderboard metric.
These scores are a rough check, not a prediction of your leaderboard score.
"""
import argparse
import os
import tempfile
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from problems import PROBLEMS, metadata_for
from solution import StudentSolution


def normalized_cost(y, p, t, c_fp, c_fn):
    alert = p >= t
    cost = c_fp * np.sum(alert & (y == 0)) + c_fn * np.sum(~alert & (y == 1))
    return cost / min(c_fn * np.sum(y == 1), c_fp * np.sum(y == 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    a = ap.parse_args()
    for prob in PROBLEMS:
        path = os.path.join(a.data_dir, prob["file"])
        if not os.path.isfile(path):
            print(f"skip {prob['file']} (not found)")
            continue
        df = pd.read_csv(path)
        X = df.drop(columns=[prob["target"], "record_id"])
        y = df[prob["target"]].astype(int)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        metadata = metadata_for(prob["objective"])

        t0 = time.time()
        sol = StudentSolution().fit(X_tr.reset_index(drop=True), y_tr.reset_index(drop=True), dict(metadata))
        t = float(sol.get_threshold()) if metadata["requires_threshold"] else None
        model_path = os.path.join(tempfile.mkdtemp(), "model.joblib")
        joblib.dump(sol, model_path)
        sol = joblib.load(model_path)
        p = np.asarray(sol.predict_proba(X_te.reset_index(drop=True), dict(metadata)), dtype=float)
        p = p[:, -1] if p.ndim == 2 else p
        elapsed = time.time() - t0

        assert p.shape == (len(X_te),), f"predict_proba must return {len(X_te)} values, got shape {p.shape}"
        assert np.all(np.isfinite(p)) and p.min() >= 0 and p.max() <= 1, "probabilities must be finite, in [0, 1]"
        yv = y_te.to_numpy()
        if prob["objective"] == "roc_auc":
            s = f"ROC-AUC {roc_auc_score(yv, p):.4f}"
        elif prob["objective"] == "log_loss":
            s = f"log loss {log_loss(yv, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1]):.4f}"
        else:
            assert t is not None and 0 <= t <= 1, "get_threshold() must return a number in [0, 1]"
            cost = normalized_cost(yv, p, t, metadata["cost_false_positive"], metadata["cost_false_negative"])
            s = f"normalized cost {cost:.4f} (threshold {t:.4f})"
        print(f"{prob['file']}: {s} | {elapsed:.0f} s")


if __name__ == "__main__":
    main()
