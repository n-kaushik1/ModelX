"""Original local harness (kept for reference; experiments/eval_splits.py is the one used for the results).
25 % stratified hold-out, the old metadata format ("metric", "id_columns") and record_id passed in, so it
also checks that the framework's metadata parsing / identifier dropping still works.

usage (from the repo root): python experiments/run_local_eval.py [p1|p2|p3 ...] [--budget SECONDS] [--seed N]
"""
import os
import sys
import time
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
sys.path.insert(0, os.path.join(ROOT, "submission", "improved"))
from solution import StudentSolution  # noqa: E402

PROBLEMS = {
    "p1": ("p1_manufacturing_development.csv", "quality_failure",
           {"metric": "roc_auc", "id_columns": ["record_id"]}),
    "p2": ("p2_credit_development.csv", "serious_delinquency",
           {"metric": "log_loss", "id_columns": ["record_id"]}),
    "p3": ("p3_telecom_development.csv", "major_service_disruption",
           {"metric": "cost", "cost_false_positive": 1, "cost_false_negative": 10, "id_columns": ["record_id"]}),
}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    budget = 240
    seed = 0
    if "--budget" in sys.argv:
        budget = float(sys.argv[sys.argv.index("--budget") + 1])
        args = [a for a in args if a != str(int(budget)) and a != str(budget)]
    if "--seed" in sys.argv:
        seed = int(sys.argv[sys.argv.index("--seed") + 1])
        args = [a for a in args if a != str(seed)]
    keys = args or list(PROBLEMS)
    for k in keys:
        fn, target, md = PROBLEMS[k]
        df = pd.read_csv(os.path.join(DATA, fn))
        y = df[target].values
        X = df.drop(columns=[target])
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
        t0 = time.time()
        sol = StudentSolution(time_budget=budget, verbose=True).fit(Xtr, ytr, md)
        # holdout rows arrive WITHOUT the target and in shuffled column order
        Xte = Xte[np.random.default_rng(1).permutation(Xte.columns)]
        p = sol.predict_proba(Xte, md)
        thr = sol.get_threshold()
        print(f"\n===== {k}  fit {time.time() - t0:.0f}s")
        if md["metric"] == "roc_auc":
            print("holdout AUC     :", round(roc_auc_score(yte, p), 5))
        print("holdout logloss :", round(log_loss(yte, p), 5), " (prior:", round(log_loss(yte, np.full(len(yte), ytr.mean())), 5), ")")
        yhat = (p >= thr).astype(int)
        cost = ((yhat == 1) & (yte == 0)).sum() + 10 * ((yhat == 0) & (yte == 1)).sum()
        cost05 = ((p >= .5) & (yte == 0)).sum() + 10 * ((p < .5) & (yte == 1)).sum()
        allneg = 10 * yte.sum()
        print(f"threshold={thr:.4f}  holdout cost={cost} (per-1000 {1000 * cost / len(yte):.1f}); "
              f"cost@0.5={cost05}; all-negative={allneg}")
        s = sol.summary()
        s.pop("profile", None)
        print(json.dumps(s, indent=1, default=str))


if __name__ == "__main__":
    main()
