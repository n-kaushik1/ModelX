"""Repeated 80/20 evaluation of a solution.py folder with the leaderboard metrics + joblib round trip.
usage: python eval_splits.py <solution_dir> <data_dir> <p1|p2|p3> <seed> [<seed> ...]"""
import os, sys, tempfile, time, json
import joblib
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split

sol_dir, data_dir, key = sys.argv[1], sys.argv[2], sys.argv[3]
seeds = [int(s) for s in sys.argv[4:]] or [42]
sys.path.insert(0, sol_dir)
from problems import PROBLEMS, metadata_for
from solution import StudentSolution

prob = [p for p in PROBLEMS if p["key"] == key][0]
df = pd.read_csv(os.path.join(data_dir, prob["file"]))
X = df.drop(columns=[prob["target"], "record_id"])
y = df[prob["target"]].astype(int)
md = metadata_for(prob["objective"])
for seed in seeds:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    t0 = time.time()
    sol = StudentSolution().fit(Xtr.reset_index(drop=True), ytr.reset_index(drop=True), dict(md))
    fit_s = time.time() - t0
    path = os.path.join(tempfile.mkdtemp(), "m.joblib")
    joblib.dump(sol, path, compress=3)
    sol = joblib.load(path)
    p = np.asarray(sol.predict_proba(Xte.reset_index(drop=True), dict(md)), dtype=float)
    yv = yte.to_numpy()
    if prob["objective"] == "roc_auc":
        s = roc_auc_score(yv, p)
    elif prob["objective"] == "log_loss":
        s = log_loss(yv, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1])
    else:
        t = float(sol.get_threshold())
        a = p >= t
        s = (np.sum(a & (yv == 0)) + 10 * np.sum(~a & (yv == 1))) / min(10 * yv.sum(), (yv == 0).sum())
    r = sol.report_
    print(json.dumps({"sol": os.path.basename(sol_dir.rstrip('/\\')), "key": key, "seed": seed,
                      "score": round(float(s), 5), "threshold": round(float(sol.get_threshold()), 4),
                      "fit_s": round(fit_s), "blend": r.get("blend_weights"),
                      "family_scores": {k: round(v, 5) for k, v in r.get("family_scores", {}).items()},
                      "calibrated": sol.calibrator_ is not None}), flush=True)
