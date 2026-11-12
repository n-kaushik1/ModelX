"""Test A: one outer fold of P3 (seed 0, fold 0) at n_jobs threads, with a given time_budget.  Reports which
configurations ran, fit time, model size, and held-out cost / log loss."""
import sys, os, time, tempfile, json, collections
sys.path.insert(0, "experiments/v4"); sys.path.insert(0, "submission/v12")
import numpy as np, joblib
from kit import load, folds, ll, auc, norm_cost
from solution import StudentSolution
key, nj, tb = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
X, y, md = load(key); tr, va = folds(y, 0)[0]
t0 = time.time()
sol = StudentSolution(n_jobs=nj, time_budget=tb).fit(X.iloc[tr].reset_index(drop=True), y[tr], dict(md))
fit_s = time.time() - t0
path = os.path.join(tempfile.mkdtemp(), "m.joblib"); joblib.dump(sol, path, compress=3)
sol = joblib.load(path); t1 = time.time()
p = sol.predict_proba(X.iloc[va].reset_index(drop=True), dict(md)); pred_s = time.time() - t1
t = sol.get_threshold()
ran = collections.Counter(f for f, *_ in sol.report_["search"])
out = dict(key=key, n_jobs=nj, time_budget=tb, fit_s=round(fit_s), size_mb=round(os.path.getsize(path) / 1e6, 1), predict_s=round(pred_s, 1),
           configs_run=dict(ran), threshold=t, blend=sol.report_["blend_weights"], heldout_logloss=round(ll(y[va], p), 5), heldout_auc=round(auc(y[va], p), 5))
if key == "p3": out["heldout_cost"] = round(norm_cost(y[va], p, t), 4)
print("RESULT", json.dumps(out))
