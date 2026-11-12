"""Test B: a large private-style problem with known truth: 150k rows, 60 numerics (mixed marginals), 6 categoricals
(one with 500 levels), a date column, 8 % positives, cost 1:10.  Measures fit time, model size, predict time and the
regret versus the Bayes-optimal decisions."""
import sys, os, time, tempfile, json
sys.path.insert(0, "submission/v12")
import numpy as np, pandas as pd, joblib
from solution import StudentSolution
rng = np.random.default_rng(11); n, ntest = 150000, 40000; N = n + ntest
X = pd.DataFrame({f"x{j:02d}": (rng.normal(size=N) if j % 3 else np.exp(rng.normal(size=N))) for j in range(60)})
for j in range(5): X[f"c{j}"] = rng.choice([f"L{i}" for i in range(rng.integers(2, 8))], N)
X["city"] = rng.choice([f"C{i:03d}" for i in range(500)], N, p=rng.dirichlet(np.ones(500) * 0.5))
X["when"] = (pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 700, N), unit="D")).astype(str)
Z = {c: (X[c] - X[c].mean()) / X[c].std() for c in X.columns if c.startswith("x")}
logit = (Z["x00"] + 0.8 * np.maximum(Z["x01"] - 0.3, 0) + 0.7 * Z["x02"] ** 2 * 0.5 - 0.6 * Z["x03"] + 0.5 * Z["x04"] * Z["x05"]
         + 0.4 * np.tanh(2 * Z["x07"]) + (X["c0"] == "L1") * 0.6 + X["city"].map(dict(zip([f"C{i:03d}" for i in range(500)], rng.normal(0, 0.5, 500)))).to_numpy()
         + 0.3 * (pd.to_datetime(X["when"]).dt.month > 6).to_numpy())
logit = (logit - logit.mean()) / logit.std() * 1.6
lo, hi = -15., 15.
for _ in range(60):
    mid = (lo + hi) / 2; hi, lo = (mid, lo) if np.mean(1 / (1 + np.exp(-(logit + mid)))) > 0.08 else (hi, mid)
p = 1 / (1 + np.exp(-(logit + (lo + hi) / 2))); y = (rng.random(N) < p).astype(int)
X.loc[rng.random(N) < 0.05, "x01"] = np.nan; X.loc[rng.random(N) < 0.03, "c1"] = None
md = {"objective": "decision_cost", "greater_is_better": False, "requires_threshold": True, "cost_false_positive": 1, "cost_false_negative": 10, "positive_label": 1}
Xtr, ytr, Xte, yte, pte = X.iloc[:n].reset_index(drop=True), y[:n], X.iloc[n:].reset_index(drop=True), y[n:], p[n:]
t0 = time.time(); sol = StudentSolution(n_jobs=int(sys.argv[1])).fit(Xtr, ytr, md); fit_s = time.time() - t0
path = os.path.join(tempfile.mkdtemp(), "m.joblib"); joblib.dump(sol, path, compress=3); sol = joblib.load(path)
t1 = time.time(); ph = sol.predict_proba(Xte, md); pred_s = time.time() - t1; t = sol.get_threshold()
cost = lambda pp, tt: (np.sum((pp >= tt) & (yte == 0)) + 10 * np.sum((pp < tt) & (yte == 1))) / min(10 * yte.sum(), (yte == 0).sum())
from sklearn.metrics import log_loss, roc_auc_score
print("RESULT", json.dumps(dict(fit_s=round(fit_s), size_mb=round(os.path.getsize(path) / 1e6, 1), predict_s=round(pred_s, 1), threshold=t,
      cost_model=round(cost(ph, t), 4), cost_bayes_optimal=round(cost(pte, 1 / 11), 4), logloss_model=round(log_loss(yte, ph), 5), logloss_optimal=round(log_loss(yte, pte), 5),
      auc_model=round(roc_auc_score(yte, ph), 4), auc_optimal=round(roc_auc_score(yte, pte), 4), final_mode=sol.report_.get("final_mode"), blend=sol.report_.get("blend_weights"),
      dropped={k: v for k, v in sol.cleaner_.dropped_.items() if v}, n_numeric=len(sol.cleaner_.numeric_), n_cat=len(sol.cleaner_.categorical_))))
