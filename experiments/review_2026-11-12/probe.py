"""Extra private-problem probes not covered by experiments/v5/stress_test.py."""
import sys, time, warnings, traceback; sys.path.insert(0, "submission/v12"); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from solution import StudentSolution
rng = np.random.default_rng(3)
def make(n, p_num=6):
    X = pd.DataFrame({f"n{j}": rng.normal(size=n) for j in range(p_num)})
    X["c0"] = rng.choice(list("abcd"), n)
    logit = X["n0"] + 0.8 * np.abs(X["n1"]) - 0.5 * X["n2"] * X["n3"] + (X["c0"] == "b") * 0.7 - 0.8
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y
cases = []
X, y = make(2000); cases.append(("labels {1,2}, no positive_label", X, pd.Series(y + 1), {"objective": "roc_auc"}))
X, y = make(2000); cases.append(("labels {-1,+1}", X, pd.Series(np.where(y == 1, 1, -1)), {"objective": "log_loss", "positive_label": 1}))
X, y = make(2000); cases.append(("string labels, positive_label='bad'", X, pd.Series(np.where(y == 1, "bad", "good")), {"objective": "log_loss", "positive_label": "bad"}))
X, y = make(2000); cases.append(("bool labels", X, pd.Series(y.astype(bool)), {"objective": "roc_auc"}))
X, y = make(2000); X2 = X.copy(); X2["account_number"] = rng.permutation(10**6 + np.arange(2000)); cases.append(("unique integer id with non-id name", X2, pd.Series(y), {"objective": "roc_auc"}))
X, y = make(4000); X2 = X.copy(); X2["zip"] = rng.choice([f"Z{i:05d}" for i in range(2500)], 4000); cases.append(("2500-level categorical", X2, pd.Series(y), {"objective": "log_loss"}))
X, y = make(2000); X2 = X.copy(); X2["mixed"] = [v if i % 7 else "n/a" for i, v in enumerate(X["n4"].round(2))]; cases.append(("numeric column with 'n/a' strings", X2, pd.Series(y), {"objective": "roc_auc"}))
X, y = make(2000); X2 = X.copy(); X2["code"] = rng.integers(1, 9, 2000); cases.append(("integer-coded categorical", X2, pd.Series(y), {"objective": "roc_auc"}))
X, y = make(2000); X2 = X.copy(); X2.columns = [0, 1, 2, 3, 4, 5, 6]; cases.append(("integer column names", X2, pd.Series(y), {"objective": "roc_auc"}))
X, y = make(2000); cases.append(("y as numpy float array", X, y.astype(float), {"objective": "roc_auc"}))
X, y = make(2000); cases.append(("decision_cost, fn cheaper than fp (1 vs 4)", X, pd.Series(y), {"objective": "decision_cost", "requires_threshold": True, "cost_false_positive": 4, "cost_false_negative": 1}))
X, y = make(2000); cases.append(("metadata metric spelled 'AUC-ROC' with extra keys", X, pd.Series(y), {"metric": "AUC-ROC", "problem_name": "x", "n_features": 7}))
ok = True
for name, X, y, md in cases:
    t0 = time.time()
    try:
        sol = StudentSolution(time_budget=20, n_jobs=2).fit(X, y, md)
        Xt = X.sample(300, random_state=0)
        p = sol.predict_proba(Xt, md); t = sol.get_threshold()
        assert p.shape == (300,) and np.isfinite(p).all() and 0 <= p.min() <= p.max() <= 1
        yt = y.loc[Xt.index] if hasattr(y, "loc") else y[Xt.index]
        from sklearn.metrics import roc_auc_score
        pos = md.get("positive_label", None)
        yb = (np.asarray(yt) == pos).astype(int) if pos is not None and set(np.unique(yt)) - {0, 1, True, False} else (np.asarray(yt) == np.unique(yt)[-1]).astype(int)
        print(f"OK   {name:45s} {time.time()-t0:4.0f}s thr={t:.3f} auc(train rows)={roc_auc_score(yb, p):.3f} dropped={ {k: v for k, v in sol.cleaner_.dropped_.items() if v} } num={len(sol.cleaner_.numeric_)} cat={len(sol.cleaner_.categorical_)} blend={sol.report_.get('blend_weights')}", flush=True)
    except Exception as e:
        ok = False; print(f"FAIL {name}: {repr(e)}"); traceback.print_exc()
print("PROBE", "PASSED" if ok else "FAILED")
