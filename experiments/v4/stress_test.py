"""Private-problem stress test: unusual shapes / metadata must never crash and must return valid output.
usage: python stress_test.py <dir containing solution.py> [time_budget]"""
import sys, time, warnings, subprocess, json
sys.path.insert(0, sys.argv[1]); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib, tempfile, os
from solution import StudentSolution
TB = float(sys.argv[2]) if len(sys.argv) > 2 else 25
rng = np.random.default_rng(1)


def make(n, p_num=8, p_cat=2, imb=0.3, hi_card=False, strings=False):
    X = pd.DataFrame({f"n{j}": rng.normal(size=n) for j in range(p_num)})
    for j in range(p_cat):
        X[f"c{j}"] = rng.choice([f"l{i}" for i in range(300 if hi_card else 4)], n)
    logit = X["n0"] + 0.8 * np.abs(X["n1"]) - 0.5 * X["n2"] * X["n3"] + (X["c0"] == "l1") * 0.7 + np.log(imb / (1 - imb))
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    X.loc[rng.random(n) < 0.05, "n0"] = np.nan
    if strings:
        y = np.where(y == 1, "yes", "no")
    return X, y


def weird(n=2500):
    X, y = make(n)
    X["customer_id"] = [f"C{i:06d}" for i in range(n)]          # identifier-like text
    X["const"] = 7.0                                            # constant
    X["all_nan"] = np.nan                                       # empty
    X["num_as_str"] = X["n4"].round(3).astype(str)              # numeric stored as text
    X["flag"] = rng.random(n) < 0.5                             # bool
    X["when"] = pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 300, n), unit="D")
    X["n5"] = X["n5"] * 1e9                                     # huge scale
    X.loc[rng.random(n) < 0.3, "c1"] = np.nan                   # heavy missing categorical
    X["n6"] = np.where(rng.random(n) < 0.01, np.inf, X["n6"])   # infinities
    return X, y


cases = [
    ("tiny n=300", make(300), {"objective": "roc_auc", "requires_threshold": False, "positive_label": 1}),
    ("tiny n=120 imbalanced cost", make(120, imb=0.15),
     {"objective": "decision_cost", "requires_threshold": True, "cost_false_positive": 1, "cost_false_negative": 5}),
    ("extreme imbalance 1%", make(6000, imb=0.01),
     {"objective": "decision_cost", "requires_threshold": True, "cost_false_positive": 1, "cost_false_negative": 20,
      "positive_label": 1}),
    ("many features p=150", make(3000, p_num=150, p_cat=3), {"objective": "log_loss", "requires_threshold": False}),
    ("high-card cats + string labels", make(3000, hi_card=True, strings=True),
     {"objective": "f1", "requires_threshold": True, "positive_label": "yes"}),
    ("large n=70k (subsample path)", make(70000, p_num=12),
     {"objective": "roc_auc", "requires_threshold": False, "positive_label": 1}),
    ("empty metadata", make(2000), {}),
    ("None metadata", make(1500), None),
    ("all-categorical, no numerics", (make(2000)[0].drop(columns=[f"n{j}" for j in range(8)]), make(2000)[1]),
     {"objective": "log_loss"}),
    ("weird columns", weird(), {"objective": "decision_cost", "requires_threshold": True,
                                "cost_false_positive": 3, "cost_false_negative": 1}),
    ("accuracy objective", make(2000), {"objective": "accuracy", "requires_threshold": True}),
]
ok = True
for name, (X, y), md in cases:
    t0 = time.time()
    try:
        sol = StudentSolution(time_budget=TB).fit(X, y, md)
        Xt = X.sample(min(500, len(X)), random_state=0).copy()
        if "c0" in Xt:
            Xt["c0"] = "UNSEEN"
        if "n1" in Xt:
            Xt["n1"] = np.nan
        Xt["extra"] = 1.0
        Xt = Xt[Xt.columns[::-1]]                                  # shuffled column order
        p = sol.predict_proba(Xt, md)
        t = sol.get_threshold()
        path = os.path.join(tempfile.mkdtemp(), "m.joblib")
        joblib.dump(sol, path)
        p2 = joblib.load(path).predict_proba(Xt, md)
        one = sol.predict_proba(Xt.iloc[:1], md)
        assert p.shape == (len(Xt),) and np.isfinite(p).all() and 0 <= p.min() and p.max() <= 1, "bad proba"
        assert 0 <= t <= 1 and np.allclose(p, p2) and one.shape == (1,), "bad threshold/reload"
        r = sol.report_
        print(f"OK   {name:32s} {time.time()-t0:5.0f}s thr={t:.3f} size={os.path.getsize(path)/1e6:.1f}MB "
              f"blend={r.get('blend_weights')} mode={r.get('final_mode')}", flush=True)
    except Exception as e:
        import traceback
        ok = False
        print(f"FAIL {name}: {repr(e)}")
        traceback.print_exc()
print("STRESS TEST " + ("PASSED" if ok else "FAILED"))
