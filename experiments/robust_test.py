"""Robustness test of a solution.py on synthetic data with nasty properties + fresh-process joblib load."""
import os, sys, subprocess, tempfile
import joblib
import numpy as np, pandas as pd

sol_dir = sys.argv[1]
sys.path.insert(0, sol_dir)
from solution import StudentSolution

rng = np.random.default_rng(0)
n = 3000
X = pd.DataFrame({
    "num_a": rng.normal(size=n),
    "num_skew": np.exp(rng.normal(size=n) * 2),
    "num_str": rng.integers(0, 100, n).astype(str),            # numeric-looking strings
    "hi_card": rng.choice([f"lvl{i}" for i in range(200)], n),  # high-cardinality categorical -> CatBoost path
    "lo_card": rng.choice(["a", "b", "c"], n),
    "flag": rng.random(n) < 0.3,
    "const": 1.0,
})
X.loc[rng.random(n) < 0.1, "num_a"] = np.nan
X.loc[rng.random(n) < 0.1, "lo_card"] = None
logit = 1.5 * X["num_a"].fillna(0) + (X["lo_card"] == "b") * 1.0 + (X["hi_card"].str[-1] == "7") * 1.5 - 2
y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int))

for md in [{"objective": "roc_auc", "greater_is_better": True, "requires_threshold": False,
            "cost_false_positive": None, "cost_false_negative": None, "positive_label": 1},
           {"objective": "decision_cost", "greater_is_better": False, "requires_threshold": True,
            "cost_false_positive": 1, "cost_false_negative": 10, "positive_label": 1}]:
    sol = StudentSolution(time_budget=40).fit(X, y, md)
    fams = sorted({m.spec.family for m in sol.models_})
    print(md["objective"], "selected:", fams, "threshold:", round(sol.get_threshold(), 4),
          "searched:", sorted({s[0] for s in sol.report_["search"]}))
    # test frame: unseen categories, all-NaN column, shuffled order, missing column, extra column
    Xt = X.sample(400, random_state=1).copy()
    Xt.loc[Xt.index[:50], "lo_card"] = "NEVER_SEEN"
    Xt.loc[Xt.index[:50], "hi_card"] = "NEW_LEVEL"
    Xt["num_a"] = np.nan
    Xt = Xt.drop(columns=["num_skew"])
    Xt["extra"] = 5
    Xt = Xt[rng.permutation(Xt.columns)]
    p = sol.predict_proba(Xt, md)
    assert p.shape == (len(Xt),) and np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1
    path = os.path.join(tempfile.mkdtemp(), "m.joblib")
    joblib.dump(sol, path, compress=3)
    code = ("import sys, joblib, pandas as pd; sys.path.insert(0, sys.argv[1]); import solution; "
            "sys.modules['__main__'] = solution; m = joblib.load(sys.argv[2]); "
            "X = pd.read_pickle(sys.argv[3]); p = m.predict_proba(X, {}); print('fresh-process ok', p[:3].round(4))")
    xp = os.path.join(tempfile.mkdtemp(), "x.pkl")
    Xt.to_pickle(xp)
    r = subprocess.run([sys.executable, "-c", code, sol_dir, path, xp], capture_output=True, text=True,
                       cwd=tempfile.gettempdir())
    print(r.stdout.strip() or r.stderr[-1500:])
    q = joblib.load(path).predict_proba(Xt, md)
    assert np.allclose(p, q)
print("ROBUSTNESS TEST PASSED")
