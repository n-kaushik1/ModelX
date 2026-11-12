# P3: which combiner criterion transfers to held-out rows?  Saved outer folds (true holdout), framework 9 libraries.
import sys, glob, numpy as np
sys.path.insert(0, "experiments/v4")
from kit import load, sigmoid, ll, norm_cost
from sklearn.linear_model import LogisticRegression
X, y, md = load("p3")
def caruana(L, yy, crit, steps=40):
    c = np.zeros(len(L)); s1 = [crit(yy, L[i]) for i in range(len(L))]; c[int(np.argmax(s1))] = 1; cur = max(s1)
    for _ in range(steps - 1):
        cand = [crit(yy, ((c + np.eye(len(L))[i]) / (c.sum() + 1)) @ L) for i in range(len(L))]
        j = int(np.argmax(cand))
        if cand[j] <= cur + 1e-7: break
        cur, c[j] = cand[j], c[j] + 1
    return c / c.sum()
def bag_caruana(L, yy, crit, nb=25, seed=0):
    rng = np.random.default_rng(seed); w = np.zeros(len(L))
    for _ in range(nb):
        i = rng.integers(0, len(yy), len(yy)); w += caruana(L[:, i], yy[i], crit)
    w /= nb; w[w < 0.02] = 0; return w / w.sum()
def platt(z, yy):
    m = LogisticRegression(C=1e6, max_iter=1000).fit(z[:, None], yy); return float(m.coef_[0, 0]), float(m.intercept_[0])
def wll(yy, z, lo=0.03, hi=0.25):
    """log loss restricted to rows whose probability lies in the decision band (where the threshold acts)."""
    p = sigmoid(z); m = (p > lo) & (p < hi)
    return -ll(yy[m], p[m]) if m.sum() > 50 and len(np.unique(yy[m])) > 1 else -ll(yy, p)
def ecost(yy, z):
    """negative cost at the Bayes threshold after an in-sample Platt map (the metric itself)."""
    a, b = platt(z, yy); p = sigmoid(a * z + b)
    return -(np.sum((p >= 1/11) & (yy == 0)) + 10 * np.sum((p < 1/11) & (yy == 1)))
crits = {"logloss": lambda yy, z: -ll(yy, sigmoid(z)), "band_logloss": wll, "cost@bayes": ecost}
res = {k: [] for k in list(crits) + ["bag_logloss", "equal", "best_single_ll"]}
lls = {k: [] for k in res}
for f in sorted(glob.glob("experiments/v5/logs/outer/v7c_p3_s*_f*.npz")):
    d = np.load(f, allow_pickle=True)
    if "Ltr" not in d: continue
    Ltr, Lte, tr, va = d["Ltr"], d["Lte"], d["tr"], d["va"]; ytr, yte = y[tr], y[va]
    ws = {k: caruana(Ltr, ytr, c) for k, c in crits.items()}
    ws["bag_logloss"] = bag_caruana(Ltr, ytr, crits["logloss"])
    ws["equal"] = np.ones(len(Ltr)) / len(Ltr)
    s = [ll(ytr, sigmoid(Ltr[i])) for i in range(len(Ltr))]; e = np.zeros(len(Ltr)); e[int(np.argmin(s))] = 1; ws["best_single_ll"] = e
    for k, w in ws.items():
        a, b = platt(w @ Ltr, ytr); p = sigmoid(a * (w @ Lte) + b)
        res[k].append(norm_cost(yte, p, 1 / 11)); lls[k].append(ll(yte, p))
for k in res:
    print(f"{k:15s} holdout cost {np.mean(res[k]):.4f} (se {np.std(res[k])/np.sqrt(len(res[k])):.4f})  log loss {np.mean(lls[k]):.5f}")
