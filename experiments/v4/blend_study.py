"""Ensemble-strategy study on cached OOF predictions (logs/oof_<key>_<name>.npy, 3 fold seeds each).
Honest protocol: rows are split 50/50 (40 repeats); a strategy picks weights on half A using fold-seed-0 OOF
and is scored on half B using fold-seed-1 OOF (different rows AND different fold noise)."""
import sys, glob, os
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from kit import *

key = sys.argv[1]
only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
log = Log(f"blend_study_{key}.log")
X, y, md = load(key)
O = {}
for f in sorted(glob.glob(f"{WORK}/logs/oof_{key}_*.npy")):
    n = os.path.basename(f)[len(f"oof_{key}_"):-4]
    if only and n not in only:
        continue
    O[n] = np.load(f)
names = list(O)
L = {s: np.vstack([logit(O[n][s]) for n in names]) for s in range(3)}
metric = "auc" if key == "p1" else "ll"


def sc(yy, z):
    return auc(yy, z) if metric == "auc" else -ll(yy, sigmoid(z))


def caruana(Lm, yy, steps=25):
    c = np.zeros(len(Lm))
    s1 = [sc(yy, Lm[i]) for i in range(len(Lm))]
    c[int(np.argmax(s1))] = 1
    cur = max(s1)
    for _ in range(steps):
        cand = [sc(yy, ((c + np.eye(len(Lm))[i]) / (c.sum() + 1)) @ Lm) for i in range(len(Lm))]
        j = int(np.argmax(cand))
        if cand[j] <= cur + 1e-7:
            break
        cur = cand[j]
        c[j] += 1
    return c / c.sum()


def optw(Lm, yy):
    k = len(Lm)
    f = lambda w: -sc(yy, (np.abs(w) / np.abs(w).sum()) @ Lm)
    r = minimize(f, np.ones(k) / k, method="Nelder-Mead", options=dict(maxiter=3000, xatol=1e-4, fatol=1e-7))
    w = np.abs(r.x)
    return w / w.sum()


log(f"===== {key}: {len(names)} members")
for i, n in enumerate(names):
    log(f"  {n:22s} " + " ".join(f"{sc(y, L[s][i]):.5f}" for s in range(3)))
rng = np.random.default_rng(0)
res = {k: [] for k in ["best_single", "B", "equal_all", "caruana", "caruana+platt", "caruana_top", "optw", "stackLR"]}
wsum = np.zeros(len(names))
for r in range(40):
    A = rng.random(len(y)) < 0.5
    B = ~A
    LA, LB = L[0][:, A], L[1][:, B]
    yA, yB = y[A], y[B]
    singles = [sc(yA, LA[i]) for i in range(len(names))]
    bi = int(np.argmax(singles))
    res["best_single"].append(sc(yB, LB[bi]))
    if "B" in names:
        res["B"].append(sc(yB, LB[names.index("B")]))
    res["equal_all"].append(sc(yB, LB.mean(0)))
    w = caruana(LA, yA)
    wsum += w
    res["caruana"].append(sc(yB, w @ LB))
    pl = LogisticRegression(C=1e6).fit((w @ LA)[:, None], yA)
    res["caruana+platt"].append(sc(yB, pl.decision_function((w @ LB)[:, None])))
    top = np.argsort(singles)[::-1][:6]
    wt = caruana(LA[top], yA)
    res["caruana_top"].append(sc(yB, wt @ LB[top]))
    t8 = np.argsort(singles)[::-1][:8]
    wo = optw(LA[t8], yA)
    res["optw"].append(sc(yB, wo @ LB[t8]))
    st = LogisticRegression(C=1.0, max_iter=2000).fit(LA.T, yA)
    res["stackLR"].append(sc(yB, st.decision_function(LB.T)))
base = np.array(res["best_single"])
for k, v in res.items():
    if v:
        v = np.array(v)
        log(f"  {k:12s} mean={v.mean():.5f}  diff_vs_best_single={np.mean(v - base):+.5f} (se {np.std(v - base) / np.sqrt(len(v)):.5f})")
log("  avg caruana weights: " + ", ".join(f"{n}={w:.2f}" for n, w in sorted(zip(names, wsum / 40), key=lambda t: -t[1]) if w > 0.01))
