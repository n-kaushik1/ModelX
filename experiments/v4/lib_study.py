"""Offline holdout analysis of combiners / calibration / thresholds on the saved v4 libraries.
For every outer fold f: Ltr (members x n_train OOF logits), Lte (members x n_test bagged logits).
usage: python lib_study.py <tag> <key>"""
import sys, glob
import numpy as np
from sklearn.linear_model import LogisticRegression
from kit import *

tag, key = sys.argv[1], sys.argv[2]
log = Log(f"lib_study_{tag}_{key}.log")
X, y, md = load(key)
files = sorted(glob.glob(f"{WORK}/logs/lib_{tag}_{key}_s0_f*.npz"))
rank = key == "p1"


def bscore(yy, z):
    return auc(yy, z) if rank else -ll(yy, sigmoid(z))


def caruana(L, yy, steps=40):
    c = np.zeros(len(L))
    s1 = [bscore(yy, L[i]) for i in range(len(L))]
    c[int(np.argmax(s1))] = 1
    cur = max(s1)
    for _ in range(steps - 1):
        cand = [bscore(yy, ((c + np.eye(len(L))[i]) / (c.sum() + 1)) @ L) for i in range(len(L))]
        j = int(np.argmax(cand))
        if cand[j] <= cur + 1e-7:
            break
        cur, c[j] = cand[j], c[j] + 1
    return c / c.sum()


def platt(z, yy):
    m = LogisticRegression(C=1e6, max_iter=1000).fit(z[:, None], yy)
    return m.coef_[0, 0], m.intercept_[0]


strategies = ["best_single", "caruana_raw", "caruana_platt", "stack_C1", "stack_C0.1", "equal", "oracle_platt_on_caruana"]
P = {s: np.zeros(len(y)) for s in strategies}
slopes = []
for f in files:
    d = np.load(f, allow_pickle=True)
    Ltr, Lte, tr, va = d["Ltr"], d["Lte"], d["tr"], d["va"]
    ytr, yte = y[tr], y[va]
    names = list(d["names"])
    s1 = [bscore(ytr, Ltr[i]) for i in range(len(Ltr))]
    P["best_single"][va] = Lte[int(np.argmax(s1))]
    w = caruana(Ltr, ytr)
    P["caruana_raw"][va] = w @ Lte
    a, b = platt(w @ Ltr, ytr)
    P["caruana_platt"][va] = a * (w @ Lte) + b
    for C in (1.0, 0.1):
        st = LogisticRegression(C=C, max_iter=3000).fit(Ltr.T, ytr)
        P[f"stack_C{C:g}"][va] = st.decision_function(Lte.T)
    P["equal"][va] = Lte.mean(0)
    ao, bo = platt(w @ Lte, yte)                      # oracle: calibrate on the test rows themselves
    P["oracle_platt_on_caruana"][va] = ao * (w @ Lte) + bo
    slopes.append((a, ao / 1.0, platt(a * (w @ Lte) + b, yte)[0]))
    log(f"fold {f[-6:-4]}: members={len(names)} caruana weights=" +
        ", ".join(f"{n.split('(')[0]}:{x:.2f}" for n, x in zip(names, w) if x > 0) +
        f" | Platt slope train={a:.3f}; slope needed on test after Platt={slopes[-1][2]:.3f}")
for s in strategies:
    z = P[s]
    msg = f"{s:26s} pooled {'AUC' if rank else 'LL'}={bscore(y, z):.5f}" if rank else f"{s:26s} pooled LL={ll(y, sigmoid(z)):.5f}"
    if key == "p3":
        msg += f" cost@1/11={norm_cost(y, sigmoid(z), 1/11):.4f}"
    log(msg)
log(f"mean test slope after train-Platt: {np.mean([s[2] for s in slopes]):.3f}  (1.0 = no bagging dilution)")
