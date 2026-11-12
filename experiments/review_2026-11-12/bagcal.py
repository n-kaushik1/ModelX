# Does k-fold bagging shift the calibration?  On the saved outer folds (framework 9, same bagging + Platt mechanism as v12):
# Ltr = single-fold-model OOF logits (what Platt is fitted on), Lte = bagged (5-model average) logits on the outer test rows.
import sys, glob, numpy as np
sys.path.insert(0, "experiments/v4")
from kit import load, sigmoid, ll, auc, norm_cost
from sklearn.linear_model import LogisticRegression
tag = sys.argv[1] if len(sys.argv) > 1 else "v7c"
def caruana(L, yy, crit, steps=40):
    c = np.zeros(len(L)); s1 = [crit(yy, L[i]) for i in range(len(L))]; c[int(np.argmax(s1))] = 1; cur = max(s1)
    for _ in range(steps - 1):
        cand = [crit(yy, ((c + np.eye(len(L))[i]) / (c.sum() + 1)) @ L) for i in range(len(L))]
        j = int(np.argmax(cand))
        if cand[j] <= cur + 1e-7: break
        cur, c[j] = cand[j], c[j] + 1
    return c / c.sum()
def platt(z, yy):
    m = LogisticRegression(C=1e6, max_iter=1000).fit(z[:, None], yy); return float(m.coef_[0, 0]), float(m.intercept_[0])
for key in ("p2", "p3", "p1"):
    X, y, md = load(key)
    files = sorted(glob.glob(f"experiments/v5/logs/outer/{tag}_{key}_s*_f*.npz"))
    rows = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        if "Ltr" not in d: continue
        Ltr, Lte, tr, va = d["Ltr"], d["Lte"], d["tr"], d["va"]
        ytr, yte = y[tr], y[va]
        crit = (lambda yy, z: auc(yy, z)) if key == "p1" else (lambda yy, z: -ll(yy, sigmoid(z)))
        w = caruana(Ltr, ytr, crit)
        ztr, zte = w @ Ltr, w @ Lte
        a, b = platt(ztr, ytr)                 # as the framework does (fitted on OOF single-model logits)
        ao, bo = platt(zte, yte)               # oracle: fitted on the bagged test logits (not attainable; diagnostic only)
        # spread of the logits: bagging should reduce the variance a little
        r = dict(sd_oof=ztr.std(), sd_bag=zte.std(), slope_oof=a, slope_oracle=ao,
                 ll_raw=ll(yte, sigmoid(zte)), ll_platt=ll(yte, sigmoid(a * zte + b)), ll_oracle=ll(yte, sigmoid(ao * zte + bo)),
                 ll_slope1=ll(yte, sigmoid(zte + bo)))
        if key == "p3":
            r["cost_platt"] = norm_cost(yte, sigmoid(a * zte + b), 1 / 11); r["cost_oracle"] = norm_cost(yte, sigmoid(ao * zte + bo), 1 / 11)
            r["cost_raw"] = norm_cost(yte, sigmoid(zte), 1 / 11)
        rows.append(r)
    if not rows: print(key, "no files"); continue
    print(f"== {key} ({len(rows)} outer folds, tag {tag})")
    for k in rows[0]:
        v = np.array([r[k] for r in rows]); print(f"   {k:12s} mean {v.mean():.5f}  (per fold: {' '.join(f'{x:.4f}' for x in v)})")
