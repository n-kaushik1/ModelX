"""Final predictor for GAM-type members: k-fold bag (5 models on 80 %) vs one refit on 100 % of the training part.
Outer 5-fold x 2 seeds; the member configuration is fixed (v12 ladder rungs).  usage: python exp_refit.py <key> <n_jobs>"""
import sys, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "v4"))
from kit import load, load_module, folds, ll, auc, logit, sigmoid
S = load_module(os.path.join(HERE, "..", "..", "submission", "v12", "solution.py"), "sol_v14")
key, NJ = sys.argv[1], int(sys.argv[2])
X, y, md = load(key); obj = S.Objective(md)
cl = S.DataCleaner().fit(X); Xc0 = cl.transform(X); d, tw = S._find_redundant(Xc0, cl.numeric_); cl.set_redundancy(d, tw); Xc = cl.transform(X)
nums, cats = cl.numeric_, cl.categorical_
G = dict(C=0.05, knots=15, basis="hinge", pairs=6, cpairs=2, penalty="l1", phinge=6, chinge=2)
H = dict(C=0.05, knots=15, pairs=6, cpairs=2, phinge=6, chinge=2, depth=2, leaves=4, lr=0.01, min_leaf=20, subsample=0.8, colsample=0.8, l2=1.0)
res = {}
for fam, P in (("gam", G), ("hyb", H)):
    for s in (0, 1):
        bag, ref = np.zeros(len(y)), np.zeros(len(y))
        for tr, va in folds(y, 100 + s):
            Xtr, ytr, Xva = Xc.iloc[tr].reset_index(drop=True), y[tr], Xc.iloc[va].reset_index(drop=True)
            P2 = dict(P)
            if fam == "hyb":                       # tree count from early stopping on inner folds, as in tuning
                _, nt = S._oof_predict(S.ModelSpec(fam, P2), Xtr, ytr, nums, cats, folds(ytr, 7), obj, NJ, early_stop=True)
                P2["n_estimators"] = int(max(20, nt))
            z = np.zeros(len(va))
            for itr, iva in folds(ytr, 11):
                m = S._fit_model(S.ModelSpec(fam, P2), Xtr.iloc[itr], ytr[itr], nums, cats, obj, NJ)
                z += logit(m.predict(Xva)) / 5
            bag[va] = sigmoid(z)
            P3 = dict(P2)
            if fam == "hyb":
                P3["n_estimators"] = int(P2["n_estimators"] * 1.125)
            ref[va] = S._fit_model(S.ModelSpec(fam, P3), Xtr, ytr, nums, cats, obj, NJ).predict(Xva)
        m_ = (lambda p: auc(y, p)) if key == "p1" else (lambda p: ll(y, p))
        res.setdefault(fam, []).append((m_(bag), m_(ref)))
        print(f"{key} {fam} seed{s} bag5={m_(bag):.5f} refit={m_(ref):.5f}", flush=True)
for fam, r in res.items():
    print(f"{key} {fam} MEAN bag5={np.mean([a for a, _ in r]):.5f} refit={np.mean([b for _, b in r]):.5f}")
