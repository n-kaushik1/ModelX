"""Category x category terms for the GAM: does a combined-level column for the measured cat x cat pairs help?
Single sparse GAM (v13 code, best L1 rung) with and without the top-k measured categorical pairs, fitted on the
training rows of a synthetic problem and scored on its 20k test rows against the truth.
usage: python exp_catcat.py <pid> [<pid> ...]"""
import sys, os, importlib.util
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "v6"))
import simgen
spec = importlib.util.spec_from_file_location("solution", os.path.join(HERE, "..", "..", "submission", "v13", "solution.py"))
m = importlib.util.module_from_spec(spec); sys.modules["solution"] = m; spec.loader.exec_module(m)

P_GAM = dict(C=0.1, knots=15, basis="hinge", pairs=4, cpairs=2, penalty="l1", phinge=4, chinge=2)
P_HYB = dict(C=0.1, knots=15, pairs=4, cpairs=2, phinge=4, chinge=2, depth=2, leaves=4, lr=0.03, min_leaf=20,
             subsample=0.8, colsample=0.8, l2=1.0, n_estimators=600)


def add_pairs(Xc, Xt, num, cat, y, k):
    st = m._detect_structure(Xc, y, num, cat, 2)
    nn = len(num)
    prs = [(cat[a - nn], cat[b - nn]) for a, b in st["pairs"] if a >= nn and b >= nn][:k]
    A, B = Xc.copy(), Xt.copy()
    for a, b in prs:
        for D in (A, B):
            D[f"{a}|{b}"] = (D[a].astype(str) + "|" + D[b].astype(str)).where(D[a].notna() & D[b].notna(), None)
    return A, B, cat + [f"{a}|{b}" for a, b in prs], prs


for pid in map(int, sys.argv[1:]):
    Xtr, ytr, Xte, yte, pte, md, sp = simgen.make(pid)
    cl = m.DataCleaner([], None).fit(Xtr)
    Xc, Xt = cl.transform(Xtr), cl.transform(Xte)
    num, cat = cl.numeric_, cl.categorical_
    obj = m.Objective(md)
    res = []
    for k in (0, 2, 4):
        m._STRUCT_CACHE.clear()
        A, B, cats, prs = add_pairs(Xc, Xt, num, cat, ytr, k) if k else (Xc, Xt, cat, [])
        for fam, P in (("gam", P_GAM), ("hyb", P_HYB)):
            m._STRUCT_CACHE.clear()
            fm = m._fit_model(m.ModelSpec(fam, P, seed=42), A, ytr, num, cats, obj, 2)
            p = fm.predict(B)
            res.append(f"{fam} k={k}: AUC {roc_auc_score(yte, p):.4f} LL {log_loss(yte, p):.4f}")
    print(f"pid {pid} {sp['name']} bayes AUC {roc_auc_score(yte, pte):.4f} LL {log_loss(yte, pte):.4f} | " + " | ".join(res),
          flush=True)
