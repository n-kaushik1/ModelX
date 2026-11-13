"""High-cardinality categoricals: how should the boosters see them?  LightGBM (fixed reasonable config, early
stopping on an inner split) on synthetic problems with a 150-300-level column, scored on the 20k test set.
  top30  : v12 encoder (30 most frequent levels, rest -> 'other')
  all    : every level as a native LightGBM categorical
  te     : top30 + out-of-fold target encoding (m-estimate, 5 folds inside the training rows) as a numeric
usage: python exp_hicard.py <pid> [<pid> ...]"""
import sys, os
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v6"))
import simgen


def te_fit(s, y, m=20):
    prior = y.mean()
    d = pd.DataFrame({"s": s.astype(str).values, "y": y}).groupby("s")["y"].agg(["sum", "count"])
    return ((d["sum"] + m * prior) / (d["count"] + m)).to_dict(), prior


def te_oof(s, y, m=20, seed=0):
    out = np.zeros(len(s))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
        mp, pr = te_fit(s.iloc[tr], y[tr], m)
        out[va] = s.iloc[va].astype(str).map(mp).fillna(pr).to_numpy()
    return out


def prep(Xtr, Xte, ytr, mode, hc):
    A, B = Xtr.copy(), Xte.copy()
    cats = [c for c in A.columns if A[c].dtype == object]
    for c in cats:
        if mode in ("top30", "te") and A[c].nunique() > 30:
            top = A[c].value_counts().index[:30]
            A[c] = A[c].where(A[c].isin(top), "other")
            B[c] = B[c].where(B[c].isin(top), "other")
        lv = sorted(A[c].dropna().astype(str).unique())
        A[c] = pd.Categorical(A[c].astype(str).where(A[c].notna(), None), categories=lv)
        B[c] = pd.Categorical(B[c].astype(str).where(B[c].notna(), None), categories=lv)
    if mode == "te":
        for c in hc:
            A[c + "__te"] = te_oof(Xtr[c], ytr)
            mp, pr = te_fit(Xtr[c], ytr)
            B[c + "__te"] = Xte[c].astype(str).map(mp).fillna(pr).to_numpy()
    return A, B


for pid in map(int, sys.argv[1:]):
    Xtr, ytr, Xte, yte, pte, md, sp = simgen.make(pid)
    hc = [c for c in Xtr.columns if Xtr[c].dtype == object and Xtr[c].nunique() > 30]
    res = {}
    for mode in ("top30", "all", "te"):
        A, B = prep(Xtr, Xte, ytr, mode, hc)
        ps = []
        for seed in range(3):
            a, v = train_test_split(np.arange(len(ytr)), test_size=0.2, stratify=ytr, random_state=seed)
            prm = dict(objective="binary", learning_rate=0.03, num_leaves=8, max_depth=3, min_child_samples=20,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8, cat_smooth=20, min_data_per_group=50,
                       verbose=-1, num_threads=2, seed=seed)
            m = lgb.train(prm, lgb.Dataset(A.iloc[a], ytr[a]), 5000, valid_sets=[lgb.Dataset(A.iloc[v], ytr[v])],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            ps.append(m.predict(B, num_iteration=m.best_iteration))
        p = np.mean(ps, 0)
        res[mode] = (roc_auc_score(yte, p), log_loss(yte, p))
    print(f"pid {pid} {sp['name']} hc={[(c, Xtr[c].nunique()) for c in hc]} bayesAUC={roc_auc_score(yte, pte):.4f} "
          + " ".join(f"{k}: AUC {a:.4f} LL {l:.4f}" for k, (a, l) in res.items()), flush=True)
