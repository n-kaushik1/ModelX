"""Nested test of supervised redundancy pruning: inside every training fold, candidates = numeric columns that
are >= R2_MIN explained by the other numeric columns; greedily drop a candidate if a quick reference booster's
CV log loss improves by more than DELTA without it.  Then fit the reference model B on the kept columns and score
the held-out fold.  Compared with no pruning on the same folds.  usage: python exp_prune.py <key> <n_jobs> [seeds]"""
import sys, os, time
import numpy as np, lightgbm as lgb
from sklearn.model_selection import KFold
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v4"))
from kit import *

HERE = os.path.dirname(os.path.abspath(__file__))
S = load_module(os.path.join(HERE, "..", "..", "submission", "v6", "solution.py"), "sol_v8")
key, NJ = sys.argv[1], int(sys.argv[2])
SEEDS = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0, 1]
R2_MIN, DELTA = 0.9, 1e-4
X, y, md = load(key)
cl = S.DataCleaner().fit(X); Xc0 = cl.transform(X)
d, tw = S._find_redundant(Xc0, cl.numeric_); cl.set_redundancy(d, tw); Xc = cl.transform(X)
NUMS, CATS = cl.numeric_, cl.categorical_
logf = open(os.path.join(HERE, "logs", f"exp_prune_{key}.log"), "a")
def log(s):
    print(s, flush=True); logf.write(s + "\n"); logf.flush()

B = dict(learning_rate=0.03, max_depth=2, num_leaves=4, min_child_samples=20, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.8, reg_lambda=1.0)


def explained(Xtr, nums):
    R = Xtr[nums].rank(pct=True).fillna(0.5).to_numpy()
    if len(R) > 6000:
        R = R[np.random.default_rng(0).choice(len(R), 6000, replace=False)]
    out = {}
    for j, c in enumerate(nums):
        t, F = R[:, j], np.delete(R, j, axis=1)
        p = np.zeros(len(t))
        for a, b in KFold(3, shuffle=True, random_state=0).split(F):
            p[b] = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.1, num_leaves=15, verbose=-1, n_jobs=NJ).fit(F[a], t[a]).predict(F[b])
        out[c] = 1 - np.mean((t - p) ** 2) / max(np.var(t), 1e-12)
    return out


def quick_ll(Xtr, ytr, nums, cats):
    enc = S.TabularEncoder("native").fit(Xtr, nums, cats)
    Z = enc.transform(Xtr)
    oof, n = np.zeros((2, len(ytr))), 0
    for r in range(2):
        for a, b in folds(ytr, 50 + r):
            m = lgb.LGBMClassifier(n_estimators=400, verbose=-1, n_jobs=NJ, random_state=r, **B)
            m.fit(Z[a], ytr[a], categorical_feature=enc.cat_idx_ or "auto")
            oof[r, b] = m.predict_proba(Z[b])[:, 1]
    return np.mean([ll(ytr, o) for o in oof])


def prune(Xtr, ytr):
    nums = list(NUMS)
    r2 = explained(Xtr, nums)
    cand = sorted((v, c) for c, v in r2.items() if v >= R2_MIN)[::-1]
    cur = quick_ll(Xtr, ytr, nums, CATS)
    dropped = []
    for v, c in cand:
        trial = [x for x in nums if x != c]
        l2 = quick_ll(Xtr, ytr, trial, CATS)
        if l2 < cur - DELTA:
            nums, cur = trial, l2
            dropped.append(c)
    return nums, dropped


def fit_B(Xtr, ytr, Xva, yva, nums):
    enc = S.TabularEncoder("native").fit(Xtr, nums, CATS)
    st = S._detect_structure(Xtr, ytr, nums, CATS, NJ)
    inter, mon = S._constraints_for({"inter_k": 8, "mono": 1}, st, len(enc.feature_names_))
    m = lgb.LGBMClassifier(n_estimators=5000, verbose=-1, n_jobs=NJ, random_state=0, cat_smooth=20, min_data_per_group=50,
                           interaction_constraints=inter, monotone_constraints=mon, monotone_constraints_method="advanced", **B)
    Ztr, Zva = enc.transform(Xtr), enc.transform(Xva)
    m.fit(Ztr, ytr, eval_set=[(Zva, yva)], categorical_feature=enc.cat_idx_ or "auto",
          callbacks=[lgb.early_stopping(200, verbose=False)])
    return m.predict_proba(Zva)[:, 1]


for s in SEEDS:
    t0 = time.time()
    base, pr = np.zeros(len(y)), np.zeros(len(y))
    drops = []
    for tr, va in folds(y, s):
        Xtr, Xva = Xc.iloc[tr], Xc.iloc[va]
        base[va] = fit_B(Xtr, y[tr], Xva, y[va], NUMS)
        nums, dr = prune(Xtr, y[tr])
        drops.append(dr)
        pr[va] = fit_B(Xtr, y[tr], Xva, y[va], nums)
    log(f"{key} seed{s} base={primary(key, y, base):.5f} pruned={primary(key, y, pr):.5f} LL {ll(y, base):.5f}->{ll(y, pr):.5f} "
        f"drops={drops} {time.time()-t0:.0f}s")
