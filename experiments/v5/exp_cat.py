"""CatBoost variants, repeated 5-fold CV over 3 fold seeds (same protocol as v4/exp_models.py; monotone
directions from the nested structure detection).  usage: python exp_cat.py <key> <n_jobs> name ..."""
import sys, os, time
import numpy as np, catboost as cb
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v4"))
from kit import *

HERE = os.path.dirname(os.path.abspath(__file__))
S = load_module(os.path.join(HERE, "..", "..", "submission", "v4", "solution.py"), "sol_v6")
key, NJ = sys.argv[1], int(sys.argv[2])
names = sys.argv[3:]
X, y, md = load(key)
cl = S.DataCleaner().fit(X)
Xc = cl.transform(X)
nums, cats = cl.numeric_, cl.categorical_
logf = open(os.path.join(HERE, "logs", f"exp_cat_{key}.log"), "a")
def log(s):
    print(s, flush=True); logf.write(s + "\n"); logf.flush()


def cat_fit(P, mono=True):
    def f(Xtr, ytr, Xva, yva):
        enc = S.TabularEncoder("frame", max_levels=1000).fit(Xtr, nums, cats)
        Ztr, Zva = enc.transform(Xtr), enc.transform(Xva)
        kw = dict(P)
        if kw.pop("dt_n", 0):
            kw["diffusion_temperature"] = float(len(ytr))
        if mono:
            st = S._detect_structure(Xtr, ytr, nums, cats, NJ)
            kw["monotone_constraints"] = [int(m) if j < len(nums) else 0 for j, m in enumerate(st["mono"])]
        m = cb.CatBoostClassifier(iterations=8000, thread_count=NJ, verbose=0, allow_writing_files=False,
                                  od_type="Iter", od_wait=200, random_seed=0, cat_features=enc.cat_idx_ or None, **kw)
        m.fit(Ztr, ytr, eval_set=(Zva, yva), use_best_model=True)
        return m.predict_proba(Zva)[:, 1], m.get_best_iteration()
    return f


R = dict(learning_rate=0.03, depth=3, l2_leaf_reg=5)
V = {"ref_d3m": cat_fit(R),
     "d2m": cat_fit(dict(R, depth=2)),
     "d3m_lgv": cat_fit(dict(R, langevin=True, dt_n=1)),
     "d3m_l2_20": cat_fit(dict(R, l2_leaf_reg=20)),
     "d4m": cat_fit(dict(R, depth=4)),
     "d3m_rs0": cat_fit(dict(R, random_strength=0.2)),
     "d3m_lr015": cat_fit(dict(R, learning_rate=0.015))}
for name in names:
    t0 = time.time()
    oofs, scs, its = [], [], []
    for s in (0, 1, 2):
        oof = np.zeros(len(y))
        for tr, va in folds(y, s):
            oof[va], it = V[name](Xc.iloc[tr], y[tr], Xc.iloc[va], y[va])
            its.append(it)
        oofs.append(oof)
        scs.append(primary(key, y, oof))
    np.save(os.path.join(HERE, "logs", f"oofcat_{key}_{name}.npy"), np.array(oofs))
    extra = f" cost@1/11cal={np.mean([platt_cv_cost(y, o) for o in oofs]):.4f}" if key == "p3" else ""
    log(f"{key} {name:12s} primary={np.mean(scs):.5f} LL={np.mean([ll(y, o) for o in oofs]):.5f} "
        f"per-seed={[round(v, 5) for v in scs]}{extra} iters={int(np.mean(its))} {time.time()-t0:.0f}s")
