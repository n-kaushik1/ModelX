"""Bag design for the final predictor (reference model B = LightGBM depth 2, 8 measured pairs, monotone).
Outer 5-fold x 2 repeats; inside each outer-train the bag members are early-stopped on their own held-out part.
  bag5     : 5-fold (each model sees 80 %)                     [v4]
  bag10    : 10-fold (each model sees 90 %)
  bag5x2   : two different 5-fold partitions (10 models, 80 %)
  bag5x3   : three 5-fold partitions (15 models)
  bag10x2  : two 10-fold partitions (20 models, 90 %)
usage: python exp_kfold.py <key> <n_jobs>"""
import sys, os
import numpy as np, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v4"))
from kit import *

HERE = os.path.dirname(os.path.abspath(__file__))
S = load_module(os.path.join(HERE, "..", "..", "submission", "v4", "solution.py"), "sol_v6")
key, NJ = sys.argv[1], int(sys.argv[2])
X, y, md = load(key)
cl = S.DataCleaner().fit(X)
Xc = cl.transform(X)
nums, cats = cl.numeric_, cl.categorical_
P = dict(learning_rate=0.03, max_depth=2, num_leaves=4, min_child_samples=20, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.8, reg_lambda=1.0)
logf = open(os.path.join(HERE, "logs", f"exp_kfold_{key}.log"), "a")
def log(s):
    print(s, flush=True); logf.write(s + "\n"); logf.flush()

designs = {"bag5": [(5, 0)], "bag10": [(10, 0)], "bag5x2": [(5, 0), (5, 1)], "bag5x3": [(5, 0), (5, 1), (5, 2)],
           "bag10x2": [(10, 0), (10, 1)]}
preds = {k: np.zeros((2, len(y))) for k in designs}
for rep in range(2):
    for tr, te in folds(y, 100 + rep):
        Xtr, ytr, Xte = Xc.iloc[tr].reset_index(drop=True), y[tr], Xc.iloc[te].reset_index(drop=True)
        enc = S.TabularEncoder("native").fit(Xtr, nums, cats)
        st = S._detect_structure(Xtr, ytr, nums, cats, NJ)
        inter, mon = S._constraints_for({"inter_k": 8, "mono": 1}, st, len(enc.feature_names_))
        kw = dict(P, interaction_constraints=inter, monotone_constraints=mon, monotone_constraints_method="advanced")
        Ztr, Zte = enc.transform(Xtr), enc.transform(Xte)
        cache = {}
        for name, parts in designs.items():
            zs = []
            for k, s in parts:
                if (k, s) not in cache:
                    out = []
                    for itr, iva in folds(ytr, 1000 * rep + s, k):
                        m = lgb.LGBMClassifier(n_estimators=5000, verbose=-1, n_jobs=NJ, random_state=s, cat_smooth=20,
                                               min_data_per_group=50, **kw)
                        m.fit(Ztr[itr], ytr[itr], eval_set=[(Ztr[iva], ytr[iva])], categorical_feature=enc.cat_idx_ or "auto",
                              callbacks=[lgb.early_stopping(200, verbose=False)])
                        out.append(m.predict_proba(Zte, raw_score=True))
                    cache[(k, s)] = out
                zs += cache[(k, s)]
            preds[name][rep, te] = sigmoid(np.mean(zs, 0))
    log(f"{key} rep{rep} done")
for k, v in preds.items():
    sc = [primary(key, y, v[r]) for r in range(2)]
    slope = [LogisticRegression(C=1e6).fit(logit(v[r])[:, None], y).coef_[0, 0] for r in range(2)]
    extra = f" cost@1/11={np.mean([norm_cost(y, v[r], 1/11) for r in range(2)]):.4f}" if key == "p3" else ""
    log(f"{key} {k:8s} primary={np.mean(sc):.5f} per-rep={[round(s, 5) for s in sc]} LL={np.mean([ll(y, v[r]) for r in range(2)]):.5f} "
        f"calib-slope={np.mean(slope):.3f}{extra}")
