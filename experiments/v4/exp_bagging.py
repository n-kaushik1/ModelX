"""Final-predictor study: refit-on-all (v5 style) vs average of inner fold models (k-fold bagging).
Outer 5-fold x 2 seeds.  Inside each outer-train: inner 5-fold with early stopping gives the tree count and
the fold models.  Predictors for the outer-test rows:
  refit1   : one model on all outer-train rows, trees = mean best_iter * (1 + 0.5/(k-1))
  refit3   : same, 3 seeds averaged in logit space (v5)
  bag5     : average (logit) of the 5 inner fold models (each stopped on its own inner val fold)
  bag5+refit3 : logit average of the two
Also reports calibration slope of each predictor on the outer-test rows (1 = calibrated)."""
import sys, os, time
import numpy as np, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from kit import *

S = load_module(os.path.join(WORK, "sol_v5.py"), "sol_v5")
key, NJ = sys.argv[1], int(sys.argv[2])
log = Log(f"exp_bagging_{key}.log")
X, y, md = load(key)
cl = S.DataCleaner().fit(X)
Xc = cl.transform(X)
nums, cats = cl.numeric_, cl.categorical_
P = dict(learning_rate=0.03, max_depth=2, num_leaves=4, min_child_samples=20, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.8, reg_lambda=1.0)
K = {"inter_k": 8, "mono": 1}


def prep(Xtr, ytr):
    enc = S.TabularEncoder("native").fit(Xtr, nums, cats)
    st = S._detect_structure(Xtr, ytr, nums, cats, NJ)
    inter, mon = S._constraints_for(K, st, len(enc.feature_names_))
    kw = dict(P, interaction_constraints=inter, monotone_constraints=mon, monotone_constraints_method="advanced")
    return enc, kw


def model(kw, n, seed):
    return lgb.LGBMClassifier(n_estimators=n, verbose=-1, n_jobs=NJ, random_state=seed, cat_smooth=20,
                              min_data_per_group=50, **kw)


preds = {k: np.zeros((2, len(y))) for k in ["refit1", "refit3", "bag5", "bag5+refit3", "bag5_3s"]}
for rep in range(2):
    for tr, te in folds(y, 100 + rep):
        Xtr, ytr, Xte = Xc.iloc[tr].reset_index(drop=True), y[tr], Xc.iloc[te].reset_index(drop=True)
        enc, kw = prep(Xtr, ytr)
        Ztr, Zte = enc.transform(Xtr), enc.transform(Xte)
        iters, bag, bag3 = [], np.zeros(len(te)), np.zeros(len(te))
        for itr, iva in folds(ytr, rep):
            for s in range(3):
                m = model(kw, 5000, s)
                m.fit(Ztr[itr], ytr[itr], eval_set=[(Ztr[iva], ytr[iva])], categorical_feature=enc.cat_idx_ or "auto",
                      callbacks=[lgb.early_stopping(200, verbose=False)])
                z = m.predict_proba(Zte, raw_score=True)
                if s == 0:
                    iters.append(m.best_iteration_)
                    bag += z / 5
                bag3 += z / 15
        n = int(np.mean(iters) * 1.125)
        zs = []
        for s in range(3):
            m = model(kw, n, s).fit(Ztr, ytr, categorical_feature=enc.cat_idx_ or "auto")
            zs.append(m.predict_proba(Zte, raw_score=True))
        preds["refit1"][rep, te] = sigmoid(zs[0])
        preds["refit3"][rep, te] = sigmoid(np.mean(zs, 0))
        preds["bag5"][rep, te] = sigmoid(bag)
        preds["bag5_3s"][rep, te] = sigmoid(bag3)
        preds["bag5+refit3"][rep, te] = sigmoid(0.5 * bag + 0.5 * np.mean(zs, 0))
    log(f"{key} rep{rep} done")
for k, v in preds.items():
    sc = [primary(key, y, v[r]) for r in range(2)]
    slope = [LogisticRegression(C=1e6).fit(logit(v[r])[:, None], y).coef_[0, 0] for r in range(2)]
    extra = f" cost@1/11={np.mean([norm_cost(y, v[r], 1/11) for r in range(2)]):.4f}" if key == "p3" else ""
    log(f"{key} {k:12s} primary={np.mean(sc):.5f} per-rep={[round(s, 5) for s in sc]} LL={np.mean([ll(y, v[r]) for r in range(2)]):.5f} "
        f"calib-slope={np.mean(slope):.3f}{extra}")
