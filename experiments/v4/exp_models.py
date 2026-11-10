"""Model-level variants, repeated 5-fold CV over 3 fold seeds (structure detection nested in each fold).
Each variant's OOF predictions are cached to logs/oof_<key>_<name>.npy so blends can be studied later.
usage: python exp_models.py p1 NJ name1 name2 ..."""
import sys, time, os
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb, catboost as cb
from sklearn.linear_model import LogisticRegression
from kit import *

S = load_module(os.path.join(WORK, "sol_v5.py"), "sol_v5")
key, NJ = sys.argv[1], int(sys.argv[2])
names = sys.argv[3:]
log = Log(f"exp_models_{key}.log")
X, y, md = load(key)
cl = S.DataCleaner().fit(X)
Xc = cl.transform(X)
nums, cats = cl.numeric_, cl.categorical_
SEEDS = (0, 1, 2)
FS = {s: folds(y, s) for s in SEEDS}


def struct(Xtr, ytr):
    return S._detect_structure(Xtr, ytr, nums, cats, NJ)


# ------------------------------------------------------------------ fitters
def lgb_fit(P, es=True, seeds=(0,)):
    P = dict(P)

    def f(Xtr, ytr, Xva, yva):
        enc = S.TabularEncoder("native").fit(Xtr, nums, cats)
        Ztr, Zva = enc.transform(Xtr), enc.transform(Xva)
        kw = dict(P)
        k, mono = kw.pop("inter_k", 0), kw.pop("mono", 0)
        single = kw.pop("single", 0)          # every tree uses one feature (EBM-like mains)
        if k or mono or single:
            st = struct(Xtr, ytr)
            inter, mon = S._constraints_for(dict(inter_k=k, mono=mono), st, Ztr.shape[1])
            if single:
                inter = [[i] for i in range(Ztr.shape[1])] + ([list(p) for p in st["pairs"][:k]] if k else [])
            if inter:
                kw["interaction_constraints"] = inter
            if mon:
                kw.update(monotone_constraints=mon, monotone_constraints_method="advanced")
        n = kw.pop("n_estimators", 5000)
        out = np.zeros(len(Zva))
        for s in seeds:
            m = lgb.LGBMClassifier(n_estimators=n, verbose=-1, n_jobs=NJ, random_state=s, cat_smooth=20,
                                   min_data_per_group=50, **kw)
            cbs = [lgb.early_stopping(200, verbose=False)] if es else []
            m.fit(Ztr, ytr, eval_set=[(Zva, yva)] if es else None, callbacks=cbs,
                  categorical_feature=enc.cat_idx_ or "auto")
            out += m.predict_proba(Zva, raw_score=True)
        return sigmoid(out / len(seeds))
    return f


def ebm_fit(main_P, pair_P, k_pairs, bags=1, es=True):
    """EBM-like two stages: (1) mains only (each tree = one feature), (2) pairs on the residual."""
    def f(Xtr, ytr, Xva, yva):
        enc = S.TabularEncoder("native").fit(Xtr, nums, cats)
        Ztr, Zva = enc.transform(Xtr), enc.transform(Xva)
        p = Ztr.shape[1]
        st = struct(Xtr, ytr) if k_pairs else None
        out = np.zeros(len(Zva))
        for b in range(bags):
            kw = dict(main_P)
            m1 = lgb.LGBMClassifier(n_estimators=8000, verbose=-1, n_jobs=NJ, random_state=b, cat_smooth=20,
                                    min_data_per_group=50, interaction_constraints=[[i] for i in range(p)], **kw)
            m1.fit(Ztr, ytr, eval_set=[(Zva, yva)], callbacks=[lgb.early_stopping(300, verbose=False)],
                   categorical_feature=enc.cat_idx_ or "auto")
            itr, iva = m1.predict_proba(Ztr, raw_score=True), m1.predict_proba(Zva, raw_score=True)
            if k_pairs and st["pairs"]:
                m2 = lgb.LGBMClassifier(n_estimators=4000, verbose=-1, n_jobs=NJ, random_state=b, cat_smooth=20,
                                        min_data_per_group=50,
                                        interaction_constraints=[list(q) for q in st["pairs"][:k_pairs]], **pair_P)
                m2.fit(Ztr, ytr, init_score=itr, eval_set=[(Zva, yva)], eval_init_score=[iva],
                       callbacks=[lgb.early_stopping(200, verbose=False)], categorical_feature=enc.cat_idx_ or "auto")
                iva = iva + m2.predict_proba(Zva, raw_score=True)
            out += iva
        return sigmoid(out / bags)
    return f


def xgb_fit(P, seeds=(0,)):
    def f(Xtr, ytr, Xva, yva):
        enc = S.TabularEncoder("native").fit(Xtr, nums, cats)
        Ztr, Zva = enc.transform(Xtr), enc.transform(Xva)
        kw = dict(P)
        k, mono = kw.pop("inter_k", 0), kw.pop("mono", 0)
        if k or mono:
            inter, mon = S._constraints_for(dict(inter_k=k, mono=mono), struct(Xtr, ytr), Ztr.shape[1])
            if inter:
                kw["interaction_constraints"] = str([[int(i) for i in g] for g in inter])
            if mon:
                kw["monotone_constraints"] = "(" + ",".join(str(int(m)) for m in mon) + ")"
        out = np.zeros(len(Zva))
        for s in seeds:
            m = xgb.XGBClassifier(n_estimators=6000, tree_method="hist", n_jobs=NJ, random_state=s, verbosity=0,
                                  eval_metric="logloss", early_stopping_rounds=200, **kw)
            m.fit(Ztr, ytr, eval_set=[(Zva, yva)], verbose=False)
            out += m.predict(xgb.DMatrix(Zva), output_margin=True) if False else logit(m.predict_proba(Zva)[:, 1])
        return sigmoid(out / len(seeds))
    return f


def cat_fit(P, mono=False):
    def f(Xtr, ytr, Xva, yva):
        enc = S.TabularEncoder("frame", max_levels=1000).fit(Xtr, nums, cats)
        Ztr, Zva = enc.transform(Xtr), enc.transform(Xva)
        kw = dict(P)
        if mono:
            st = struct(Xtr, ytr)
            kw["monotone_constraints"] = [int(m) if j < len(nums) else 0 for j, m in enumerate(st["mono"])]
        m = cb.CatBoostClassifier(iterations=6000, thread_count=NJ, verbose=0, allow_writing_files=False,
                                  od_type="Iter", od_wait=200, random_seed=0,
                                  cat_features=enc.cat_idx_ or None, **kw)
        m.fit(Ztr, ytr, eval_set=(Zva, yva), use_best_model=True)
        return m.predict_proba(Zva)[:, 1]
    return f


def gam_fit(C=0.2, knots=7, basis="hinge", two_sided=False, pairs=0, pair_knots=0):
    def f(Xtr, ytr, Xva, yva):
        enc = S.TabularEncoder("linear").fit(Xtr, nums, cats)
        Ztr, Zva = enc.transform(Xtr), enc.transform(Xva)
        nn = len(nums)
        cols = [j for j in range(nn) if len(np.unique(Ztr[:, j])) > 10]
        qs = np.linspace(0.5 / knots, 1 - 0.5 / knots, knots)
        K = np.array([np.quantile(Ztr[:, j], qs) for j in cols])
        prs = []
        if pairs:
            st = struct(Xtr, ytr)
            prs = [(a, b) for a, b in st["pairs"] if a < nn and b < nn][:pairs]

        def design(Z):
            B = [Z]
            S_ = Z[:, cols]
            B.append(np.maximum(S_[:, :, None] - K[None], 0).reshape(len(Z), -1))
            if two_sided:
                B.append(np.maximum(K[None] - S_[:, :, None], 0).reshape(len(Z), -1))
            for a, b in prs:
                B.append((Z[:, a] * Z[:, b])[:, None])
                if pair_knots:
                    ka = np.quantile(Ztr[:, a], [0.25, 0.5, 0.75]); kb = np.quantile(Ztr[:, b], [0.25, 0.5, 0.75])
                    for u in ka:
                        for v in kb:
                            B.append((np.maximum(Z[:, a] - u, 0) * np.maximum(Z[:, b] - v, 0))[:, None])
            return np.hstack(B)
        D = design(Ztr)
        mu, sd = D[:, nn:].mean(0), D[:, nn:].std(0) + 1e-9
        def std(Dm):
            Dm = Dm.copy(); Dm[:, nn:] = (Dm[:, nn:] - mu) / sd; return Dm
        lr = LogisticRegression(C=C, max_iter=5000).fit(std(D), ytr)
        return lr.predict_proba(std(design(Zva)))[:, 1]
    return f


def v5gam_fit(C=0.05, knots=7):
    def f(Xtr, ytr, Xva, yva):
        enc = S.TabularEncoder("linear").fit(Xtr, nums, cats)
        m = S.SplineLogit(len(nums), C=C, n_knots=knots, basis="hinge").fit(enc.transform(Xtr), ytr)
        return m.predict_proba(enc.transform(Xva))[:, 1]
    return f


def et_fit(min_leaf=5, mf=0.4):
    from sklearn.ensemble import ExtraTreesClassifier
    def f(Xtr, ytr, Xva, yva):
        enc = S.TabularEncoder("imputed").fit(Xtr, nums, cats)
        m = ExtraTreesClassifier(n_estimators=500, min_samples_leaf=min_leaf, max_features=mf, n_jobs=NJ,
                                 random_state=0).fit(enc.transform(Xtr), ytr)
        return m.predict_proba(enc.transform(Xva))[:, 1]
    return f


B = dict(learning_rate=0.03, max_depth=2, num_leaves=4, min_child_samples=20, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.8, reg_lambda=1.0, inter_k=8, mono=1)
U = dict(learning_rate=0.03, max_depth=2, num_leaves=4, min_child_samples=20, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.8, reg_lambda=1.0)
EBM_MAIN = dict(learning_rate=0.02, num_leaves=3, min_child_samples=20, subsample=0.85, subsample_freq=1,
                colsample_bytree=1.0, reg_lambda=0.0)
EBM_PAIR = dict(learning_rate=0.02, num_leaves=4, max_depth=2, min_child_samples=30, subsample=0.85, subsample_freq=1,
                colsample_bytree=1.0, reg_lambda=1.0)

V = {
    "B": lgb_fit(B),
    "U_d2": lgb_fit(U),
    "B_lr01": lgb_fit(dict(B, learning_rate=0.01)),
    "B_ss05": lgb_fit(dict(B, subsample=0.5)),
    "B_mcs60": lgb_fit(dict(B, min_child_samples=60)),
    "B_k16": lgb_fit(dict(B, inter_k=16)),
    "B_3seeds": lgb_fit(B, seeds=(0, 1, 2)),
    "B_l3": lgb_fit(dict(B, num_leaves=3)),
    "B_cs05": lgb_fit(dict(B, colsample_bytree=0.5)),
    "B_lr01_ss06_cs05": lgb_fit(dict(B, learning_rate=0.01, subsample=0.6, colsample_bytree=0.5)),
    "single_l3": lgb_fit(dict(EBM_MAIN, single=1)),
    "single_l3_k8": lgb_fit(dict(EBM_MAIN, single=1, inter_k=8, max_depth=2, num_leaves=4)),
    "ebm_k0": ebm_fit(EBM_MAIN, EBM_PAIR, 0),
    "ebm_k8": ebm_fit(EBM_MAIN, EBM_PAIR, 8),
    "ebm_k16": ebm_fit(EBM_MAIN, EBM_PAIR, 16),
    "ebm_k8_mono": ebm_fit(EBM_MAIN, EBM_PAIR, 8),
    "xgb_d2_k8m": xgb_fit(dict(learning_rate=0.03, max_depth=2, min_child_weight=5, subsample=0.8,
                               colsample_bytree=0.8, reg_lambda=1.0, inter_k=8, mono=1)),
    "xgb_d3": xgb_fit(dict(learning_rate=0.03, max_depth=3, min_child_weight=5, subsample=0.8,
                           colsample_bytree=0.7, reg_lambda=1.0)),
    "cat_d4": cat_fit(dict(learning_rate=0.05, depth=4, l2_leaf_reg=5)),
    "cat_d4_lr03": cat_fit(dict(learning_rate=0.03, depth=4, l2_leaf_reg=5)),
    "cat_d6_lr03": cat_fit(dict(learning_rate=0.03, depth=6, l2_leaf_reg=5)),
    "cat_d3_lr03_m": cat_fit(dict(learning_rate=0.03, depth=3, l2_leaf_reg=5), mono=True),
    "v5gam_c05": v5gam_fit(0.05, 7),
    "v5gam_c2": v5gam_fit(0.2, 7),
    "et_l5": et_fit(5, 0.4),
    "et_l15": et_fit(15, 0.5),
    "xgb_d2": xgb_fit(dict(learning_rate=0.03, max_depth=2, min_child_weight=5, subsample=0.8,
                           colsample_bytree=0.8, reg_lambda=1.0)),
    "xgb_d3_k16m": xgb_fit(dict(learning_rate=0.03, max_depth=3, min_child_weight=5, subsample=0.8,
                                colsample_bytree=0.7, reg_lambda=1.0, inter_k=16, mono=1)),
    "B_d3_k16": lgb_fit(dict(B, max_depth=3, num_leaves=8, inter_k=16)),
    "U_d1": lgb_fit(dict(U, max_depth=1, num_leaves=2, learning_rate=0.05, subsample=1.0, colsample_bytree=1.0,
                         reg_lambda=0.0)),
    "U_d4": lgb_fit(dict(U, max_depth=4, num_leaves=15, min_child_samples=40, colsample_bytree=0.7)),
    "gam7": gam_fit(0.2, 7),
    "gam7_c05": gam_fit(0.05, 7),
    "gam12_2s": gam_fit(0.05, 12, two_sided=True),
    "gam20_c02": gam_fit(0.02, 20),
    "gam12_p8": gam_fit(0.05, 12, pairs=8),
    "gam12_p8k": gam_fit(0.05, 12, pairs=8, pair_knots=1),
}

for name in names:
    t0 = time.time()
    oofs, scs = [], []
    for s in SEEDS:
        oof = np.zeros(len(y))
        for tr, va in FS[s]:
            oof[va] = V[name](Xc.iloc[tr], y[tr], Xc.iloc[va], y[va])
        oofs.append(oof)
        scs.append(primary(key, y, oof))
    np.save(f"{WORK}/logs/oof_{key}_{name}.npy", np.array(oofs))
    lls = [ll(y, o) for o in oofs]
    extra = f" cost@1/11cal={np.mean([platt_cv_cost(y, o) for o in oofs]):.4f}" if key == "p3" else ""
    log(f"{key} {name:22s} primary={np.mean(scs):.5f} sd={np.std(scs):.5f} LL={np.mean(lls):.5f} "
        f"per-seed={[round(s, 5) for s in scs]}{extra} {time.time()-t0:.0f}s")
