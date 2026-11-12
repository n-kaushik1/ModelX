"""Model zoo for ensemble studies: honest OOF predictions of many diverse candidates on fixed folds.
Every fold's model early-stops on an inner 15 % split of its own training part (never on the fold it predicts).
Data: the v6 cleaner (noisy copies removed, twin readings averaged).  Two fold seeds (0, 1) per candidate.
usage: python zoo.py <key> <name> <n_jobs>   -> logs/zoo_<key>_<name>.npy  (2 x n probabilities)"""
import sys, os, time
import numpy as np, lightgbm as lgb, xgboost as xgb, catboost as cb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "v4"))
from kit import load, load_module, folds, ll, auc, sigmoid, logit

S = load_module(os.path.join(HERE, "..", "..", "submission", os.environ.get("ZOO_SOL", "v6"), "solution.py"), "sol_zoo")
key, name, NJ = sys.argv[1], sys.argv[2], int(sys.argv[3])
X, y, md = load(key)
cl = S.DataCleaner().fit(X); Xc0 = cl.transform(X)
d, tw = S._find_redundant(Xc0, cl.numeric_); cl.set_redundancy(d, tw); Xc = cl.transform(X)
nums, cats = cl.numeric_, cl.categorical_


def split_es(ytr):
    return train_test_split(np.arange(len(ytr)), test_size=0.15, stratify=ytr, random_state=11)


def struct(Xtr, ytr):
    return S._detect_structure(Xtr, ytr, nums, cats, NJ)


def lgb_f(P, cw=None, linear=False, extra=False, single=False):
    def f(Xtr, ytr, Xva):
        enc = S.TabularEncoder("native").fit(Xtr, nums, cats)
        Ztr, Zva = enc.transform(Xtr), enc.transform(Xva)
        kw = dict(P); k, mono = kw.pop("inter_k", 0), kw.pop("mono", 0)
        if k or mono or single:
            inter, mon = S._constraints_for(dict(inter_k=k, mono=mono), struct(Xtr, ytr), Ztr.shape[1])
            if single:
                inter = [[i] for i in range(Ztr.shape[1])]
            if inter: kw["interaction_constraints"] = inter
            if mon: kw.update(monotone_constraints=mon, monotone_constraints_method="advanced")
        if cw: kw["scale_pos_weight"] = cw
        if linear: kw.update(linear_tree=True, linear_lambda=1.0)
        if extra: kw["extra_trees"] = True
        a, b = split_es(ytr)
        m = lgb.LGBMClassifier(n_estimators=8000, verbose=-1, n_jobs=NJ, random_state=0, cat_smooth=20, min_data_per_group=50, **kw)
        m.fit(Ztr[a], ytr[a], eval_set=[(Ztr[b], ytr[b])], categorical_feature=enc.cat_idx_ or "auto",
              callbacks=[lgb.early_stopping(200, verbose=False)])
        n = int(m.best_iteration_ * 1.1)
        m = lgb.LGBMClassifier(n_estimators=max(n, 20), verbose=-1, n_jobs=NJ, random_state=0, cat_smooth=20, min_data_per_group=50, **kw)
        m.fit(Ztr, ytr, categorical_feature=enc.cat_idx_ or "auto")
        return m.predict_proba(Zva)[:, 1]
    return f


def lgb_prod_f(P, n_prod):
    """Booster with extra features: products of the z-scores of the top measured numeric pairs."""
    base = lgb_f(P)
    def f(Xtr, ytr, Xva):
        global nums
        st = struct(Xtr, ytr)
        nn = len(nums)
        prs = [(a, b) for a, b in st["pairs"] if a < nn and b < nn][:n_prod]
        Xtr2, Xva2 = Xtr.copy(), Xva.copy()
        extra = []
        for a, b in prs:
            ca, cb_ = nums[a], nums[b]
            ma, sa, mb, sb = Xtr[ca].mean(), Xtr[ca].std() + 1e-12, Xtr[cb_].mean(), Xtr[cb_].std() + 1e-12
            nm = f"{ca}*{cb_}"
            Xtr2[nm] = ((Xtr[ca] - ma) / sa) * ((Xtr[cb_] - mb) / sb)
            Xva2[nm] = ((Xva[ca] - ma) / sa) * ((Xva[cb_] - mb) / sb)
            extra.append(nm)
        old = nums; nums = old + extra
        try:
            return base(Xtr2, ytr, Xva2)
        finally:
            nums = old
    return f


def gam_boost_f(P, pairs=2, C=0.05):
    """Hybrid: GAM with measured pair products, then a booster fitted on its residual (GAM logit = init score)."""
    def f(Xtr, ytr, Xva):
        encl = S.TabularEncoder("linear").fit(Xtr, nums, cats)
        Ltr, Lva = encl.transform(Xtr), encl.transform(Xva)
        nn = len(nums); st = struct(Xtr, ytr)
        prs = [tuple(q) for q in st["pairs"] if q[0] < nn and q[1] < nn][:pairs]
        g = S.SplineLogit(nn, C=C, n_knots=7, basis="hinge", pairs=prs).fit(Ltr, ytr)
        gtr = logit(g.predict_proba(Ltr)[:, 1]); gva = logit(g.predict_proba(Lva)[:, 1])
        enc = S.TabularEncoder("native").fit(Xtr, nums, cats)
        Ztr, Zva = enc.transform(Xtr), enc.transform(Xva)
        kw = dict(P); k, mono = kw.pop("inter_k", 0), kw.pop("mono", 0)
        if k or mono:
            inter, mon = S._constraints_for(dict(inter_k=k, mono=mono), st, Ztr.shape[1])
            if inter: kw["interaction_constraints"] = inter
        a, b = split_es(ytr)
        base = dict(verbose=-1, n_jobs=NJ, random_state=0, cat_smooth=20, min_data_per_group=50, **kw)
        m = lgb.LGBMClassifier(n_estimators=8000, **base).fit(Ztr[a], ytr[a], init_score=gtr[a], eval_set=[(Ztr[b], ytr[b])],
                                                         eval_init_score=[gtr[b]], categorical_feature=enc.cat_idx_ or "auto",
                                                         callbacks=[lgb.early_stopping(200, verbose=False)])
        n = max(int(m.best_iteration_ * 1.1), 10)
        m = lgb.LGBMClassifier(n_estimators=n, **base).fit(Ztr, ytr, init_score=gtr, categorical_feature=enc.cat_idx_ or "auto")
        return sigmoid(gva + m.predict_proba(Zva, raw_score=True))
    return f


def gam_rich_f(C=0.05, knots=7, steps=0, num_pairs=0, cat_pairs=0, and_pairs=0, and_q=(0.25, 0.5, 0.75)):
    """Diagnostic GAM with switchable bases: hinge (framework), step indicators at `steps` quantiles, products of
    the top numeric pairs, category x numeric slopes for the top cat-num pairs, AND-rule terms
    (1[a > qa] * 1[b > qb]) for the top numeric pairs.  All extra columns standardised."""
    def f(Xtr, ytr, Xva):
        enc = S.TabularEncoder("linear").fit(Xtr, nums, cats)
        Ztr, Zva = enc.transform(Xtr), enc.transform(Xva)
        nn = len(nums); st = struct(Xtr, ytr)
        g = S.SplineLogit(nn, C=C, n_knots=knots, basis="hinge")
        base_tr, base_va = g._design(Ztr, fitting=True), g._design(Zva)
        npairs = [tuple(q) for q in st["pairs"] if q[0] < nn and q[1] < nn]
        cpairs = [tuple(q) for q in st["pairs"] if (q[0] < nn) != (q[1] < nn)]
        cols = [j for j in range(nn) if len(np.unique(Ztr[:, j])) > 10]
        sq = np.linspace(0, 1, steps + 2)[1:-1] if steps else []
        SK = {j: np.quantile(Ztr[:, j], sq) for j in cols} if steps else {}
        AQ = {}
        for a, b in npairs[:and_pairs]:
            AQ[(a, b)] = (np.quantile(Ztr[:, a], and_q), np.quantile(Ztr[:, b], and_q))
        Xtr_raw, Xva_raw = Xtr.reset_index(drop=True), Xva.reset_index(drop=True)
        levels = {}
        for a, b in cpairs[:cat_pairs]:
            ci, ni = (a, b) if a >= nn else (b, a)
            cname = cats[ci - nn]
            levels[(ci, ni)] = (cname, list(Xtr_raw[cname].value_counts().index[:8]))
        def extra(Z, Xr):
            B = []
            for j in SK:
                B.append((Z[:, [j]] > SK[j][None, :]).astype(float))
            for a, b in npairs[:num_pairs]:
                B.append(np.clip(Z[:, a] * Z[:, b], -10, 10)[:, None])
            for (a, b), (qa, qb) in AQ.items():
                B.append(((Z[:, [a]] > qa[None, :])[:, :, None] & (Z[:, [b]] > qb[None, :])[:, None, :]).reshape(len(Z), -1).astype(float))
            for (ci, ni), (cname, lv) in levels.items():
                col = Xr[cname].astype(object).to_numpy()
                B.append(np.column_stack([(col == l).astype(float) * Z[:, ni] for l in lv]))
            return np.hstack(B) if B else np.zeros((len(Z), 0))
        Etr, Eva = extra(Ztr, Xtr_raw), extra(Zva, Xva_raw)
        mu, sd = Etr.mean(0), Etr.std(0) + 1e-9
        Dtr = np.hstack([base_tr, (Etr - mu) / sd]); Dva = np.hstack([base_va, (Eva - mu) / sd])
        with S._blas_limit():
            lr = LogisticRegression(C=C, max_iter=5000).fit(Dtr, ytr)
        return lr.predict_proba(Dva)[:, 1]
    return f


def gam_l1_f(C=0.1, nk=15, steps=True, hinges=True, num_pairs=4, cat_pairs=2, l1_ratio=None, pair_hinge=0, cat_hinge=0, and_terms=0, hh_terms=0, tails=(), sq_walls=False, relax=None):
    """Sparse GAM: L1 (or elastic-net) logistic regression on a large basis -- per numeric column the linear term,
    hinges and step indicators at `nk` quantiles (L1 on jumps / kinks = total-variation / trend-filtering
    penalty), products of the top measured numeric pairs, category x numeric slopes; all columns standardised."""
    def f(Xtr, ytr, Xva):
        enc = S.TabularEncoder("linear").fit(Xtr, nums, cats)
        Ztr, Zva = enc.transform(Xtr), enc.transform(Xva)
        nn = len(nums); st = struct(Xtr, ytr)
        cols = [j for j in range(nn) if len(np.unique(Ztr[:, j])) > 10]
        qs = np.r_[np.linspace(0, 1, nk + 2)[1:-1], np.asarray(tails, float)]
        K = {j: np.unique(np.quantile(Ztr[:, j], qs)) for j in cols}
        npairs = [tuple(q) for q in st["pairs"] if q[0] < nn and q[1] < nn][:num_pairs]
        cps = []
        for q in st["pairs"]:
            if len(cps) >= cat_pairs: break
            if (q[0] < nn) == (q[1] < nn): continue
            ni, ci = (q[0], q[1]) if q[0] < nn else (q[1], q[0])
            oh = [k for k, nm in enumerate(enc.feature_names_) if nm.startswith(f"{cats[ci - nn]}=")]
            if oh: cps.append((ni, oh))
        PK = {j: np.quantile(Ztr[:, j], [0.2, 0.4, 0.6, 0.8]) for j in range(nn)}
        def design(Z):
            B = [Z]
            for j in cols:
                if sq_walls:                                     # smooth convex walls beyond the outer knots
                    B.append(np.minimum(np.maximum(Z[:, [j]] - K[j][-1], 0) ** 2, 25)); B.append(np.minimum(np.maximum(K[j][0] - Z[:, [j]], 0) ** 2, 25))
                if hinges: B.append(np.maximum(Z[:, [j]] - K[j][None, :], 0))
                if steps: B.append((Z[:, [j]] > K[j][None, :]).astype(float))
            for a, b in npairs: B.append(np.clip(Z[:, a] * Z[:, b], -10, 10)[:, None])
            for n_, oh in cps: B.append(Z[:, oh] * Z[:, [n_]])
            for a, b in npairs[:pair_hinge]:                     # kinked interactions: hinge(a) * b and a * hinge(b)
                for u in PK[a]: B.append(np.clip(np.maximum(Z[:, a] - u, 0) * Z[:, b], -10, 10)[:, None])
                for v in PK[b]: B.append(np.clip(Z[:, a] * np.maximum(Z[:, b] - v, 0), -10, 10)[:, None])
            for n_, oh in cps[:cat_hinge]:                       # category-specific kinks
                for u in PK[n_]: B.append(Z[:, oh] * np.maximum(Z[:, [n_]] - u, 0))
            for a, b in npairs[:and_terms]:                      # AND rules: 1[a > u] * 1[b > v]
                B.append(((Z[:, [a]] > PK[a][None, :])[:, :, None] & (Z[:, [b]] > PK[b][None, :])[:, None, :]).reshape(len(Z), -1).astype(float))
            for a, b in npairs[:hh_terms]:                       # hinge x hinge
                B.append((np.maximum(Z[:, [a]] - PK[a][None, :], 0)[:, :, None] * np.maximum(Z[:, [b]] - PK[b][None, :], 0)[:, None, :]).reshape(len(Z), -1))
            return np.hstack(B)
        Dtr, Dva = design(Ztr), design(Zva)
        mu, sd = Dtr.mean(0), Dtr.std(0) + 1e-9
        with S._blas_limit():
            if l1_ratio is None:
                lr = LogisticRegression(penalty="l1", C=C, solver="liblinear", max_iter=2000).fit((Dtr - mu) / sd, ytr)
            else:
                lr = LogisticRegression(penalty="elasticnet", l1_ratio=l1_ratio, C=C, solver="saga", max_iter=3000).fit((Dtr - mu) / sd, ytr)
            if relax is not None:                               # relaxed lasso: L2 refit on the selected columns
                keep = np.flatnonzero(np.abs(lr.coef_.ravel()) > 1e-8)
                lr2 = LogisticRegression(C=relax, max_iter=3000).fit(((Dtr - mu) / sd)[:, keep], ytr)
                return lr2.predict_proba(((Dva - mu) / sd)[:, keep])[:, 1]
        return lr.predict_proba((Dva - mu) / sd)[:, 1]
    return f


def l1_boost_f(P, gam_kw):
    """Hybrid: the v10 sparse GAM (kinked interactions), then a booster on its residual (GAM logit = init score)."""
    def f(Xtr, ytr, Xva):
        a_, b_ = split_es(ytr)
        spec = S.ModelSpec("gam", gam_kw)
        num, cat = nums, cats
        g = S._fit_model(spec, Xtr, ytr, num, cat, None, NJ)
        gtr, gva = logit(g.predict(Xtr)), logit(g.predict(Xva))
        enc = S.TabularEncoder("native").fit(Xtr, num, cat)
        Ztr, Zva = enc.transform(Xtr), enc.transform(Xva)
        base = dict(verbose=-1, n_jobs=NJ, random_state=0, cat_smooth=20, min_data_per_group=50, **P)
        m = lgb.LGBMClassifier(n_estimators=8000, **base).fit(Ztr[a_], ytr[a_], init_score=gtr[a_], eval_set=[(Ztr[b_], ytr[b_])],
                                                         eval_init_score=[gtr[b_]], categorical_feature=enc.cat_idx_ or "auto",
                                                         callbacks=[lgb.early_stopping(200, verbose=False)])
        n = max(int(m.best_iteration_ * 1.1), 5)
        m = lgb.LGBMClassifier(n_estimators=n, **base).fit(Ztr, ytr, init_score=gtr, categorical_feature=enc.cat_idx_ or "auto")
        return sigmoid(gva + m.predict_proba(Zva, raw_score=True))
    return f


def ebm_f(k_pairs):
    MAIN = dict(learning_rate=0.02, num_leaves=3, min_child_samples=20, subsample=0.85, subsample_freq=1, reg_lambda=0.0)
    PAIR = dict(learning_rate=0.02, num_leaves=4, max_depth=2, min_child_samples=30, subsample=0.85, subsample_freq=1, reg_lambda=1.0)
    def f(Xtr, ytr, Xva):
        enc = S.TabularEncoder("native").fit(Xtr, nums, cats)
        Ztr, Zva = enc.transform(Xtr), enc.transform(Xva)
        p = Ztr.shape[1]; a, b = split_es(ytr)
        c1 = dict(verbose=-1, n_jobs=NJ, random_state=0, cat_smooth=20, min_data_per_group=50, interaction_constraints=[[i] for i in range(p)], **MAIN)
        m1 = lgb.LGBMClassifier(n_estimators=8000, **c1).fit(Ztr[a], ytr[a], eval_set=[(Ztr[b], ytr[b])], categorical_feature=enc.cat_idx_ or "auto",
                                                         callbacks=[lgb.early_stopping(300, verbose=False)])
        n1 = int(m1.best_iteration_ * 1.1)
        m1 = lgb.LGBMClassifier(n_estimators=n1, **c1).fit(Ztr, ytr, categorical_feature=enc.cat_idx_ or "auto")
        itr, iva = m1.predict_proba(Ztr, raw_score=True), m1.predict_proba(Zva, raw_score=True)
        st = struct(Xtr, ytr)
        if k_pairs and st["pairs"]:
            c2 = dict(verbose=-1, n_jobs=NJ, random_state=0, cat_smooth=20, min_data_per_group=50,
                      interaction_constraints=[list(q) for q in st["pairs"][:k_pairs]], **PAIR)
            m2 = lgb.LGBMClassifier(n_estimators=4000, **c2).fit(Ztr[a], ytr[a], init_score=itr[a], eval_set=[(Ztr[b], ytr[b])],
                                                             eval_init_score=[itr[b]], callbacks=[lgb.early_stopping(200, verbose=False)],
                                                             categorical_feature=enc.cat_idx_ or "auto")
            n2 = max(int(m2.best_iteration_ * 1.1), 10)
            m2 = lgb.LGBMClassifier(n_estimators=n2, **c2).fit(Ztr, ytr, init_score=itr, categorical_feature=enc.cat_idx_ or "auto")
            iva = iva + m2.predict_proba(Zva, raw_score=True)
        return sigmoid(iva)
    return f


def xgb_f(P):
    def f(Xtr, ytr, Xva):
        enc = S.TabularEncoder("native").fit(Xtr, nums, cats)
        Ztr, Zva = enc.transform(Xtr), enc.transform(Xva)
        kw = dict(P); k, mono = kw.pop("inter_k", 0), kw.pop("mono", 0)
        if k or mono:
            inter, mon = S._constraints_for(dict(inter_k=k, mono=mono), struct(Xtr, ytr), Ztr.shape[1])
            if inter: kw["interaction_constraints"] = str([[int(i) for i in g] for g in inter])
            if mon: kw["monotone_constraints"] = "(" + ",".join(str(int(v)) for v in mon) + ")"
        a, b = split_es(ytr)
        m = xgb.XGBClassifier(n_estimators=8000, tree_method="hist", n_jobs=NJ, random_state=0, verbosity=0, eval_metric="logloss",
                              early_stopping_rounds=200, **kw).fit(Ztr[a], ytr[a], eval_set=[(Ztr[b], ytr[b])], verbose=False)
        n = int((m.best_iteration + 1) * 1.1)
        m = xgb.XGBClassifier(n_estimators=n, tree_method="hist", n_jobs=NJ, random_state=0, verbosity=0, **kw).fit(Ztr, ytr)
        return m.predict_proba(Zva)[:, 1]
    return f


def cat_f(P, mono=False):
    def f(Xtr, ytr, Xva):
        enc = S.TabularEncoder("frame", max_levels=1000).fit(Xtr, nums, cats)
        Ztr, Zva = enc.transform(Xtr), enc.transform(Xva)
        kw = dict(P)
        if mono:
            st = struct(Xtr, ytr)
            kw["monotone_constraints"] = [int(v) if j < len(nums) else 0 for j, v in enumerate(st["mono"])]
        a, b = split_es(ytr)
        base = dict(thread_count=NJ, verbose=0, allow_writing_files=False, random_seed=0, cat_features=enc.cat_idx_ or None, **kw)
        m = cb.CatBoostClassifier(iterations=8000, od_type="Iter", od_wait=200, **base)
        m.fit(Ztr.iloc[a], ytr[a], eval_set=(Ztr.iloc[b], ytr[b]), use_best_model=True)
        n = int((m.get_best_iteration() + 1) * 1.1)
        m = cb.CatBoostClassifier(iterations=max(n, 20), **base).fit(Ztr, ytr)
        return m.predict_proba(Zva)[:, 1]
    return f


def gam_sl_f(C=0.05, knots=7, pairs=4):
    """framework GAM design (standardised numerics + raw hinge terms) + products of the top measured pairs."""
    def f(Xtr, ytr, Xva):
        enc = S.TabularEncoder("linear").fit(Xtr, nums, cats)
        Ztr, Zva = enc.transform(Xtr), enc.transform(Xva)
        nn = len(nums)
        st = struct(Xtr, ytr)
        prs = [(a, b) for a, b in st["pairs"] if a < nn and b < nn][:pairs]
        g = S.SplineLogit(nn, C=C, n_knots=knots, basis="hinge")
        P_ = lambda Z: np.column_stack([np.clip(Z[:, a] * Z[:, b], -10, 10) for a, b in prs]) if prs else np.zeros((len(Z), 0))
        Dtr = np.hstack([g._design(Ztr, fitting=True), P_(Ztr)]); Dva = np.hstack([g._design(Zva), P_(Zva)])
        with S._blas_limit():
            lr = LogisticRegression(C=C, max_iter=5000).fit(Dtr, ytr)
        return lr.predict_proba(Dva)[:, 1]
    return f


def gam_f(C=0.05, knots=7, basis="hinge", pairs=0, std_all=False):
    def f(Xtr, ytr, Xva):
        enc = S.TabularEncoder("linear").fit(Xtr, nums, cats)
        Ztr, Zva = enc.transform(Xtr), enc.transform(Xva)
        if not pairs and not std_all:
            g = S.SplineLogit(len(nums), C=C, n_knots=knots, basis=basis).fit(Ztr, ytr)
            return g.predict_proba(Zva)[:, 1]
        nn = len(nums)
        st = struct(Xtr, ytr)
        prs = [(a, b) for a, b in st["pairs"] if a < nn and b < nn][:pairs] if pairs else []
        g = S.SplineLogit(nn, C=C, n_knots=knots, basis=basis)
        D = g._design(Ztr, fitting=True)
        P_ = lambda Z: np.column_stack([Z[:, a] * Z[:, b] for a, b in prs]) if prs else np.zeros((len(Z), 0))
        Dtr = np.hstack([D, P_(Ztr)]); Dva = np.hstack([g._design(Zva), P_(Zva)])
        mu, sd = Dtr.mean(0), Dtr.std(0) + 1e-9
        with S._blas_limit():
            lr = LogisticRegression(C=C, max_iter=5000).fit((Dtr - mu) / sd, ytr)
        return lr.predict_proba((Dva - mu) / sd)[:, 1]
    return f


def forest_f(cls, min_leaf, mf, cw=None):
    def f(Xtr, ytr, Xva):
        enc = S.TabularEncoder("imputed").fit(Xtr, nums, cats)
        m = cls(n_estimators=500, min_samples_leaf=min_leaf, max_features=mf, n_jobs=NJ, random_state=0, class_weight=cw)
        return m.fit(enc.transform(Xtr), ytr).predict_proba(enc.transform(Xva))[:, 1]
    return f


def hgb_f(depth):
    def f(Xtr, ytr, Xva):
        enc = S.TabularEncoder("native").fit(Xtr, nums, cats)
        m = HistGradientBoostingClassifier(learning_rate=0.03, max_iter=5000, max_depth=depth, max_leaf_nodes=2 ** depth, min_samples_leaf=30,
                                           l2_regularization=1.0, early_stopping=True, validation_fraction=0.15, n_iter_no_change=200,
                                           random_state=0, categorical_features=enc.cat_idx_ or None)
        return m.fit(enc.transform(Xtr), ytr).predict_proba(enc.transform(Xva))[:, 1]
    return f


L2 = dict(learning_rate=0.03, max_depth=2, num_leaves=4, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0)
L3 = dict(L2, max_depth=3, num_leaves=8)
L4 = dict(L2, max_depth=4, num_leaves=15, min_child_samples=40, colsample_bytree=0.7)
X2 = dict(learning_rate=0.03, max_depth=2, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0)
X3 = dict(X2, max_depth=3, colsample_bytree=0.7)
C3 = dict(learning_rate=0.03, depth=3, l2_leaf_reg=5)
V = {
    # framework-library-like members
    "lgb_B": lgb_f(dict(L2, inter_k=8, mono=1)), "lgb_d3k16m": lgb_f(dict(L3, inter_k=16, mono=1)),
    "lgb_d2": lgb_f(L2), "lgb_d4": lgb_f(L4), "lgb_d1": lgb_f(dict(L2, max_depth=1, num_leaves=2, learning_rate=0.05, subsample=1.0, colsample_bytree=1.0, reg_lambda=0.0)),
    "xgb_d2k8m": xgb_f(dict(X2, inter_k=8, mono=1)), "xgb_d3": xgb_f(X3), "xgb_d3k16m": xgb_f(dict(X3, inter_k=16, mono=1)),
    "cat_d3m": cat_f(C3, mono=True), "cat_d4": cat_f(dict(C3, depth=4, learning_rate=0.05)), "cat_d6": cat_f(dict(C3, depth=6, learning_rate=0.05, l2_leaf_reg=3)),
    "gam_c05": gam_f(0.05, 7), "gam_c2": gam_f(0.2, 7), "et_l5": forest_f(ExtraTreesClassifier, 5, 0.4), "et_l15": forest_f(ExtraTreesClassifier, 15, 0.5),
    # new candidates
    "ebm_k0": ebm_f(0), "ebm_k8": ebm_f(8), "ebm_k16": ebm_f(16),
    "gam_p4": gam_f(0.05, 7, pairs=4), "gam_p8": gam_f(0.05, 12, pairs=8), "gam_spline": gam_f(0.1, 9, basis="spline"),
    "lgb_linear": lgb_f(dict(L2, inter_k=8, mono=1), linear=True), "lgb_extra_d3": lgb_f(L3, extra=True),
    "lgb_single": lgb_f(dict(L2, max_depth=2, num_leaves=3), single=True),
    "lgb_B_ss05": lgb_f(dict(L2, inter_k=8, mono=1, subsample=0.5)), "lgb_B_lr01": lgb_f(dict(L2, inter_k=8, mono=1, learning_rate=0.01)),
    "lgb_B_cw3": lgb_f(dict(L2, inter_k=8, mono=1), cw=3.0), "lgb_B_cw10": lgb_f(dict(L2, inter_k=8, mono=1), cw=10.0),
    "lgb_d3_cw5": lgb_f(L3, cw=5.0),
    "cat_d4_lgv": cat_f(dict(C3, depth=4, learning_rate=0.05, langevin=True, diffusion_temperature=10000)),
    "cat_d3m_cw": cat_f(dict(C3, auto_class_weights="SqrtBalanced"), mono=True),
    "gam_std": gam_f(0.05, 7, std_all=True), "gam_std_c2": gam_f(0.2, 7, std_all=True), "gam_p4_c2": gam_f(0.2, 7, pairs=4),
    "gam_p2": gam_f(0.05, 7, pairs=2), "gam_p6": gam_f(0.05, 7, pairs=6), "gam_p4_k12": gam_f(0.05, 12, pairs=4),
    "gam_sl_p2": gam_sl_f(0.05, 7, 2), "gam_sl_p4": gam_sl_f(0.05, 7, 4), "gam_sl_p6": gam_sl_f(0.05, 7, 6), "gam_sl_p4_c2": gam_sl_f(0.2, 7, 4),
    "lgb_B_prod2": lgb_prod_f(dict(L2, inter_k=8, mono=1), 2), "lgb_B_prod4": lgb_prod_f(dict(L2, inter_k=8, mono=1), 4),
    "lgb_d2_prod2": lgb_prod_f(L2, 2),
    "cat_d3m_lr015": cat_f(dict(C3, learning_rate=0.015), mono=True), "xgb_d2k8m_lr01": xgb_f(dict(X2, inter_k=8, mono=1, learning_rate=0.01)),
    "lgb_linear_lr01": lgb_f(dict(L2, inter_k=8, mono=1, learning_rate=0.01), linear=True), "lgb_B_lr005": lgb_f(dict(L2, inter_k=8, mono=1, learning_rate=0.005)),
    "gamboost_p2": gam_boost_f(dict(L2, learning_rate=0.01, inter_k=8)), "gamboost_p2_d3": gam_boost_f(dict(L3, learning_rate=0.01)),
    "grich_base": gam_rich_f(), "grich_step10": gam_rich_f(steps=10), "grich_step20": gam_rich_f(steps=20),
    "grich_p2": gam_rich_f(num_pairs=2), "grich_cat2": gam_rich_f(cat_pairs=2), "grich_and4": gam_rich_f(and_pairs=4),
    "grich_all": gam_rich_f(steps=10, num_pairs=2, cat_pairs=2, and_pairs=4),
    "grich_all_c02": gam_rich_f(C=0.02, steps=10, num_pairs=2, cat_pairs=2, and_pairs=4),
    "grich_c1": gam_rich_f(cat_pairs=1), "grich_c4": gam_rich_f(cat_pairs=4), "grich_c2p2": gam_rich_f(cat_pairs=2, num_pairs=2),
    "grich_c2p4": gam_rich_f(cat_pairs=2, num_pairs=4), "grich_c4p4": gam_rich_f(cat_pairs=4, num_pairs=4), "grich_c2p6": gam_rich_f(cat_pairs=2, num_pairs=6),
    "gl1_c003": gam_l1_f(C=0.03), "gl1_c01": gam_l1_f(C=0.1), "gl1_c03": gam_l1_f(C=0.3),
    "gl1_steps_c01": gam_l1_f(C=0.1, hinges=False), "gl1_hinge_c01": gam_l1_f(C=0.1, steps=False),
    "gl1_nk30_c01": gam_l1_f(C=0.1, nk=30), "gl1_nopair_c01": gam_l1_f(C=0.1, num_pairs=0, cat_pairs=0),
    "gl1h_nk15_c005": gam_l1_f(C=0.05, steps=False), "gl1h_nk15_c02": gam_l1_f(C=0.2, steps=False),
    "gl1h_nk25_c01": gam_l1_f(C=0.1, nk=25, steps=False), "gl1h_nk40_c01": gam_l1_f(C=0.1, nk=40, steps=False),
    "gl1h_nk25_c005": gam_l1_f(C=0.05, nk=25, steps=False), "gl1h_nk40_c005": gam_l1_f(C=0.05, nk=40, steps=False),
    "gl1h_nk25_c01_p8": gam_l1_f(C=0.1, nk=25, steps=False, num_pairs=8, cat_pairs=4),
    "gl1h_ph4": gam_l1_f(C=0.1, steps=False, pair_hinge=4), "gl1h_ph8": gam_l1_f(C=0.1, steps=False, num_pairs=8, pair_hinge=8),
    "gl1h_ph4_ch2": gam_l1_f(C=0.1, steps=False, pair_hinge=4, cat_hinge=2),
    "gl1h_ph6_ch2": gam_l1_f(C=0.1, steps=False, num_pairs=6, pair_hinge=6, cat_hinge=2),
    "gl1h_ph4_ch4": gam_l1_f(C=0.1, steps=False, cat_pairs=4, pair_hinge=4, cat_hinge=4),
    "gl1h_ph8_ch2": gam_l1_f(C=0.1, steps=False, num_pairs=8, pair_hinge=8, cat_hinge=2),
    "gl1h_ph4_ch2_c02": gam_l1_f(C=0.2, steps=False, pair_hinge=4, cat_hinge=2),
    "gl1h_ph4_ch2_c005": gam_l1_f(C=0.05, steps=False, pair_hinge=4, cat_hinge=2),
    "l1boost_d2": l1_boost_f(dict(L2, learning_rate=0.01), dict(C=0.1, knots=15, basis="hinge", pairs=4, cpairs=2, penalty="l1", phinge=4, chinge=2)), "l1boost_d3": l1_boost_f(dict(L3, learning_rate=0.01), dict(C=0.1, knots=15, basis="hinge", pairs=4, cpairs=2, penalty="l1", phinge=4, chinge=2)),
    "gl1h_and4": gam_l1_f(C=0.1, steps=False, pair_hinge=4, cat_hinge=2, and_terms=4),
    "gl1h_hh4": gam_l1_f(C=0.1, steps=False, pair_hinge=4, cat_hinge=2, hh_terms=4),
    "gl1h_and4_hh4": gam_l1_f(C=0.1, steps=False, pair_hinge=4, cat_hinge=2, and_terms=4, hh_terms=4),
    "gl1h_ph4_t4": gam_l1_f(C=0.1, steps=False, pair_hinge=4, cat_hinge=2, tails=(0.005, 0.01, 0.02, 0.04, 0.96, 0.98, 0.99, 0.995)),
    "gl1h_ph4_t2": gam_l1_f(C=0.1, steps=False, pair_hinge=4, cat_hinge=2, tails=(0.01, 0.03, 0.97, 0.99)),
    "gl1h_ph6_t4": gam_l1_f(C=0.1, steps=False, num_pairs=6, pair_hinge=6, cat_hinge=2, tails=(0.005, 0.01, 0.02, 0.04, 0.96, 0.98, 0.99, 0.995)),
    "gl1h_ph4_sq": gam_l1_f(C=0.1, steps=False, pair_hinge=4, cat_hinge=2, sq_walls=True),
    "gl1h_ph6_sq": gam_l1_f(C=0.1, steps=False, num_pairs=6, pair_hinge=6, cat_hinge=2, sq_walls=True),
    "gl1h_ph4_rel1": gam_l1_f(C=0.1, steps=False, pair_hinge=4, cat_hinge=2, relax=1.0),
    "gl1h_ph4_rel01": gam_l1_f(C=0.1, steps=False, pair_hinge=4, cat_hinge=2, relax=0.1),
    "gl1h_ph6_c005": gam_l1_f(C=0.05, steps=False, num_pairs=6, pair_hinge=6, cat_hinge=2),
    "gl1h_ph6_c005_rel01": gam_l1_f(C=0.05, steps=False, num_pairs=6, pair_hinge=6, cat_hinge=2, relax=0.1),
    "rf_l5": forest_f(RandomForestClassifier, 5, 0.3), "hgb_d3": hgb_f(3), "xgb_d4": xgb_f(dict(X3, max_depth=4)),
}
if name == "list":
    print(" ".join(V)); sys.exit()
t0 = time.time()
out = np.zeros((2, len(y)))
for s in (0, 1):
    for tr, va in folds(y, s):
        out[s, va] = V[name](Xc.iloc[tr].reset_index(drop=True), y[tr], Xc.iloc[va].reset_index(drop=True))
np.save(os.path.join(HERE, "logs", f"zoo_{key}_{name}.npy"), out)
sc = [auc(y, o) for o in out] if key == "p1" else [ll(y, o) for o in out]
line = f"{key} {name:14s} {'AUC' if key == 'p1' else 'LL'}={np.mean(sc):.5f} per-seed={[round(v, 5) for v in sc]} LL={np.mean([ll(y, o) for o in out]):.5f} {time.time()-t0:.0f}s"
print(line, flush=True)
with open(os.path.join(HERE, "logs", "zoo.log"), "a") as f:
    f.write(line + "\n")
