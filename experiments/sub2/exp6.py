import sys; sys.path.insert(0, 'experiments/sub2')
from common import *
import lightgbm as lgb, xgboost as xgb, catboost as cb, pickle
from scipy.stats import spearmanr
from scipy.special import logit, expit
key = sys.argv[1]; NJ = 1
X, y, nums, cats = load(key); Z = native(X, nums, cats); F = folds(y); cols = list(Z.columns)
cat_idx = [cols.index(c) for c in cats]
res = pickle.load(open(f'experiments/sub2/exp2_{key}.pkl','rb'))
def detect(Zt, yt):
    """pairs by depth-2 gain; monotone signs by stumps partial dependence"""
    m2 = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=2, num_leaves=4, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=NJ, random_state=0).fit(Zt, yt, categorical_feature=cat_idx)
    pairs = {}
    for t in m2.booster_.dump_model()['tree_info']:
        def walk(node, parents):
            if 'split_feature' not in node: return
            f = node['split_feature']; g = node.get('split_gain', 0)
            for pf in parents:
                if pf != f: pairs[tuple(sorted((pf, f)))] = pairs.get(tuple(sorted((pf, f))), 0) + g
            for ch in ('left_child', 'right_child'):
                if ch in node: walk(node[ch], parents + [f])
        walk(t['tree_structure'], [])
    top = [p for p, g in sorted(pairs.items(), key=lambda kv: -kv[1])]
    m1 = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.1, num_leaves=2, min_child_samples=20, verbose=-1, n_jobs=NJ, random_state=0).fit(Zt, yt, categorical_feature=cat_idx)
    mono = [0]*len(cols); med = Zt.median()
    for j, c in enumerate(cols):
        if c in cats: continue
        x = Zt[c].dropna(); 
        if x.nunique() < 3: continue
        qs = np.quantile(x, np.linspace(0.02, 0.98, 25)); Zq = pd.DataFrame(np.tile(med.values, (25,1)), columns=cols); Zq[c] = qs
        pd_ = m1.predict(Zq, raw_score=True)
        if pd_.max()-pd_.min() < 1e-9: continue
        rho = spearmanr(qs, pd_).correlation; d = np.diff(pd_); sc = int(np.sum(np.sign(d[1:])*np.sign(d[:-1]) < 0))
        if abs(rho) > 0.9 and sc <= 2: mono[j] = int(np.sign(rho))
    return top, mono
top, mono = detect(Z, y)
print(key, 'n monotone', sum(1 for s in mono if s), 'top pairs', [(cols[a], cols[b]) for a, b in top[:6]], flush=True)
def cons(k): return [[i] for i in range(len(cols))] + [list(p) for p in top[:k]]
def xcons(k): return [[cols[i] for i in g] for g in cons(k)]
def cv_lgb(name, seeds=(0,), **kw):
    t0 = time.time(); oof = np.zeros(len(y)); its = []
    for s in seeds:
        for tr, va in F:
            m = lgb.LGBMClassifier(n_estimators=5000, verbose=-1, n_jobs=NJ, random_state=s, **kw)
            m.fit(Z.iloc[tr], y[tr], eval_set=[(Z.iloc[va], y[va])], callbacks=[lgb.early_stopping(150, verbose=False)], categorical_feature=cat_idx)
            oof[va] += m.predict_proba(Z.iloc[va])[:, 1]/len(seeds); its.append(m.best_iteration_)
    report(f"{name} it={int(np.mean(its))}", y, oof, t0); return oof
def cv_xgb(name, **kw):
    t0 = time.time(); oof = np.zeros(len(y)); its = []
    for tr, va in F:
        m = xgb.XGBClassifier(n_estimators=5000, n_jobs=NJ, random_state=0, tree_method='hist', early_stopping_rounds=150, eval_metric='logloss', **kw)
        m.fit(Z.iloc[tr], y[tr], eval_set=[(Z.iloc[va], y[va])], verbose=False)
        oof[va] = m.predict_proba(Z.iloc[va])[:, 1]; its.append(m.best_iteration)
    report(f"{name} it={int(np.mean(its))}", y, oof, t0); return oof
def cv_cat(name, **kw):
    t0 = time.time(); oof = np.zeros(len(y)); its = []
    Xc = X.copy()
    for c in cats: Xc[c] = Xc[c].fillna('__na__')
    for tr, va in F:
        m = cb.CatBoostClassifier(iterations=3000, verbose=0, thread_count=NJ, random_seed=0, cat_features=cats, allow_writing_files=False, od_type='Iter', od_wait=100, **kw)
        m.fit(Xc.iloc[tr], y[tr], eval_set=(Xc.iloc[va], y[va])); oof[va] = m.predict_proba(Xc.iloc[va])[:, 1]; its.append(m.get_best_iteration())
    report(f"{name} it={int(np.mean(its))}", y, oof, t0); return oof
D2 = dict(learning_rate=0.03, max_depth=2, num_leaves=4, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
D3 = dict(learning_rate=0.03, max_depth=3, num_leaves=8, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
MC = dict(monotone_constraints=mono, monotone_constraints_method='advanced')
for k in (4, 8, 12, 16, 24):
    res[f'c{k}_d2m'] = cv_lgb(f'lgb d2 inter{k} + mono', interaction_constraints=cons(k), **MC, **D2)
res['c8_d3m'] = cv_lgb('lgb d3 inter8 + mono', interaction_constraints=cons(8), **MC, **D3)
res['c16_d3m'] = cv_lgb('lgb d3 inter16 + mono', interaction_constraints=cons(16), **MC, **D3)
res['c8_d2m_cs1'] = cv_lgb('lgb d2 inter8 + mono colsample1 lr.02', interaction_constraints=cons(8), **MC, learning_rate=0.02, max_depth=2, num_leaves=4, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=1.0)
res['c8_d2m_3s'] = cv_lgb('lgb d2 inter8 + mono 3 seeds', seeds=(0,1,2), interaction_constraints=cons(8), **MC, **D2)
# xgboost with the same structure (feature names needed for interaction constraints -> use indices)
xmono = '(' + ','.join(str(s) for s in mono) + ')'
res['x_c8_d2m'] = cv_xgb('xgb d2 inter8 + mono', interaction_constraints=xcons(8), monotone_constraints=xmono, learning_rate=0.03, max_depth=2, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=1)
res['x_c8_d3m'] = cv_xgb('xgb d3 inter8 + mono', interaction_constraints=xcons(8), monotone_constraints=xmono, learning_rate=0.03, max_depth=3, min_child_weight=5, subsample=0.8, colsample_bytree=0.7, reg_lambda=1)
res['x_d3m'] = cv_xgb('xgb d3 mono only', monotone_constraints=xmono, learning_rate=0.03, max_depth=3, min_child_weight=5, subsample=0.8, colsample_bytree=0.7, reg_lambda=1)
# catboost with monotone (numerics only, by name)
cmono = {c: mono[cols.index(c)] for c in nums if mono[cols.index(c)]}
res['cat_d4m'] = cv_cat('cat d4 + mono', learning_rate=0.05, depth=4, l2_leaf_reg=5, monotone_constraints=cmono)
pickle.dump(res, open(f'experiments/sub2/exp6_{key}.pkl','wb'))
def lb(names): return expit(np.mean([logit(np.clip(res[n],1e-6,1-1e-6)) for n in names], 0))
report('OLD blend lgb_d2+xgb_d3+cat_d4', y, lb(['lgb_d2','xgb_d3','cat_d4']))
report('NEW blend c8_d2m+x_c8_d3m+cat_d4m', y, lb(['c8_d2m','x_c8_d3m','cat_d4m']))
report('NEW blend c8_d2m+c16_d3m+x_c8_d2m+x_c8_d3m+cat_d4m', y, lb(['c8_d2m','c16_d3m','x_c8_d2m','x_c8_d3m','cat_d4m']))
report('MIX blend c8_d2m+x_c8_d3m+cat_d4m+lgb_d2+xgb_d3+cat_d4', y, lb(['c8_d2m','x_c8_d3m','cat_d4m','lgb_d2','xgb_d3','cat_d4']))
