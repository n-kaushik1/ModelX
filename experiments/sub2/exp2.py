import sys; sys.path.insert(0, 'experiments/sub2')
from common import *
import lightgbm as lgb, xgboost as xgb, catboost as cb, pickle
from scipy.special import logit, expit
key = sys.argv[1]; NJ = int(sys.argv[2]) if len(sys.argv) > 2 else 1
X, y, nums, cats = load(key); Z0 = native(X, nums, cats); F = folds(y)
res = pickle.load(open(f'experiments/sub2/exp1_{key}.pkl','rb'))

def add_pairs(Z, thr=0.9):
    """generic: for numeric pairs with |corr|>thr add the mean of z-scores (denoised latent) and the difference"""
    Zn = Z[nums]; cor = Zn.corr()
    Z = Z.copy(); added = []
    cols = list(nums)
    for i, a in enumerate(cols):
        for b in cols[i+1:]:
            r = cor.loc[a, b]
            if abs(r) > thr:
                za = (Zn[a]-Zn[a].mean())/Zn[a].std(); zb = (Zn[b]-Zn[b].mean())/Zn[b].std()*np.sign(r)
                Z[f'{a}__{b}__mean'] = (za.fillna(zb) + zb.fillna(za))/2
                Z[f'{a}__{b}__diff'] = za - zb
                added.append((a, b, round(r, 3)))
    return Z, added
Z1, added = add_pairs(Z0); print(key, 'pairs added:', added, flush=True)

def cv_lgb(name, Z, seeds=(0,), **kw):
    t0 = time.time(); oof = np.zeros(len(y)); its = []
    for s in seeds:
        for tr, va in F:
            m = lgb.LGBMClassifier(n_estimators=4000, verbose=-1, n_jobs=NJ, random_state=s, **kw)
            m.fit(Z.iloc[tr], y[tr], eval_set=[(Z.iloc[va], y[va])], callbacks=[lgb.early_stopping(150, verbose=False)])
            oof[va] += m.predict_proba(Z.iloc[va])[:, 1]/len(seeds); its.append(m.best_iteration_)
    report(name + f" it={int(np.mean(its))}", y, oof, t0); return oof
def cv_xgb(name, Z, seeds=(0,), **kw):
    t0 = time.time(); oof = np.zeros(len(y)); its = []
    for s in seeds:
        for tr, va in F:
            m = xgb.XGBClassifier(n_estimators=4000, n_jobs=NJ, random_state=s, tree_method='hist', early_stopping_rounds=150, eval_metric='logloss', **kw)
            m.fit(Z.iloc[tr], y[tr], eval_set=[(Z.iloc[va], y[va])], verbose=False)
            oof[va] += m.predict_proba(Z.iloc[va])[:, 1]/len(seeds); its.append(m.best_iteration)
    report(name + f" it={int(np.mean(its))}", y, oof, t0); return oof
def cv_cat(name, Zx, **kw):
    t0 = time.time(); oof = np.zeros(len(y)); its = []
    Xc = Zx.copy()
    for c in cats: Xc[c] = X[c].fillna('__na__')
    for tr, va in F:
        m = cb.CatBoostClassifier(iterations=3000, verbose=0, thread_count=NJ, random_seed=0, cat_features=cats, allow_writing_files=False, od_type='Iter', od_wait=100, **kw)
        m.fit(Xc.iloc[tr], y[tr], eval_set=(Xc.iloc[va], y[va]))
        oof[va] = m.predict_proba(Xc.iloc[va])[:, 1]; its.append(m.get_best_iteration())
    report(name + f" it={int(np.mean(its))}", y, oof, t0); return oof

P_st = dict(learning_rate=0.05, num_leaves=2, min_child_samples=20)
P_d2 = dict(learning_rate=0.03, max_depth=2, num_leaves=4, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
P_d3 = dict(learning_rate=0.03, max_depth=3, num_leaves=8, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
res['st_pairs'] = cv_lgb('lgb stumps +pairs', Z1, **P_st)
res['d2_pairs'] = cv_lgb('lgb depth2 +pairs', Z1, **P_d2)
res['d3_pairs'] = cv_lgb('lgb depth3 +pairs', Z1, **P_d3)
res['st_lr02'] = cv_lgb('lgb stumps lr.02', Z0, learning_rate=0.02, num_leaves=2, min_child_samples=20)
res['d2_3seed'] = cv_lgb('lgb depth2 3 seeds', Z0, seeds=(0,1,2), **P_d2)
res['d2_cs5'] = cv_lgb('lgb depth2 colsample.5 lr.02', Z0, learning_rate=0.02, max_depth=2, num_leaves=4, min_child_samples=30, subsample=0.7, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5)
res['d2_cs5_pairs'] = cv_lgb('lgb depth2 colsample.5 lr.02 +pairs', Z1, learning_rate=0.02, max_depth=2, num_leaves=4, min_child_samples=30, subsample=0.7, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5)
res['xgb_d2'] = cv_xgb('xgb depth2 lr.03', Z0, learning_rate=0.03, max_depth=2, min_child_weight=5, subsample=0.8, colsample_bytree=0.7, reg_lambda=1)
res['xgb_d3_pairs'] = cv_xgb('xgb depth3 +pairs', Z1, learning_rate=0.03, max_depth=3, min_child_weight=5, subsample=0.8, colsample_bytree=0.7, reg_lambda=1)
res['cat_d4_pairs'] = cv_cat('cat depth4 +pairs', Z1[[c for c in Z1.columns if c not in cats]].join(X[cats]), learning_rate=0.05, depth=4, l2_leaf_reg=5)
res['cat_d3'] = cv_cat('cat depth3', X, learning_rate=0.05, depth=3, l2_leaf_reg=5)
pickle.dump(res, open(f'experiments/sub2/exp2_{key}.pkl', 'wb'))
def lb(names): return expit(np.mean([logit(np.clip(res[n],1e-6,1-1e-6)) for n in names], 0))
report('blend st+d2+xgb3+cat4', y, lb(['lgb_stump','lgb_d2','xgb_d3','cat_d4']))
report('blend pairs: st+d2+d3+xgb3+cat4', y, lb(['st_pairs','d2_pairs','d3_pairs','xgb_d3_pairs','cat_d4_pairs']))
report('blend pairs+plain all', y, lb(['st_pairs','d2_pairs','xgb_d3_pairs','cat_d4_pairs','lgb_stump','lgb_d2','xgb_d3','cat_d4']))
