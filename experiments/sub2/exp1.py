import sys; sys.path.insert(0, 'experiments/sub2')
from common import *
import lightgbm as lgb, xgboost as xgb, catboost as cb
from sklearn.preprocessing import SplineTransformer, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
key = sys.argv[1]; NJ = int(sys.argv[2]) if len(sys.argv) > 2 else 1
X, y, nums, cats = load(key); Z = native(X, nums, cats); F = folds(y)
print(key, Z.shape, flush=True)

def cv_lgb(name, **kw):
    t0 = time.time(); oof = np.zeros(len(y)); its = []
    for tr, va in F:
        m = lgb.LGBMClassifier(n_estimators=3000, verbose=-1, n_jobs=NJ, random_state=0, **kw)
        m.fit(Z.iloc[tr], y[tr], eval_set=[(Z.iloc[va], y[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict_proba(Z.iloc[va])[:, 1]; its.append(m.best_iteration_)
    report(name + f" it={int(np.mean(its))}", y, oof, t0); return oof
def cv_xgb(name, **kw):
    t0 = time.time(); oof = np.zeros(len(y)); its = []
    for tr, va in F:
        m = xgb.XGBClassifier(n_estimators=3000, n_jobs=NJ, random_state=0, tree_method='hist', early_stopping_rounds=100, eval_metric='logloss', **kw)
        m.fit(Z.iloc[tr], y[tr], eval_set=[(Z.iloc[va], y[va])], verbose=False)
        oof[va] = m.predict_proba(Z.iloc[va])[:, 1]; its.append(m.best_iteration)
    report(name + f" it={int(np.mean(its))}", y, oof, t0); return oof
def cv_cat(name, **kw):
    t0 = time.time(); oof = np.zeros(len(y)); its = []
    Xc = X.copy()
    for c in cats: Xc[c] = Xc[c].fillna('__na__')
    for tr, va in F:
        m = cb.CatBoostClassifier(iterations=3000, verbose=0, thread_count=NJ, random_seed=0, cat_features=cats, allow_writing_files=False, od_type='Iter', od_wait=100, **kw)
        m.fit(Xc.iloc[tr], y[tr], eval_set=(Xc.iloc[va], y[va]))
        oof[va] = m.predict_proba(Xc.iloc[va])[:, 1]; its.append(m.get_best_iteration())
    report(name + f" it={int(np.mean(its))}", y, oof, t0); return oof
def cv_sk(name, make):
    t0 = time.time(); oof = np.zeros(len(y))
    for tr, va in F:
        m = make().fit(X.iloc[tr], y[tr]); oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    report(name, y, oof, t0); return oof

def spline_lr(C=1.0, knots=8, degree=3):
    num_pipe = make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(), SplineTransformer(n_knots=knots, degree=degree, knots='quantile'))
    cat_pipe = make_pipeline(SimpleImputer(strategy='constant', fill_value='__na__'), OneHotEncoder(handle_unknown='ignore'))
    ct = ColumnTransformer([('n', num_pipe, nums), ('c', cat_pipe, cats)])
    return make_pipeline(ct, LogisticRegression(C=C, max_iter=3000))

res = {}
res['lgb_stump'] = cv_lgb('lgb stumps lr.05', learning_rate=0.05, num_leaves=2, min_child_samples=20)
res['lgb_d2'] = cv_lgb('lgb depth2 lr.03', learning_rate=0.03, max_depth=2, num_leaves=4, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
res['lgb_d3'] = cv_lgb('lgb depth3 lr.03', learning_rate=0.03, max_depth=3, num_leaves=8, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
res['lgb_15'] = cv_lgb('lgb leaves15 lr.02', learning_rate=0.02, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1)
res['lgb_31'] = cv_lgb('lgb leaves31 lr.02', learning_rate=0.02, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=3)
res['xgb_d3'] = cv_xgb('xgb depth3 lr.03', learning_rate=0.03, max_depth=3, min_child_weight=5, subsample=0.8, colsample_bytree=0.7, reg_lambda=1)
res['xgb_d5'] = cv_xgb('xgb depth5 lr.02', learning_rate=0.02, max_depth=5, min_child_weight=5, subsample=0.8, colsample_bytree=0.7, reg_lambda=1)
res['cat_d6'] = cv_cat('cat depth6 lr.05', learning_rate=0.05, depth=6)
res['cat_d4'] = cv_cat('cat depth4 lr.05', learning_rate=0.05, depth=4, l2_leaf_reg=5)
for C in [0.03, 0.1, 0.3, 1.0]:
    res[f'spl_{C}'] = cv_sk(f'spline-LR C={C} knots8', lambda: spline_lr(C=C))
res['spl_k5'] = cv_sk('spline-LR C=0.1 knots5', lambda: spline_lr(C=0.1, knots=5))
res['spl_k12'] = cv_sk('spline-LR C=0.1 knots12', lambda: spline_lr(C=0.1, knots=12))
import pickle; pickle.dump(res, open(f'experiments/sub2/exp1_{key}.pkl', 'wb'))
# blends
from scipy.special import logit, expit
def lb(names):
    return expit(np.mean([logit(np.clip(res[n],1e-6,1-1e-6)) for n in names], 0))
report('blend lgb15+xgb3+cat6', y, lb(['lgb_15','xgb_d3','cat_d6']))
report('blend lgb15+xgb3+cat6+spl0.1', y, lb(['lgb_15','xgb_d3','cat_d6','spl_0.1']))
report('blend stump+spl0.1', y, lb(['lgb_stump','spl_0.1']))
