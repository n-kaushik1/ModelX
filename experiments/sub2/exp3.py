import sys; sys.path.insert(0, 'experiments/sub2')
from common import *
import lightgbm as lgb, xgboost as xgb, pickle, optuna
from scipy.special import logit, expit
from sklearn.linear_model import LogisticRegression
optuna.logging.set_verbosity(optuna.logging.WARNING)
key = sys.argv[1]; NJ = 1
X, y, nums, cats = load(key); Z = native(X, nums, cats); F = folds(y)
res = pickle.load(open(f'experiments/sub2/exp2_{key}.pkl','rb'))
def cv_lgb(name, seeds=(0,), quiet=False, **kw):
    t0 = time.time(); oof = np.zeros(len(y)); its = []
    for s in seeds:
        for tr, va in F:
            m = lgb.LGBMClassifier(n_estimators=4000, verbose=-1, n_jobs=NJ, random_state=s, **kw)
            m.fit(Z.iloc[tr], y[tr], eval_set=[(Z.iloc[va], y[va])], callbacks=[lgb.early_stopping(150, verbose=False)])
            oof[va] += m.predict_proba(Z.iloc[va])[:, 1]/len(seeds); its.append(m.best_iteration_)
    if not quiet: report(name + f" it={int(np.mean(its))}", y, oof, t0)
    return oof, int(np.mean(its))
# A. class weights for cost (P3 relevant, but run everywhere)
for spw in [2, 4, 10]:
    cv_lgb(f'lgb d2 spw={spw}', learning_rate=0.03, max_depth=2, num_leaves=4, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, scale_pos_weight=spw)
# B. optuna TPE 40 trials on shallow-ish space, objective = OOF logloss
def objective(trial):
    p = dict(learning_rate=trial.suggest_float('lr', 0.01, 0.1, log=True),
             num_leaves=trial.suggest_int('leaves', 2, 16), max_depth=trial.suggest_int('depth', 1, 5),
             min_child_samples=trial.suggest_int('mcs', 5, 200, log=True),
             subsample=trial.suggest_float('subsample', 0.5, 1.0), subsample_freq=1,
             colsample_bytree=trial.suggest_float('colsample', 0.3, 1.0),
             reg_lambda=trial.suggest_float('l2', 1e-3, 50, log=True), reg_alpha=trial.suggest_float('l1', 1e-3, 10, log=True),
             min_split_gain=trial.suggest_float('msg', 0, 1))
    oof, it = cv_lgb('', quiet=True, **p); trial.set_user_attr('it', it); trial.set_user_attr('auc', roc_auc_score(y, oof))
    return log_loss(y, oof)
t0 = time.time()
st = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=10))
st.optimize(objective, n_trials=40)
print(f'optuna best LL={st.best_value:.5f} auc={st.best_trial.user_attrs["auc"]:.5f} it={st.best_trial.user_attrs["it"]} params={st.best_params} {time.time()-t0:.0f}s', flush=True)
top = sorted(st.trials, key=lambda t: t.value)[:5]
for t in top: print('  ', round(t.value,5), round(t.user_attrs['auc'],5), t.params)
# C. stacking vs mean of logits
names = ['lgb_stump','lgb_d2','lgb_d3','xgb_d2','xgb_d3','cat_d3','cat_d4','spl_0.3']
L = np.column_stack([logit(np.clip(res[n],1e-6,1-1e-6)) for n in names])
report('mean-logit blend 8 members', y, expit(L.mean(1)))
# nested stacking: LR on logits with CV
oof_s = np.zeros(len(y))
for tr, va in folds(y, seed=7):
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(L[tr], y[tr]); oof_s[va] = lr.predict_proba(L[va])[:,1]
lr = LogisticRegression(C=1.0, max_iter=2000).fit(L, y); print('stack coefs', dict(zip(names, lr.coef_[0].round(3))))
report('LR-stack on logits (nested)', y, oof_s)
# greedy caruana selection on log loss
from itertools import count
sel = []; cur = None
for k in range(1, 15):
    best = None
    for j in range(len(names)):
        c = L[:, sel + [j]].mean(1); s = log_loss(y, expit(c))
        if best is None or s < best[0]: best = (s, j)
    if cur is not None and best[0] > cur - 1e-5: break
    cur = best[0]; sel.append(best[1])
print('greedy picks', [names[j] for j in sel]); report('greedy blend', y, expit(L[:, sel].mean(1)))
# Platt on top of mean-logit blend (nested)
b = L.mean(1)[:, None]; oof_c = np.zeros(len(y))
for tr, va in folds(y, seed=7):
    oof_c[va] = LogisticRegression(C=1e6).fit(b[tr], y[tr]).predict_proba(b[va])[:,1]
report('mean-logit blend + Platt (nested)', y, oof_c)
