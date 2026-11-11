import sys; sys.path.insert(0, 'experiments/sub2'); sys.path.insert(0, 'submission/sub2')
from common import *
import lightgbm as lgb, pickle, optuna
from scipy.special import logit, expit
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import solution as S
optuna.logging.set_verbosity(optuna.logging.WARNING)
key = sys.argv[1]; NJ = 1
X, y, nums, cats = load(key)
Xc = S.DataCleaner().fit(X).transform(X)
FS = {s: folds(y, seed=s) for s in (0, 1, 2)}
METRIC = {'p1': 'auc', 'p2': 'll', 'p3': 'll'}[key]
def score(p): return roc_auc_score(y, p) if METRIC == 'auc' else -log_loss(y, np.clip(p, 1e-6, 1-1e-6))
res = {}
def fit_lgb(Xtr, ytr, Xva, params, seed=0, mono_thr=0.9):
    enc = S.TabularEncoder("native").fit(Xtr, nums, cats); Ztr, Zva = enc.transform(Xtr), enc.transform(Xva)
    kw = dict(params); k = kw.pop('inter_k', 0); use_mono = kw.pop('mono', 1)
    if k or use_mono:
        st = S._detect_structure(Xtr, ytr, nums, cats, NJ)
        if mono_thr != 0.9: st = dict(st, mono=[m if abs(r) > mono_thr else 0 for m, r in zip(st['mono'], st.get('rho', [1]*len(st['mono'])))])
        inter, mono = S._constraints_for(dict(inter_k=k, mono=use_mono), st, Ztr.shape[1])
        if inter: kw['interaction_constraints'] = inter
        if mono: kw.update(monotone_constraints=mono, monotone_constraints_method='advanced')
    m = lgb.LGBMClassifier(n_estimators=4000, verbose=-1, n_jobs=NJ, random_state=seed, cat_smooth=20, min_data_per_group=50, **kw)
    m.fit(Ztr, ytr, eval_set=[(Zva, y[Xva.index])], callbacks=[lgb.early_stopping(150, verbose=False)], categorical_feature=enc.cat_idx_ or 'auto')
    return m.predict_proba(Zva)[:, 1]
def rcv(name, params, seeds=(0, 1, 2), fitter=None, quiet=False):
    t0 = time.time(); scs = []; oofs = []
    for s in seeds:
        oof = np.zeros(len(y))
        for tr, va in FS[s]:
            oof[va] = (fitter or fit_lgb)(Xc.iloc[tr], y[tr], Xc.iloc[va], params)
        scs.append(score(oof)); oofs.append(oof)
    res[name] = oofs
    if not quiet: print(f"{name:48s} {METRIC}={np.mean(scs):.5f} sd={np.std(scs):.5f} per-seed={[round(s,5) for s in scs]} {time.time()-t0:.0f}s", flush=True)
    return np.mean(scs)
B = dict(learning_rate=0.03, max_depth=2, num_leaves=4, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, inter_k=8, mono=1)
print('=====', key, '===== (repeated 5-fold x 3 fold seeds; higher is better: AUC or -logloss)', flush=True)
base = rcv('B structured d2 inter8 mono', B)
rcv('B inter_k=16', dict(B, inter_k=16)); rcv('B inter_k=4', dict(B, inter_k=4))
rcv('B dart (drop_rate .1)', dict(B, boosting_type='dart', drop_rate=0.1, learning_rate=0.05))
rcv('B num_leaves=3', dict(B, num_leaves=3))
rcv('B depth3 leaves6 inter16', dict(B, max_depth=3, num_leaves=6, inter_k=16))
rcv('B min_sum_hessian 5', dict(B, min_child_weight=5))
rcv('B cat_l2 30 cat_smooth 50', dict(B, cat_l2=30))
# hinge GAM: ReLU bases at quantile knots + LR (different bias from cubic splines and from trees)
def fit_hinge(Xtr, ytr, Xva, params):
    enc = S.TabularEncoder("linear").fit(Xtr, nums, cats); Ztr, Zva = enc.transform(Xtr), enc.transform(Xva)
    nn = len(nums); knots = [np.quantile(Ztr[:, j], np.linspace(0.1, 0.9, params['k'])) for j in range(nn)]
    def design(Z):
        H = [Z] + [np.maximum(Z[:, j:j+1] - knots[j][None, :], 0) for j in range(nn)] + [np.maximum(knots[j][None, :] - Z[:, j:j+1], 0) for j in range(nn)]
        return np.hstack(H)
    lr = LogisticRegression(C=params['C'], max_iter=3000).fit(design(Ztr), ytr); return lr.predict_proba(design(Zva))[:, 1]
for C in (0.05, 0.2):
    rcv(f'hinge-GAM k=7 C={C}', dict(k=7, C=C), fitter=fit_hinge)
def blend(names, w=None):
    out = []
    for i in range(3):
        L = np.array([logit(np.clip(res[n][i], 1e-6, 1-1e-6)) for n in names]); ww = np.ones(len(names))/len(names) if w is None else np.array(w)
        out.append(expit(np.tensordot(ww, L, 1)))
    return out
def rep(name, oofs): print(f"{name:48s} {METRIC}={np.mean([score(o) for o in oofs]):.5f} per-seed={[round(score(o),5) for o in oofs]}", flush=True)
hg = 'hinge-GAM k=7 C=0.2'
rep('blend B + hinge-GAM (1:1)', blend(['B structured d2 inter8 mono', hg]))
rep('blend B + hinge-GAM (3:1)', blend(['B structured d2 inter8 mono', hg], [0.75, 0.25]))
rep('blend B + B16 + B4', blend(['B structured d2 inter8 mono', 'B inter_k=16', 'B inter_k=4']))
# Optuna on the structured space, fold seed 0 only; then check the best on fold seeds 1, 2 (does the gain transfer?)
def objective(trial):
    p = dict(learning_rate=trial.suggest_float('lr', 0.01, 0.08, log=True), max_depth=trial.suggest_int('depth', 1, 4),
             min_child_samples=trial.suggest_int('mcs', 5, 150, log=True), subsample=trial.suggest_float('subsample', 0.5, 1.0), subsample_freq=1,
             colsample_bytree=trial.suggest_float('colsample', 0.3, 1.0), reg_lambda=trial.suggest_float('l2', 1e-3, 30, log=True),
             reg_alpha=trial.suggest_float('l1', 1e-3, 3, log=True), inter_k=trial.suggest_categorical('inter_k', [0, 4, 8, 12, 16, 24, 32]),
             mono=trial.suggest_categorical('mono', [0, 1]))
    p['num_leaves'] = trial.suggest_int('leaves', 2, 2 ** p['max_depth'])
    return rcv('opt', p, seeds=(0,), quiet=True)
t0 = time.time(); st = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=12)); st.optimize(objective, n_trials=45)
bp = st.best_params; P = dict(learning_rate=bp['lr'], max_depth=bp['depth'], num_leaves=bp['leaves'], min_child_samples=bp['mcs'], subsample=bp['subsample'], subsample_freq=1, colsample_bytree=bp['colsample'], reg_lambda=bp['l2'], reg_alpha=bp['l1'], inter_k=bp['inter_k'], mono=bp['mono'])
print(f"optuna best on fold-seed 0: {st.best_value:.5f} params={bp} ({time.time()-t0:.0f}s)", flush=True)
rcv('optuna-best checked on fold seeds 0,1,2', P)
rep('blend B + optuna-best', blend(['B structured d2 inter8 mono', 'optuna-best checked on fold seeds 0,1,2']))
# calibration variants on the B OOF (nested 5-fold inside the OOF)
if METRIC == 'll':
    for cname in ('none', 'platt', 'isotonic', 'beta'):
        sc = []
        for i in range(3):
            p = np.clip(res['B structured d2 inter8 mono'][i], 1e-6, 1-1e-6); out = p.copy()
            if cname != 'none':
                for tr, va in folds(y, seed=7):
                    if cname == 'platt': out[va] = LogisticRegression(C=1e6).fit(logit(p[tr])[:, None], y[tr]).predict_proba(logit(p[va])[:, None])[:, 1]
                    elif cname == 'isotonic': out[va] = IsotonicRegression(out_of_bounds='clip', y_min=1e-4, y_max=1-1e-4).fit(p[tr], y[tr]).predict(p[va])
                    else:
                        Zb = np.column_stack([np.log(p), -np.log(1-p)]); out[va] = LogisticRegression(C=1e6).fit(Zb[tr], y[tr]).predict_proba(Zb[va])[:, 1]
            sc.append(score(out))
        print(f"calibration {cname:9s} {METRIC}={np.mean(sc):.5f} per-seed={[round(s,5) for s in sc]}", flush=True)
pickle.dump(res, open(f'experiments/sub2/exp8_{key}.pkl', 'wb'))
