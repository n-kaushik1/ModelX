import sys; sys.path.insert(0, 'experiments/sub2'); sys.path.insert(0, 'submission/sub2')
from common import *
import lightgbm as lgb, pickle
from scipy.special import logit, expit
import solution as S
key = sys.argv[1]; NJ = 1
X, y, nums, cats = load(key); F = folds(y)
Xc = S.DataCleaner().fit(X).transform(X)          # same cleaning as the framework
res = {}
def run(name, params, onehot=False, drop_zero=False, seeds=(0,), max_bin=255, struct=True):
    t0 = time.time(); oof = np.zeros(len(y)); its = []
    for s in seeds:
        for tr, va in F:
            Xtr, Xva, ytr = Xc.iloc[tr], Xc.iloc[va], y[tr]
            mode_cats = [] if onehot else cats
            if onehot:
                enc = S.TabularEncoder("imputed").fit(Xtr, nums, cats)   # one-hot, imputed
            else:
                enc = S.TabularEncoder("native").fit(Xtr, nums, cats)
            Ztr, Zva = enc.transform(Xtr), enc.transform(Xva)
            kw = dict(params)
            if struct:
                st = S._detect_structure(Xtr, ytr, nums, cats, NJ)
                if onehot:   # map structure to one-hot layout: numerics first in both layouts; pairs only among numerics
                    nn = len(nums); pairs = [p for p in st['pairs'] if max(p) < nn]
                    inter = [[i] for i in range(Ztr.shape[1])] + [list(p) for p in pairs[:kw.pop('inter_k', 8)]]
                    mono = list(st['mono'][:nn]) + [0]*(Ztr.shape[1]-nn)
                else:
                    inter, mono = S._constraints_for(dict(inter_k=kw.pop('inter_k', 8), mono=1), st, Ztr.shape[1])
                if inter: kw['interaction_constraints'] = inter
                if mono and kw.pop('use_mono', True): kw.update(monotone_constraints=mono, monotone_constraints_method='advanced')
                kw.pop('use_mono', None)
            else:
                kw.pop('inter_k', None); kw.pop('use_mono', None)
            if drop_zero:
                pre = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=2, verbose=-1, n_jobs=NJ, random_state=0).fit(Ztr, ytr, categorical_feature=enc.cat_idx_ or 'auto')
                keep = np.where(pre.booster_.feature_importance('split') > 0)[0]
                if 'interaction_constraints' in kw:
                    remap = {old: new for new, old in enumerate(keep)}
                    kw['interaction_constraints'] = [[remap[i] for i in g if i in remap] for g in kw['interaction_constraints']]
                    kw['interaction_constraints'] = [g for g in kw['interaction_constraints'] if g]
                if 'monotone_constraints' in kw: kw['monotone_constraints'] = [kw['monotone_constraints'][i] for i in keep]
                Ztr, Zva = Ztr[:, keep], Zva[:, keep]; cat_idx = [list(keep).index(i) for i in (enc.cat_idx_ or []) if i in keep]
            else:
                cat_idx = enc.cat_idx_ or []
            m = lgb.LGBMClassifier(n_estimators=5000, verbose=-1, n_jobs=NJ, random_state=s, max_bin=max_bin, cat_smooth=20, min_data_per_group=50, **kw)
            m.fit(Ztr, ytr, eval_set=[(Zva, y[va])], callbacks=[lgb.early_stopping(150, verbose=False)], categorical_feature=cat_idx if cat_idx else 'auto')
            oof[va] += m.predict_proba(Zva)[:, 1]/len(seeds); its.append(m.best_iteration_)
    res[name] = oof; report(f"{name} it={int(np.mean(its))}", y, oof, t0); return oof
B = dict(learning_rate=0.03, max_depth=2, num_leaves=4, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, inter_k=8)
print('=====', key, '=====', flush=True)
run('B: d2 inter8 mono (native cats)', B)
run('B + 5 seeds', B, seeds=(0,1,2,3,4))
run('B one-hot cats', B, onehot=True)
run('B max_bin 1023', B, max_bin=1023)
run('B max_bin 63', B, max_bin=63)
run('B extra_trees', dict(B, extra_trees=True))
run('B drop zero-gain feats (nested)', B, drop_zero=True)
run('B min_child 50', dict(B, min_child_samples=50))
run('B colsample 1.0 subsample 1.0', dict(B, colsample_bytree=1.0, subsample=1.0))
run('B lr 0.015', dict(B, learning_rate=0.015))
run('B path_smooth 10', dict(B, path_smooth=10))
run('B inter8 no mono', dict(B, use_mono=False))
try: run('linear_tree stumps (no constraints)', dict(learning_rate=0.03, num_leaves=2, min_child_samples=50, linear_tree=True, linear_lambda=1.0), struct=False)
except Exception as e: print('linear_tree stumps failed', repr(e)[:200])
try: run('linear_tree d2 (no constraints)', dict(learning_rate=0.03, max_depth=2, num_leaves=4, min_child_samples=50, linear_tree=True, linear_lambda=1.0, subsample=0.8, subsample_freq=1, colsample_bytree=0.8), struct=False)
except Exception as e: print('linear_tree d2 failed', repr(e)[:200])
try: run('linear_tree d2 + inter8 (no mono)', dict(B, linear_tree=True, linear_lambda=1.0, min_child_samples=50, use_mono=False))
except Exception as e: print('linear_tree constrained failed', repr(e)[:200])
pickle.dump(res, open(f'experiments/sub2/exp7_{key}.pkl','wb'))
def lb(names): return expit(np.mean([logit(np.clip(res[n],1e-6,1-1e-6)) for n in names if n in res], 0))
report('blend B + one-hot + max_bin1023 + extra_trees', y, lb(['B: d2 inter8 mono (native cats)', 'B one-hot cats', 'B max_bin 1023', 'B extra_trees']))
if 'linear_tree d2 (no constraints)' in res: report('blend B + linear_tree d2', y, lb(['B: d2 inter8 mono (native cats)', 'linear_tree d2 (no constraints)']))
