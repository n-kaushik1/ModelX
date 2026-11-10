import sys; sys.path.insert(0, 'experiments/sub2'); sys.path.insert(0, 'submission/sub2')
from common import *
import pickle
from scipy.special import logit, expit
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from solution import _conf_curve
X, y, nums, cats = load('p3'); Z = native(X, nums, cats); F = folds(y)
res = pickle.load(open('experiments/sub2/exp2_p3.pkl','rb'))
Zi = Z.copy()
for c in nums: Zi[c+'_na'] = Zi[c].isna().astype(float); Zi[c] = Zi[c].fillna(Zi[c].median())
Zi = pd.get_dummies(Zi, columns=cats, dummy_na=True).astype(float)
for ml in (3, 10):
    t0=time.time(); oof = np.zeros(len(y))
    for tr, va in F:
        m = ExtraTreesClassifier(400, min_samples_leaf=ml, max_features=0.4, n_jobs=2, random_state=0).fit(Zi.iloc[tr], y[tr]); oof[va] = m.predict_proba(Zi.iloc[va])[:,1]
    res[f'et_{ml}'] = oof; report(f'ET min_leaf={ml}', y, oof, t0)
pickle.dump(res, open('experiments/sub2/exp4_p3.pkl','wb'))
def lb(names): return expit(np.mean([logit(np.clip(res[n],1e-6,1-1e-6)) for n in names], 0))
def fast_best(yv, pv, smooth=0):
    thr, tp, fp, fn, tn = _conf_curve(yv, pv); sc = -(fp + 10*fn)
    if smooth:
        k = np.ones(smooth)/smooth; h = smooth//2
        sm = np.convolve(np.pad(sc, (h, smooth-1-h), mode='edge'), k, mode='valid'); i = int(np.argmax(sm))
    else: i = int(np.argmax(sc))
    t = thr[i] if i+1 >= len(thr) else 0.5*(thr[i]+thr[i+1])
    return t, -sc[int(np.argmax(sc))]/min(10*yv.sum(), (yv==0).sum())
cands = {'cat_d4': res['cat_d4'], 'cat_d3': res['cat_d3'], 'xgb_d3': res['xgb_d3'], 'lgb_d2': res['lgb_d2'], 'et_3': res['et_3'], 'et_10': res['et_10'],
         'blend cat4+xgb3+cat3 (LL-greedy)': lb(['cat_d4','xgb_d3','cat_d3']),
         'blend cat4+xgb3+lgb2+et10': lb(['cat_d4','xgb_d3','lgb_d2','et_10']),
         'blend 8 (st,d2,d3,xgb2,xgb3,cat3,cat4,spl)': lb(['lgb_stump','lgb_d2','lgb_d3','xgb_d2','xgb_d3','cat_d3','cat_d4','spl_0.3'])}
rng = np.random.default_rng(0); n = len(y)
def robust_t(yA, pA):
    sw = max(5, len(yA)//400); ts = []
    for _ in range(100):
        idx = rng.integers(0, len(yA), len(yA)); ts.append(fast_best(yA[idx], pA[idx], sw)[0])
    return 0.5*(np.median(ts)+fast_best(yA, pA, sw)[0])
print(f"{'model':45s} {'LL':>7s} {'emp-thr':>8s} {'bayes(cal)':>10s} {'logit-avg':>10s} {'oracle':>7s}")
splits = [rng.permutation(n) for _ in range(100)]
for name, p in cands.items():
    emp, bay, avg, ora = [], [], [], []
    for perm in splits:
        A, B = perm[:n//2], perm[n//2:]
        cal = LogisticRegression(C=1e6).fit(logit(np.clip(p[A],1e-6,1-1e-6))[:,None], y[A])
        pc = lambda idx: cal.predict_proba(logit(np.clip(p[idx],1e-6,1-1e-6))[:,None])[:,1]
        pA, pB = pc(A), pc(B); te = robust_t(y[A], pA)
        emp.append(norm_cost(y[B], pB, te)); bay.append(norm_cost(y[B], pB, 1/11))
        avg.append(norm_cost(y[B], pB, expit(0.5*(logit(te)+logit(1/11))))); ora.append(fast_best(y[B], pB)[1])
    print(f"{name:45s} {log_loss(y,p):7.5f} {np.mean(emp):8.4f} {np.mean(bay):10.4f} {np.mean(avg):10.4f} {np.mean(ora):7.4f}  sd={np.std(emp):.3f}", flush=True)
