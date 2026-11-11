import pandas as pd, numpy as np, time, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
D = 'data/'
PROBS = {'p1': ('p1_manufacturing_development.csv','quality_failure'),
         'p2': ('p2_credit_development.csv','serious_delinquency'),
         'p3': ('p3_telecom_development.csv','major_service_disruption')}
def load(key):
    f, t = PROBS[key]; df = pd.read_csv(D+f)
    X = df.drop(columns=['record_id', t]); y = df[t].astype(int).values
    cats = [c for c in X.columns if X[c].dtype == object]
    nums = [c for c in X.columns if c not in cats]
    return X, y, nums, cats
def native(X, nums, cats):
    Z = X[nums].astype(float).copy()
    for c in cats:
        Z[c] = X[c].astype('category').cat.codes.replace(-1, np.nan).astype(float)
    return Z
def norm_cost(y, p, t, cfp=1, cfn=10):
    a = p >= t
    return (cfp*np.sum(a & (y==0)) + cfn*np.sum(~a & (y==1))) / min(cfn*y.sum(), cfp*(y==0).sum())
def best_cost(y, p):
    ts = np.unique(np.round(p, 4))
    cs = [norm_cost(y, p, t) for t in ts]
    i = int(np.argmin(cs)); return cs[i], ts[i]
def report(name, y, p, t0=None):
    s = f"{name:40s} AUC={roc_auc_score(y,p):.5f} LL={log_loss(y,np.clip(p,1e-6,1-1e-6)):.5f}"
    bc, bt = best_cost(y, p)
    s += f" cost@best={bc:.4f}(t={bt:.3f}) cost@1/11={norm_cost(y,p,1/11):.4f}"
    if t0: s += f" {time.time()-t0:.0f}s"
    print(s, flush=True)
def folds(y, seed=42, k=5):
    return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))
