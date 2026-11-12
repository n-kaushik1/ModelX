# How noisy is the public holdout? Quick depth-2 LightGBM OOF on each problem, then bootstrap holdout-sized samples.
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
P = {"p1": ("p1_manufacturing_development.csv", "quality_failure"), "p2": ("p2_credit_development.csv", "serious_delinquency"),
     "p3": ("p3_telecom_development.csv", "major_service_disruption")}
rng = np.random.default_rng(0)
for k, (f, t) in P.items():
    df = pd.read_csv("data/" + f); y = df[t].to_numpy(); X = df.drop(columns=[t, "record_id"])
    for c in X.columns:
        if X[c].dtype == object: X[c] = X[c].astype("category")
    oof = np.zeros(len(y))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=4, max_depth=2, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, verbose=-1, n_jobs=2).fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    n_hold = {"p1": 2500, "p2": 3000, "p3": 3750}[k]   # dev is probably 80 %: holdout = n/4
    B = 2000
    if k == "p1":
        s = [roc_auc_score(y[i], oof[i]) for i in (rng.choice(len(y), n_hold, replace=False) for _ in range(B))]
        print(k, "full OOF AUC %.4f | holdout(n=%d) AUC sd %.4f, 90%% range %.4f..%.4f" % (roc_auc_score(y, oof), n_hold, np.std(s), *np.quantile(s, [.05, .95])))
    elif k == "p2":
        s = [log_loss(y[i], oof[i]) for i in (rng.choice(len(y), n_hold, replace=False) for _ in range(B))]
        print(k, "full OOF LL %.4f | holdout(n=%d) LL sd %.4f, 90%% range %.4f..%.4f" % (log_loss(y, oof), n_hold, np.std(s), *np.quantile(s, [.05, .95])))
    else:
        def nc(yy, pp, t=1/11):
            a = pp >= t; return (np.sum(a & (yy == 0)) + 10 * np.sum(~a & (yy == 1))) / min(10 * yy.sum(), (yy == 0).sum())
        s = [nc(y[i], oof[i]) for i in (rng.choice(len(y), n_hold, replace=False) for _ in range(B))]
        npos = n_hold * y.mean()
        print(k, "full OOF cost %.4f | holdout(n=%d, ~%d positives) cost sd %.4f, 90%% range %.4f..%.4f" % (nc(y, oof), n_hold, npos, np.std(s), *np.quantile(s, [.05, .95])))
        print("   one false negative on the holdout = %.4f normalized cost; one false positive = %.5f" % (10 / (10 * npos), 1 / (10 * npos)))
        # threshold sensitivity on full OOF
        for t in (0.06, 0.07, 0.08, 0.0909, 0.10, 0.11, 0.12, 0.14):
            print("   thr %.3f cost %.4f alerts %.3f" % (t, nc(y, oof, t), (oof >= t).mean()))
