"""P3 threshold strategies on repeated 80/20 splits (LightGBM default + Platt on OOF).
Compares holdout normalized cost of: empirical robust threshold (current), Bayes 1/11, logit-average."""
import sys
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

sys.path.insert(0, sys.argv[2])            # folder containing solution.py
from solution import StudentSolution, Objective, _best_threshold

df = pd.read_csv(f"{sys.argv[1]}/p3_telecom_development.csv")
y_all = df["major_service_disruption"].values
X_all = df.drop(columns=["major_service_disruption", "record_id"])
for c in X_all.select_dtypes("object"):
    X_all[c] = X_all[c].astype("category")
obj = Objective({"objective": "decision_cost", "requires_threshold": True,
                 "cost_false_positive": 1, "cost_false_negative": 10})


def ncost(y, p, t):
    return (((p >= t) & (y == 0)).sum() + 10 * ((p < t) & (y == 1)).sum()) / min(10 * y.sum(), (1 - y).sum())


def mk():
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.04, num_leaves=15, min_child_samples=40,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=16)


lg = lambda p: np.log(p / (1 - p))
res = {k: [] for k in ["empirical", "bayes", "logit_avg", "oracle"]}
for rep in range(10):
    Xtr, Xte, ytr, yte = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=100 + rep)
    oof = np.zeros(len(ytr))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=rep).split(Xtr, ytr):
        oof[va] = mk().fit(Xtr.iloc[tr], ytr[tr]).predict_proba(Xtr.iloc[va])[:, 1]
    z = lg(np.clip(oof, 1e-4, 1 - 1e-4))[:, None]
    cal = LogisticRegression(C=1e6).fit(z, ytr)
    oofc = cal.predict_proba(z)[:, 1]
    pte = mk().fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    ptec = cal.predict_proba(lg(np.clip(pte, 1e-4, 1 - 1e-4))[:, None])[:, 1]
    t_emp = StudentSolution._robust_threshold(ytr, oofc, obj, np.random.default_rng(rep))
    t_bay = 1 / 11
    t_avg = 1 / (1 + np.exp(-0.5 * (lg(t_emp) + lg(t_bay))))
    ts = np.linspace(0.01, 0.5, 500)
    res["empirical"].append(ncost(yte, ptec, t_emp))
    res["bayes"].append(ncost(yte, ptec, t_bay))
    res["logit_avg"].append(ncost(yte, ptec, t_avg))
    res["oracle"].append(min(ncost(yte, ptec, t) for t in ts))
    print(f"rep{rep} t_emp={t_emp:.4f} emp={res['empirical'][-1]:.4f} bayes={res['bayes'][-1]:.4f} "
          f"avg={res['logit_avg'][-1]:.4f} oracle={res['oracle'][-1]:.4f}", flush=True)
for k, v in res.items():
    print(f"{k:10s} mean={np.mean(v):.4f} sd={np.std(v):.4f}")
