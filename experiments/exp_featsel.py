"""Nested test of shadow-feature (Boruta-lite) selection with LightGBM: selection runs inside each outer fold."""
import sys
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
import lightgbm as lgb

DATA = sys.argv[1]
PROBS = [("p1_manufacturing_development.csv", "quality_failure"),
         ("p2_credit_development.csv", "serious_delinquency"),
         ("p3_telecom_development.csv", "major_service_disruption")]


def mk(seed=0):
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.04, num_leaves=15, min_child_samples=40,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=16,
                              random_state=seed, importance_type="gain")


def select(X, y, rng, runs=4):
    """keep features whose gain beats the best shadow feature in >= half of the runs"""
    hits = np.zeros(X.shape[1])
    for r in range(runs):
        S = X.apply(lambda c: c.sample(frac=1.0, random_state=int(rng.integers(1e9))).values)
        S.columns = [f"shadow_{c}" for c in X.columns]
        for c in S.columns:
            if str(X[c[7:]].dtype) == "category":
                S[c] = S[c].astype(X[c[7:]].dtype)
        m = mk(r).fit(pd.concat([X.reset_index(drop=True), S.reset_index(drop=True)], axis=1), y)
        imp = m.feature_importances_
        hits += imp[:X.shape[1]] > imp[X.shape[1]:].max()
    return [c for c, h in zip(X.columns, hits) if h >= runs / 2]


def ncost_best(y, p):
    ts = np.linspace(0.02, 0.4, 200)
    return min(((((p >= t) & (y == 0)).sum() + 10 * ((p < t) & (y == 1)).sum()) for t in ts)) / min(10 * y.sum(), (1 - y).sum())


for f, t in PROBS:
    df = pd.read_csv(f"{DATA}/{f}")
    y = df[t].values
    X = df.drop(columns=[t, "record_id"])
    for c in X.select_dtypes("object"):
        X[c] = X[c].astype("category")
    rng = np.random.default_rng(0)
    oof_all, oof_sel = np.zeros(len(y)), np.zeros(len(y))
    kept_sets = []
    for rep in range(2):
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y):
            keep = select(X.iloc[tr], y[tr], rng)
            kept_sets.append(set(keep))
            oof_all[va] += mk().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1] / 2
            oof_sel[va] += mk().fit(X.iloc[tr][keep], y[tr]).predict_proba(X.iloc[va][keep])[:, 1] / 2
    always_dropped = set(X.columns) - set.union(*kept_sets)
    sometimes = set(X.columns) - set.intersection(*kept_sets) - always_dropped
    print(f"== {f[:2]} all: auc={roc_auc_score(y, oof_all):.4f} ll={log_loss(y, oof_all):.4f} cost={ncost_best(y, oof_all):.4f}")
    print(f"== {f[:2]} sel: auc={roc_auc_score(y, oof_sel):.4f} ll={log_loss(y, oof_sel):.4f} cost={ncost_best(y, oof_sel):.4f}")
    print(f"   always dropped: {sorted(always_dropped)}\n   sometimes dropped: {sorted(sometimes)}", flush=True)
