"""Quick family comparison: 5-fold OOF on the full dev data, default-ish params."""
import sys, time
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
import lightgbm as lgb, xgboost as xgb, catboost as cb

DATA = sys.argv[1]
PROBS = [("p1_manufacturing_development.csv", "quality_failure"),
         ("p2_credit_development.csv", "serious_delinquency"),
         ("p3_telecom_development.csv", "major_service_disruption")]


def cost_curve_best(y, p):
    best = 1e18
    for t in np.linspace(0.01, 0.6, 300):
        c = ((p >= t) & (y == 0)).sum() + 10 * ((p < t) & (y == 1)).sum()
        best = min(best, c)
    return best / min(10 * y.sum(), (1 - y).sum())


def cost_at(y, p, t):
    return (((p >= t) & (y == 0)).sum() + 10 * ((p < t) & (y == 1)).sum()) / min(10 * y.sum(), (1 - y).sum())


for f, t in PROBS[int(sys.argv[2]) - 1: int(sys.argv[2])] if len(sys.argv) > 2 else PROBS:
    df = pd.read_csv(f"{DATA}/{f}")
    y = df[t].values
    X = df.drop(columns=[t, "record_id"])
    cats = X.select_dtypes("object").columns.tolist()
    Xn = X.copy()
    for c in cats:
        Xn[c] = Xn[c].astype("category")
    Xc = X.copy()
    for c in cats:
        Xc[c] = Xc[c].fillna("__NA__").astype(str)
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    res = {}
    models = {
        "lgbm": lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.04, num_leaves=15, min_child_samples=40,
                                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1,
                                           verbose=-1, n_jobs=16),
        "xgb": lambda: xgb.XGBClassifier(n_estimators=400, learning_rate=0.04, max_depth=4, min_child_weight=5,
                                         subsample=0.8, colsample_bytree=0.8, reg_lambda=1, tree_method="hist",
                                         enable_categorical=True, max_cat_to_onehot=8, n_jobs=16, verbosity=0),
        "cat": lambda: cb.CatBoostClassifier(iterations=800, learning_rate=0.05, depth=6, verbose=0,
                                             thread_count=16, allow_writing_files=False, cat_features=cats),
    }
    oofs = {}
    for name, mk in models.items():
        oof = np.zeros(len(y))
        t0 = time.time()
        for tr, va in skf.split(X, y):
            Xu = Xc if name == "cat" else Xn
            m = mk().fit(Xu.iloc[tr], y[tr])
            oof[va] = m.predict_proba(Xu.iloc[va])[:, 1]
        oofs[name] = oof
        print(f"{f[:3]} {name:5s} auc={roc_auc_score(y, oof):.4f} ll={log_loss(y, oof):.4f} "
              f"bestcost={cost_curve_best(y, oof):.4f} cost@1/11={cost_at(y, oof, 1/11):.4f} {time.time()-t0:.0f}s",
              flush=True)
    for a, b in [("lgbm", "cat"), ("xgb", "cat"), ("lgbm", "xgb")]:
        o = (oofs[a] + oofs[b]) / 2
        print(f"{f[:3]} {a}+{b} auc={roc_auc_score(y, o):.4f} ll={log_loss(y, o):.4f} bestcost={cost_curve_best(y, o):.4f}")
    o = sum(oofs.values()) / 3
    print(f"{f[:3]} all3 auc={roc_auc_score(y, o):.4f} ll={log_loss(y, o):.4f} bestcost={cost_curve_best(y, o):.4f}")
