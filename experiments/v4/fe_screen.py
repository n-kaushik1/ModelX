"""OpenFE-style screen: does an engineered feature carry signal that a strong base model misses?
Base = LightGBM depth 2 OOF logits.  For every candidate, a small residual booster (init_score = base
logit, the candidate as its only input) is cross-validated; the OOF log-loss gain is reported.
Raw columns are re-offered as candidates to give the noise floor."""
import sys, itertools, time
import numpy as np, pandas as pd, lightgbm as lgb
from joblib import Parallel, delayed
from kit import *

key = sys.argv[1]
log = Log(f"fe_screen_{key}.log")
X, y, md = load(key)
cats = [c for c in X.columns if X[c].dtype == object]
nums = [c for c in X.columns if c not in cats]
Z = X[nums].astype(float).copy()
for c in cats:
    Z[c] = X[c].astype("category").cat.codes.replace(-1, np.nan).astype(float)

# ---- base OOF logits (5-fold, ES on the val fold; screening only) ----
t0 = time.time()
base = np.zeros(len(y))
for tr, va in folds(y, 0):
    m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.03, max_depth=2, num_leaves=4, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, reg_lambda=1, verbose=-1, n_jobs=16)
    m.fit(Z.iloc[tr], y[tr], eval_set=[(Z.iloc[va], y[va])], callbacks=[lgb.early_stopping(150, verbose=False)],
          categorical_feature=[Z.columns.get_loc(c) for c in cats])
    base[va] = m.predict_proba(Z.iloc[va], raw_score=True)
pb = sigmoid(base)
LL0 = ll(y, pb)
log(f"===== {key} base LL={LL0:.5f} AUC={auc(y, pb):.5f} ({time.time()-t0:.0f}s)")

# ---- candidates ----
zs = (Z[nums] - Z[nums].mean()) / Z[nums].std()
cand = {}
for c in nums + cats:
    cand[f"RAW:{c}"] = (Z[c].to_numpy(), c in cats)
rng = np.random.default_rng(0)
for i in range(5):
    cand[f"NOISE:{i}"] = (rng.normal(size=len(y)), False)
for a, b in itertools.combinations(nums, 2):
    A, B = Z[a].to_numpy(), Z[b].to_numpy()
    za, zb = zs[a].to_numpy(), zs[b].to_numpy()
    with np.errstate(all="ignore"):
        r1 = np.where(np.abs(B) > 1e-12, A / B, np.nan)
        r2 = np.where(np.abs(A) > 1e-12, B / A, np.nan)
    cand[f"{a}/{b}"] = (r1, False)
    cand[f"{b}/{a}"] = (r2, False)
    cand[f"{a}*{b}"] = (A * B, False)
    cand[f"z{a}-z{b}"] = (za - zb, False)
    cand[f"z{a}+z{b}"] = (za + zb, False)
    cand[f"z{a}*z{b}"] = (za * zb, False)
for g in cats:
    for c in nums:
        mu = Z.groupby(X[g].fillna("__na__"))[c].transform("mean").to_numpy()
        sd = Z.groupby(X[g].fillna("__na__"))[c].transform("std").to_numpy()
        cand[f"{c}-mean|{g}"] = (Z[c].to_numpy() - mu, False)
        cand[f"{c}z|{g}"] = ((Z[c].to_numpy() - mu) / sd, False)
for a, b in itertools.combinations(cats, 2):
    combo = (X[a].fillna("NA").astype(str) + "|" + X[b].fillna("NA").astype(str))
    cand[f"{a}x{b}"] = (combo.astype("category").cat.codes.astype(float).to_numpy(), True)
for c in nums:
    vc = Z[c].round(6).map(Z[c].round(6).value_counts())
    cand[f"COUNT:{c}"] = (vc.to_numpy(dtype=float), False)
log(f"{len(cand)} candidates")

F2 = folds(y, 1)


def score(name, x, is_cat):
    oof = np.zeros(len(y))
    x = x.reshape(-1, 1)
    for tr, va in F2:
        m = lgb.LGBMClassifier(n_estimators=40, learning_rate=0.1, num_leaves=4, max_depth=2, min_child_samples=100,
                               verbose=-1, n_jobs=1, cat_smooth=20, min_data_per_group=100)
        m.fit(x[tr], y[tr], init_score=base[tr], categorical_feature=[0] if is_cat else "auto")
        oof[va] = base[va] + m.predict(x[va], raw_score=True)
    return name, LL0 - ll(y, sigmoid(oof))


t0 = time.time()
out = Parallel(n_jobs=16, batch_size=8)(delayed(score)(n, x, c) for n, (x, c) in cand.items())
out.sort(key=lambda t: -t[1])
floor = max(g for n, g in out if n.startswith(("RAW:", "NOISE:")))
log(f"scored in {time.time()-t0:.0f}s; noise floor (best RAW/NOISE gain) = {floor:.5f}")
for n, g in out[:45]:
    log(f"  {g:+.5f}  {n}")
log("raw/noise:", ", ".join(f"{n}:{g:+.5f}" for n, g in out if n.startswith(("RAW:", "NOISE:")))[:3000])
pd.DataFrame(out, columns=["cand", "gain"]).to_csv(f"{WORK}/logs/fe_screen_{key}.csv", index=False)
