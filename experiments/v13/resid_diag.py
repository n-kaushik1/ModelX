"""What does the whole framework still miss?  Uses the outer-CV held-out predictions of a framework tag
(logs/outer/<tag>_<key>_s<seed>_f<fold>.npz, every row predicted by a framework that never saw it).
usage: python resid_diag.py <tag> <key> [n_jobs]
  A. calibration of the held-out probabilities (reliability z-scores, nested Platt gain)
  B. misfit per feature: sum(y - p) per quantile bin / level, chi-square against the binomial variance
  C. misfit per numeric pair: 5 x 5 quantile cells, chi-square
  D. residual boosters: LightGBM with init_score = framework logit, own 5-fold CV -> held-out gain
"""
import sys, os, warnings
import numpy as np, pandas as pd
from scipy import stats
import lightgbm as lgb
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v4"))
from kit import load, folds, ll, auc, logit, sigmoid, norm_cost
warnings.filterwarnings("ignore")

tag, key = sys.argv[1], sys.argv[2]
nj = int(sys.argv[3]) if len(sys.argv) > 3 else 4
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v5", "logs", "outer")
X, y, md = load(key)
num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
cat = [c for c in X.columns if c not in num]


def held_out(seed):
    p = np.full(len(y), np.nan); t = []
    for f in range(5):
        z = np.load(os.path.join(OUT, f"{tag}_{key}_s{seed}_f{f}.npz"), allow_pickle=True)
        p[z["va"]] = z["p"]; t.append(float(z["thr"]))
    return p, float(np.mean(t))


def chi(r, v, groups):
    """chi-square of per-group residual sums against the binomial variance; returns (stat, df, worst z, group)."""
    d = pd.DataFrame({"r": r, "v": v, "g": groups}).groupby("g", dropna=False)[["r", "v"]].sum()
    z = d["r"] / np.sqrt(d["v"])
    return float((z ** 2).sum()), len(d), float(z.abs().max()), z.abs().idxmax()


def bins(s, q):
    s = pd.to_numeric(s, errors="coerce")
    b = pd.qcut(s.rank(method="first"), q, labels=False)
    return b.where(s.notna(), -1).astype(int).to_numpy()


for seed in (0, 1):
    p, thr = held_out(seed)
    r, v = y - p, p * (1 - p)
    print(f"\n===== {tag} {key} seed {seed}: LL {ll(y, p):.5f} AUC {auc(y, p):.5f}"
          + (f" cost {norm_cost(y, p, thr):.4f}" if key == "p3" else ""), flush=True)

    # A. calibration
    s, df, zmax, g = chi(r, v, bins(pd.Series(p), 20))
    print(f"A reliability 20 bins: chi2={s:.1f} df={df} p={stats.chi2.sf(s, df):.3f} worst|z|={zmax:.2f} (bin {g})")
    L = logit(p)
    pl = np.zeros(len(y))
    for tr, va in folds(y, 7):
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(C=1e4).fit(L[tr, None], y[tr])
        pl[va] = m.predict_proba(L[va, None])[:, 1]
    print(f"  nested Platt on held-out logits: LL {ll(y, p):.5f} -> {ll(y, pl):.5f}")

    # B. per feature
    rows = []
    for c in X.columns:
        g = bins(X[c], 20) if c in num else X[c].astype(str).to_numpy()
        s, df, zmax, gg = chi(r, v, g)
        rows.append((stats.chi2.sf(s, df), c, s, df, zmax, gg))
    rows.sort()
    nb = len(rows)
    print(f"B per-feature misfit (Bonferroni over {nb} features), top 6:")
    for pv, c, s, df, zmax, gg in rows[:6]:
        print(f"   {c:32s} chi2={s:7.1f} df={df:3d} p={pv:.4f} padj={min(1, pv * nb):.3f} worst|z|={zmax:.2f} at {gg}")

    # C. numeric pairs
    q = {c: bins(X[c], 5) for c in num}
    pr = []
    for i, a in enumerate(num):
        for b in num[i + 1:]:
            s, df, zmax, gg = chi(r, v, q[a] * 10 + q[b])
            pr.append((stats.chi2.sf(s, df), a, b, s, df, zmax))
    pr.sort()
    npairs = len(pr)
    print(f"C pair misfit (Bonferroni over {npairs} pairs), top 5:")
    for pv, a, b, s, df, zmax in pr[:5]:
        print(f"   {a} x {b}: chi2={s:.1f} df={df} p={pv:.5f} padj={min(1, pv * npairs):.3f} worst|z|={zmax:.2f}")

    # D. residual boosters on the held-out logits
    Xl = X.copy()
    for c in cat:
        Xl[c] = Xl[c].astype("category")
    for depth, leaves, lr, mcs in [(1, 2, 0.02, 50), (2, 4, 0.02, 50), (3, 8, 0.02, 50), (6, 31, 0.05, 100)]:
        pd_ = np.zeros(len(y)); its = []
        for tr, va in folds(y, 11):
            itr, ies = folds(y[tr], 3)[0]                       # inner early-stopping split
            a_, b_ = tr[itr], tr[ies]
            prm = dict(objective="binary", learning_rate=lr, max_depth=depth, num_leaves=leaves,
                       min_child_samples=mcs, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                       reg_lambda=5, verbose=-1, num_threads=nj, seed=0)
            dtr = lgb.Dataset(Xl.iloc[a_], y[a_], init_score=L[a_])
            des = lgb.Dataset(Xl.iloc[b_], y[b_], init_score=L[b_])
            bst = lgb.train(prm, dtr, 2000, valid_sets=[des], callbacks=[lgb.early_stopping(100, verbose=False)])
            n_it = max(1, bst.best_iteration)
            its.append(n_it)
            dfull = lgb.Dataset(Xl.iloc[tr], y[tr], init_score=L[tr])
            bst = lgb.train(prm, dfull, n_it)
            pd_[va] = sigmoid(L[va] + bst.predict(Xl.iloc[va], raw_score=True))
        extra = ""
        if key == "p3":
            extra = f" cost@1/11 {norm_cost(y, p, 1 / 11):.4f}->{norm_cost(y, pd_, 1 / 11):.4f}"
        print(f"D residual lgbm d{depth}: LL {ll(y, p):.5f}->{ll(y, pd_):.5f} AUC {auc(y, p):.5f}->{auc(y, pd_):.5f}"
              f" trees={its}{extra}", flush=True)
