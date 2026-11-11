"""Synthetic binary-classification problems in the style of the public development data, for testing the
framework's generalisation (development only: the framework itself never sees these).

Every problem: numeric columns with assorted marginals (normal, skewed, bounded, counts), low-cardinality
categoricals, an additive logit made of hinge / U / step / sigmoid / linear / bump effects of a subset of the
columns, sparse pairwise interactions (product, AND-rule, numeric x category), categorical main effects,
distractor columns, noisy copies (copy = original + noise), an exact ratio column, MCAR and MNAR missingness.
The logit is scaled to a target spread and shifted to a target positive rate; y ~ Bernoulli(sigmoid(logit)).
make(pid) returns (Xtr, ytr, Xte, yte, p_true_te, metadata, spec)."""
import numpy as np, pandas as pd

SPECS = [
    # name, n_train, pos_rate, objective, (c_fp, c_fn), n_num, n_cat, logit_sd, n_inter, extras
    dict(name="p1_like", n=10000, rate=0.30, obj="roc_auc", nn=20, nc=7, sd=1.6, ni=2),
    dict(name="p2_like", n=12000, rate=0.27, obj="log_loss", nn=30, nc=3, sd=1.3, ni=2, mono=True),
    dict(name="p3_like", n=15000, rate=0.06, obj="decision_cost", cost=(1, 10), nn=29, nc=6, sd=1.5, ni=2),
    dict(name="small_auc", n=1500, rate=0.20, obj="roc_auc", nn=12, nc=3, sd=1.5, ni=1),
    dict(name="small_cost20", n=3000, rate=0.05, obj="decision_cost", cost=(1, 20), nn=15, nc=3, sd=1.6, ni=1),
    dict(name="large_ll", n=80000, rate=0.10, obj="log_loss", nn=18, nc=4, sd=1.5, ni=3),
    dict(name="strong_inter", n=10000, rate=0.35, obj="roc_auc", nn=15, nc=3, sd=1.8, ni=6, inter_scale=2.0),
    dict(name="linearish", n=5000, rate=0.25, obj="log_loss", nn=15, nc=3, sd=1.4, ni=0, shapes=("linear",)),
    dict(name="highcard", n=20000, rate=0.20, obj="roc_auc", nn=12, nc=3, sd=1.5, ni=1, highcard=150),
    dict(name="many_noise", n=8000, rate=0.30, obj="log_loss", nn=100, nc=4, sd=1.4, ni=1, informative=10),
    dict(name="mnar_heavy", n=10000, rate=0.30, obj="roc_auc", nn=18, nc=4, sd=1.5, ni=1, miss=0.25),
    dict(name="fp_costly", n=10000, rate=0.40, obj="decision_cost", cost=(5, 1), nn=18, nc=4, sd=1.5, ni=1),
    dict(name="extreme_imb", n=30000, rate=0.015, obj="roc_auc", nn=20, nc=4, sd=1.8, ni=2),
    dict(name="weak_balanced", n=6000, rate=0.50, obj="log_loss", nn=20, nc=4, sd=0.8, ni=1),
    dict(name="catnum_inter", n=12000, rate=0.25, obj="log_loss", nn=16, nc=5, sd=1.5, ni=3, catnum=True),
    dict(name="cost5_mid", n=8000, rate=0.15, obj="decision_cost", cost=(1, 5), nn=22, nc=5, sd=1.4, ni=2),
]


def _marginal(rng, n, kind):
    if kind == "normal":
        return rng.normal(rng.uniform(-50, 200), rng.uniform(0.5, 30), n)
    if kind == "skew":
        return np.exp(rng.normal(rng.uniform(0, 8), rng.uniform(0.5, 1.2), n))
    if kind == "bounded":
        return rng.beta(rng.uniform(1.5, 5), rng.uniform(1.5, 5), n)
    if kind == "count":
        return rng.poisson(rng.uniform(0.5, 6), n).astype(float)
    return rng.uniform(rng.uniform(-10, 10), rng.uniform(11, 100), n)


def _z(x):
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)


def _shape(rng, z, kind):
    k = rng.uniform(-1.0, 1.0)
    a = rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)
    if kind == "linear":
        return a * z
    if kind == "hinge":
        return a * np.maximum(z - k, 0)
    if kind == "U":
        return abs(a) * 0.6 * (z - k * 0.5) ** 2
    if kind == "step":
        return a * (z > k)
    if kind == "sigmoid":
        return a * np.tanh(2 * (z - k))
    if kind == "bump":
        return a * np.exp(-((z - k) ** 2))
    raise ValueError(kind)


def make(pid, n_test=20000):
    sp = SPECS[pid]
    rng = np.random.default_rng(7919 + 31 * pid)
    n = sp["n"] + n_test
    nn, nc = sp["nn"], sp["nc"]
    kinds = rng.choice(["normal", "normal", "skew", "bounded", "count", "uniform"], nn)
    X = pd.DataFrame({f"x{j:02d}": _marginal(rng, n, kinds[j]) for j in range(nn)})
    for j in range(nc):
        L = int(rng.integers(2, 7))
        p = rng.dirichlet(np.ones(L) * 3)
        X[f"cat{j}"] = rng.choice([f"L{j}_{i}" for i in range(L)], n, p=p)
    if sp.get("highcard"):
        H = sp["highcard"]
        X["hc"] = rng.choice([f"H{i:03d}" for i in range(H)], n, p=rng.dirichlet(np.ones(H) * 0.7))
    Z = {c: _z(X[c].to_numpy()) for c in X.columns if c.startswith("x")}
    n_inf = sp.get("informative", max(4, int(0.6 * nn)))
    inf = list(rng.choice(sorted(Z), size=min(n_inf, nn), replace=False))
    shapes = sp.get("shapes", ("hinge", "U", "step", "sigmoid", "linear", "bump", "hinge", "linear"))
    logit = np.zeros(n)
    for c in inf:
        kind = "linear" if sp.get("mono") and rng.random() < 0.5 else rng.choice(shapes)
        if sp.get("mono") and kind in ("U", "bump"):
            kind = "hinge"
        w = rng.uniform(0.3, 1.0)
        logit += w * _shape(rng, Z[c], kind)
    for j in range(nc):
        if rng.random() < 0.6:
            lv = X[f"cat{j}"].unique()
            eff = dict(zip(lv, rng.normal(0, 0.4, len(lv))))
            logit += X[f"cat{j}"].map(eff).to_numpy()
    if sp.get("highcard"):
        lv = X["hc"].unique()
        eff = dict(zip(lv, rng.normal(0, 0.8, len(lv))))
        logit += X["hc"].map(eff).to_numpy()
    for _ in range(sp["ni"]):
        a, b = rng.choice(inf, 2, replace=False)
        s = sp.get("inter_scale", 1.0) * rng.uniform(0.4, 0.9) * rng.choice([-1, 1])
        t = rng.choice(["prod", "and", "and"])
        if t == "prod":
            logit += s * Z[a] * Z[b]
        else:
            logit += 2 * s * (Z[a] > rng.uniform(-0.5, 1)) * (Z[b] > rng.uniform(-0.5, 1))
    if sp.get("catnum"):
        for j in range(min(2, nc)):
            c = rng.choice(inf)
            lv = X[f"cat{j}"].unique()
            sl = dict(zip(lv, rng.normal(0, 0.8, len(lv))))
            logit += X[f"cat{j}"].map(sl).to_numpy() * Z[c]
    # scale and shift
    logit = (logit - logit.mean()) / (logit.std() + 1e-12) * sp["sd"]
    lo, hi = -15.0, 15.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if np.mean(1 / (1 + np.exp(-(logit + mid)))) > sp["rate"]:
            hi = mid
        else:
            lo = mid
    p = 1 / (1 + np.exp(-(logit + 0.5 * (lo + hi))))
    y = (rng.random(n) < p).astype(int)
    # nuisance columns built after the signal: noisy copies, an exact ratio, missingness
    for c in rng.choice(inf, min(3, len(inf)), replace=False):
        X[f"{c}_copy"] = X[c] + rng.normal(0, 0.25 * np.nanstd(X[c]), n)
    if pid % 3 == 1:                       # "twin sensors": the true column is replaced by two equally noisy readings
        c = rng.choice([v for v in inf if v in X.columns])
        sd_c = np.nanstd(X[c])
        X[f"{c}_s1"] = X[c] + rng.normal(0, 0.3 * sd_c, n)
        X[f"{c}_s2"] = X[c] + rng.normal(0, 0.3 * sd_c, n)
        X = X.drop(columns=[c] + [k for k in X.columns if k == f"{c}_copy"])
        Z = {k: v for k, v in Z.items() if k != c}
    pos = [c for c in Z if (X[c] > 0).all()]
    if len(pos) >= 2:
        X["ratio"] = X[pos[0]] / X[pos[1]]
    miss = sp.get("miss", 0.05)
    for c in rng.choice(sorted(k for k in Z if k in X.columns), max(2, nn // 4), replace=False):
        X.loc[rng.random(n) < rng.uniform(0.01, miss), c] = np.nan
    c = rng.choice([v for v in inf if v in X.columns])             # MNAR: high values go missing more often
    X.loc[(rng.random(n) < miss * 1.5) & (Z[c] > 0.5), c] = np.nan
    for j in range(min(2, nc)):
        X.loc[rng.random(n) < 0.03, f"cat{j}"] = None
    X = X[rng.permutation(X.columns)]
    ntr = sp["n"]
    cost = sp.get("cost")
    md = {"objective": sp["obj"], "greater_is_better": sp["obj"] == "roc_auc",
          "requires_threshold": sp["obj"] == "decision_cost",
          "cost_false_positive": cost[0] if cost else None, "cost_false_negative": cost[1] if cost else None,
          "positive_label": 1}
    return (X.iloc[:ntr].reset_index(drop=True), y[:ntr], X.iloc[ntr:].reset_index(drop=True), y[ntr:], p[ntr:],
            md, sp)


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score, log_loss
    for i, sp in enumerate(SPECS):
        Xtr, ytr, Xte, yte, pte, md, _ = make(i, n_test=5000)
        print(f"{i:2d} {sp['name']:14s} train={Xtr.shape} rate={ytr.mean():.3f} bayesAUC={roc_auc_score(yte, pte):.3f} "
              f"bayesLL={log_loss(yte, pte):.4f} miss={Xtr.isna().mean().mean():.3f}")
