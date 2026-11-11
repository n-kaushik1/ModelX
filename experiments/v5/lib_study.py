"""Offline holdout analysis on the saved outer-fold libraries (outer_fold.py with keep_oof).
Per outer fold: Ltr (members x n_train OOF logits of the k-fold re-scoring), Lte (members x n_test logits of
the k-fold bag).  Compares combiners, calibrators and P3 threshold rules on the outer-test rows.
usage: python lib_study.py <tag> <key> [seed]"""
import sys, os, glob
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v4"))
from kit import load, folds, auc, ll, norm_cost, sigmoid, logit

HERE = os.path.dirname(os.path.abspath(__file__))
tag, key = sys.argv[1], sys.argv[2]
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
X, y, md = load(key)
files = sorted(glob.glob(f"{HERE}/logs/outer/{tag}_{key}_s{seed}_f*.npz"))
rank = key == "p1"


def bscore(yy, z):
    return auc(yy, z) if rank else -ll(yy, sigmoid(z))


def caruana(L, yy, steps=40, crit=None, init_all=False):
    crit = crit or bscore
    c = np.zeros(len(L))
    if init_all:                      # start from the equal-weight average of every member
        c[:] = 1
        cur = crit(yy, L.mean(0))
    else:
        s1 = [crit(yy, L[i]) for i in range(len(L))]
        c[int(np.argmax(s1))] = 1
        cur = max(s1)
    for _ in range(steps - 1):
        cand = [crit(yy, ((c + np.eye(len(L))[i]) / (c.sum() + 1)) @ L) for i in range(len(L))]
        j = int(np.argmax(cand))
        if cand[j] <= cur + 1e-7:
            break
        cur, c[j] = cand[j], c[j] + 1
    return c / c.sum()


def platt(z, yy):
    m = LogisticRegression(C=1e6, max_iter=1000).fit(z[:, None], yy)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def nonneg_stack(L, yy, l2=1.0):
    """Logistic stack with non-negative member weights (free intercept), L-BFGS-B."""
    from scipy.optimize import minimize
    m = len(L)
    def f(w):
        z = w[:m] @ L + w[m]
        p = sigmoid(z)
        pc = np.clip(p, 1e-12, 1 - 1e-12)
        loss = -np.mean(yy * np.log(pc) + (1 - yy) * np.log(1 - pc)) + 0.5 * l2 * np.sum(w[:m] ** 2) / len(yy)
        g = (p - yy) / len(yy)
        return loss, np.r_[L @ g + l2 * w[:m] / len(yy), g.sum()]
    w0 = np.r_[np.full(m, 1.0 / m), 0.0]
    r = minimize(f, w0, jac=True, method="L-BFGS-B", bounds=[(0, None)] * m + [(None, None)])
    return r.x[:m], r.x[m]


def beta_cal(ztr, yy):
    """Beta calibration (Kull et al.): logistic on [log p, -log(1-p)]."""
    p = sigmoid(ztr)
    F = lambda p: np.c_[np.log(np.clip(p, 1e-12, 1)), -np.log(np.clip(1 - p, 1e-12, 1))]
    m = LogisticRegression(C=1e6, max_iter=1000).fit(F(p), yy)
    return lambda z: m.decision_function(F(sigmoid(z)))


def best_thr(yy, p, cfp=1, cfn=10, smooth=0):
    o = np.argsort(-p); ps, ys = p[o], yy[o]
    tp = np.cumsum(ys); fp = np.arange(1, len(ys) + 1) - tp
    cost = cfp * fp + cfn * (ys.sum() - tp)
    if smooth:
        k = np.ones(smooth) / smooth
        cost = np.convolve(np.pad(cost, (smooth // 2, smooth - 1 - smooth // 2), mode="edge"), k, mode="valid")
    i = int(np.argmin(cost))
    return 0.5 * (ps[i] + ps[min(i + 1, len(ps) - 1)])


def combiners(Ltr, ytr):
    """name -> function(Lte) fitted on (Ltr, ytr)."""
    out = {}
    w = caruana(Ltr, ytr); a, b = platt(w @ Ltr, ytr)
    out["caruana_platt"] = lambda L, w=w, a=a, b=b: a * (w @ L) + b
    ae, be = platt(Ltr.mean(0), ytr)
    out["equal_platt"] = lambda L, a=ae, b=be: a * L.mean(0) + b
    for C in (1.0, 0.01):
        st = LogisticRegression(C=C, max_iter=3000).fit(Ltr.T, ytr)
        out[f"stack_C{C:g}"] = lambda L, st=st: st.decision_function(L.T)
    return out


def auto_select(Ltr, ytr, pool, prefer="equal_platt", margin=0.0):
    zs = {k: np.zeros(len(ytr)) for k in pool}
    for itr, iva in folds(ytr, 303):
        cs = combiners(Ltr[:, itr], ytr[itr])
        for k in pool:
            zs[k][iva] = cs[k](Ltr[:, iva])
    sc = {k: bscore(ytr, z) for k, z in zs.items()}
    best = max(sc, key=sc.get)
    if prefer in sc and sc[best] - sc[prefer] <= margin:
        best = prefer
    return best, sc


S = {}
def put(name, va, z):
    S.setdefault(name, np.zeros(len(y)))[va] = z

thr_rules = {}
slopes = []
for f in files:
    d = np.load(f, allow_pickle=True)
    if "Ltr" not in d:
        continue
    Ltr, Lte, tr, va = d["Ltr"], d["Lte"], d["tr"], d["va"]
    ytr = y[tr]
    names = list(d["names"])
    s1 = [bscore(ytr, Ltr[i]) for i in range(len(Ltr))]
    put("best_single", va, Lte[int(np.argmax(s1))])
    w = caruana(Ltr, ytr)
    put("caruana_raw", va, w @ Lte)
    a, b = platt(w @ Ltr, ytr)
    put("caruana_platt", va, a * (w @ Lte) + b)
    bc = beta_cal(w @ Ltr, ytr)
    put("caruana_beta", va, bc(w @ Lte))
    wll = caruana(Ltr, ytr, crit=lambda yy, z: -ll(yy, sigmoid(z)))
    a2, b2 = platt(wll @ Ltr, ytr)
    put("caruanaLL_platt", va, a2 * (wll @ Lte) + b2)
    for C in (1.0, 0.1, 0.01):
        st = LogisticRegression(C=C, max_iter=3000).fit(Ltr.T, ytr)
        put(f"stack_C{C:g}", va, st.decision_function(Lte.T))
    wn, bn = nonneg_stack(Ltr, ytr)
    put("stack_nonneg", va, wn @ Lte + bn)
    put("equal", va, Lte.mean(0))
    wh = 0.5 * w + 0.5 / len(w)
    ah, bh = platt(wh @ Ltr, ytr); put("half_platt", va, ah * (wh @ Lte) + bh)
    wi = caruana(Ltr, ytr, init_all=True)
    ai, bi = platt(wi @ Ltr, ytr); put("carinit_platt", va, ai * (wi @ Lte) + bi)
    rngb = np.random.default_rng(1)
    wb = np.mean([caruana(Ltr[:, ib], ytr[ib]) for ib in [rngb.integers(0, len(ytr), len(ytr)) for _ in range(20)]], 0)
    ab, bb = platt(wb @ Ltr, ytr); put("carbag_platt", va, ab * (wb @ Lte) + bb)
    for nm, pool, pref, mg in [("auto_all", ["caruana_platt", "equal_platt", "stack_C1", "stack_C0.01"], None, 0),
                               ("auto_v4like", ["caruana_platt", "stack_C1"], None, 0),
                               ("auto_eq_car", ["caruana_platt", "equal_platt"], None, 0),
                               ("auto_pref_eq", ["caruana_platt", "equal_platt", "stack_C0.01"], "equal_platt", 2e-4)]:
        ch, sc = auto_select(Ltr, ytr, pool, pref, mg)
        put(nm, va, combiners(Ltr, ytr)[ch](Lte))
        print(f"   {nm}: chose {ch}  " + " ".join(f"{k}={v:.5f}" for k, v in sc.items()))
    ae, be = platt(Ltr.mean(0), ytr)
    put("equal_platt", va, ae * Lte.mean(0) + be)
    # how much would the test rows have wanted on top of the train-fitted Platt?
    zt = a * (w @ Lte) + b
    slopes.append(platt(zt, y[va])[0])
    print(f"fold {f[-6:-4]} members={len(names)} weights=" +
          ", ".join(f"{n.split('(')[0]}:{x:.2f}" for n, x in zip(names, w) if x > 0) + f" | test slope after Platt={slopes[-1]:.3f}")
    if key == "p3":
        pc = sigmoid(a * (w @ Ltr) + b)                      # calibrated OOF of the blend on the train part
        pt = sigmoid(zt)
        rng = np.random.default_rng(0)
        t_full = best_thr(ytr, pc, smooth=max(5, len(ytr) // 400))
        ts = [best_thr(ytr[i], pc[i], smooth=max(5, len(ytr) // 400)) for i in [rng.integers(0, len(ytr), len(ytr)) for _ in range(200)]]
        rules = {"learned(v4)": 0.5 * (np.median(ts) + t_full), "bayes_1/11": 1 / 11, "raw_opt": best_thr(ytr, pc),
                 "boot_median": float(np.median(ts)), "boot_mean": float(np.mean(ts))}
        for r, t in rules.items():
            thr_rules.setdefault(r, []).append((va, pt, t))
for s, z in S.items():
    msg = f"{tag} {key} {s:18s} " + (f"AUC={auc(y, z):.5f} " if rank else "") + f"LL={ll(y, sigmoid(z)):.5f}"
    if key == "p3":
        msg += f" cost@1/11={norm_cost(y, sigmoid(z), 1/11):.4f}"
    print(msg)
print(f"mean test slope needed after train-Platt: {np.mean(slopes):.3f}")
for r, lst in thr_rules.items():
    fp = sum(np.sum((p >= t) & (y[va] == 0)) for va, p, t in lst)
    fn = sum(np.sum((p < t) & (y[va] == 1)) for va, p, t in lst)
    print(f"P3 threshold rule {r:12s} pooled normcost={(fp + 10 * fn) / min(10 * y.sum(), (y == 0).sum()):.4f} "
          f"thr=[{' '.join(f'{t:.3f}' for _, _, t in lst)}]")
