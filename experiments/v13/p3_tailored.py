"""Blend selection for a decision-cost objective with a proper scoring rule tailored to the decision threshold
(Buja, Stuetzle & Shen 2005, beta family: weight on thresholds c ~ c^(a-1) (1-c)^(b-1), mode at the Bayes
threshold) instead of log loss.  Offline on the saved outer-CV fold libraries of a framework tag: the bagged
Caruana selection is run on each fold's training-side OOF logits, the blend is applied to the held-out fold
and scored there (cost at the Bayes threshold, pooled over the 5 folds of a seed).
usage: python p3_tailored.py <tag> [key=p3] [cfp=1] [cfn=10]"""
import sys, os
import numpy as np
from scipy.special import betainc, beta as B
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v4"))
from kit import load, ll, sigmoid

tag = sys.argv[1]
key = sys.argv[2] if len(sys.argv) > 2 else "p3"
cfp, cfn = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (1.0, 10.0)
t_b = cfp / (cfp + cfn)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v5", "logs", "outer")
X, y, md = load(key)


def beta_loss(a, b):
    """mean loss of the beta-family proper scoring rule; a, b -> 0 is log loss (up to scale)."""
    def f(yy, q):
        q = np.clip(q, 1e-9, 1 - 1e-9)
        l1 = B(a, b + 1) * (1 - betainc(a, b + 1, q))        # integral_q^1 c^(a-1) (1-c)^b dc
        l0 = B(a + 1, b) * betainc(a + 1, b, q)              # integral_0^q c^a (1-c)^(b-1) dc
        return float(np.mean(np.where(yy == 1, l1, l0)))
    return f


def cost_hard(yy, q):
    return float(np.mean(np.where(yy == 1, cfn * (q < t_b), cfp * (q >= t_b))))


def make_scores():
    S = {"logloss": lambda yy, q: ll(yy, q)}
    for conc in (6, 12, 24, 48):                             # a + b - 2 = conc, mode (a-1)/(a+b-2) = t_b
        a = 1 + t_b * conc
        b = 1 + (1 - t_b) * conc
        S[f"beta{conc}"] = beta_loss(a, b)
    S["hardcost"] = cost_hard
    return S


def caruana(score, yy, L, max_steps=40):
    counts = np.zeros(len(L))
    singles = [score(yy, sigmoid(L[i])) for i in range(len(L))]
    i0 = int(np.argmin(singles)); counts[i0] = 1; cur = singles[i0]
    for _ in range(max_steps - 1):
        sc = [score(yy, sigmoid(((counts + np.eye(len(L))[i]) / (counts.sum() + 1)) @ L)) for i in range(len(L))]
        j = int(np.argmin(sc))
        if sc[j] < cur - 1e-9:
            cur = sc[j]; counts[j] += 1
        else:
            break
    return counts / counts.sum()


def bagged(score, yy, L, n_bags=25, min_w=0.02, seed=42):
    rng = np.random.default_rng([seed, 2006])
    w = np.zeros(len(L)); done = 0
    for _ in range(n_bags):
        idx = rng.integers(0, len(yy), len(yy))
        if len(np.unique(yy[idx])) < 2:
            continue
        w += caruana(score, yy[idx], L[:, idx]); done += 1
    w /= max(done, 1); w[w < min_w] = 0
    return w / w.sum()


S = make_scores()
res = {k: [] for k in S}
for seed in ((0, 1) if __name__ == "__main__" else ()):
    agg = {k: [0.0, 0.0, [], []] for k in S}                # FP, FN, p, y
    for f in range(5):
        z = np.load(os.path.join(OUT, f"{tag}_{key}_s{seed}_f{f}.npz"), allow_pickle=True)
        tr, va, Ltr, Lte = z["tr"], z["va"], z["Ltr"], z["Lte"]
        for k, sc in S.items():
            w = bagged(sc, y[tr], Ltr)
            p = sigmoid(w @ Lte)
            a = p >= t_b
            agg[k][0] += np.sum(a & (y[va] == 0)); agg[k][1] += np.sum(~a & (y[va] == 1))
            agg[k][2].append(p); agg[k][3].append(y[va])
    base = min(cfn * y.sum(), cfp * (y == 0).sum())
    line = []
    for k, (fp, fn, ps, ys) in agg.items():
        c = (cfp * fp + cfn * fn) / base
        res[k].append(c)
        line.append(f"{k}: cost {c:.4f} LL {ll(np.concatenate(ys), np.concatenate(ps)):.5f} (FP {int(fp)} FN {int(fn)})")
    print(f"{tag} {key} seed {seed}\n   " + "\n   ".join(line), flush=True)
if __name__ == "__main__":
    print("mean over seeds: " + "  ".join(f"{k}={np.mean(v):.4f}" for k, v in res.items()))
