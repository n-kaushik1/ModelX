"""Honest ensemble study on the zoo OOF predictions (logs/zoo_<key>_<name>.npy, 2 fold seeds each).
40 repeats: rows split 50/50 (stratified); the combiner (bagged Caruana + Platt, as in the framework) is fitted on
half A with fold-seed-0 OOF and scored on half B with fold-seed-1 OOF.  Reports the framework-like library, the
library plus each new candidate (marginal value), and everything.
usage: python blend_zoo.py <key> [extra libraries as name+name+...]"""
import sys, os, glob
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "v4"))
from kit import load, ll, auc, sigmoid, logit, norm_cost

key = sys.argv[1]
y = load(key)[1]
O = {os.path.basename(f)[len(f"zoo_{key}_"):-4]: np.load(f) for f in sorted(glob.glob(os.path.join(HERE, "logs", f"zoo_{key}_*.npy")))}
FW = [n for n in ["lgb_B", "lgb_d3k16m", "lgb_d2", "lgb_d4", "xgb_d2k8m", "xgb_d3", "cat_d3m", "cat_d4", "gam_c05", "gam_c2", "et_l5"] if n in O]
rank = key == "p1"
Z = {n: np.vstack([logit(O[n][s]) for s in (0, 1)]) for n in O}


def crit(yy, z):
    return auc(yy, z) if rank else -ll(yy, sigmoid(z))


def caruana(L, yy, steps=40):
    c = np.zeros(len(L)); s1 = [crit(yy, L[i]) for i in range(len(L))]
    c[int(np.argmax(s1))] = 1; cur = max(s1)
    for _ in range(steps - 1):
        cand = [crit(yy, ((c + np.eye(len(L))[i]) / (c.sum() + 1)) @ L) for i in range(len(L))]
        j = int(np.argmax(cand))
        if cand[j] <= cur + 1e-7: break
        cur, c[j] = cand[j], c[j] + 1
    return c / c.sum()


def bagged(L, yy, rng, nb=10):
    w = np.zeros(len(L))
    for _ in range(nb):
        i = rng.integers(0, len(yy), len(yy)); w += caruana(L[:, i], yy[i])
    w /= nb; w[w < 0.02] = 0
    return w / w.sum()


splits = list(StratifiedShuffleSplit(40, test_size=0.5, random_state=0).split(np.zeros(len(y)), y))


def evaluate(lib):
    res = {"main": [], "ll": [], "cost": []}
    for r, (A, B) in enumerate(splits):
        LA = np.vstack([Z[n][0][A] for n in lib]); LB = np.vstack([Z[n][1][B] for n in lib])
        w = bagged(LA, y[A], np.random.default_rng(r)) if len(lib) > 1 else np.ones(1)
        pl = LogisticRegression(C=1e6, max_iter=1000).fit((w @ LA)[:, None], y[A])
        pB = pl.predict_proba((w @ LB)[:, None])[:, 1]
        res["main"].append(auc(y[B], pB) if rank else ll(y[B], pB)); res["ll"].append(ll(y[B], pB))
        if key == "p3":
            res["cost"].append(norm_cost(y[B], pB, 1 / 11))
    return {k: np.array(v) for k, v in res.items()}


def show(tag, r, base=None):
    s = f"{tag:34s} {'AUC' if rank else 'LL'}={r['main'].mean():.5f}"
    if base is not None:
        d = r["main"] - base["main"]; s += f" d={d.mean():+.5f} (se {d.std() / np.sqrt(len(d)):.5f})"
    if key == "p3":
        s += f" cost@1/11={r['cost'].mean():.4f}" + (f" d={np.mean(r['cost'] - base['cost']):+.4f} (se {np.std(r['cost'] - base['cost']) / np.sqrt(len(r['cost'])):.4f})" if base is not None else "")
    print(s, flush=True)


QUICK = os.environ.get("QUICK")
if not QUICK:
    print(f"=== {key}: singles (scored with Platt on half A)")
    for n in sorted(O):
        show(n, evaluate([n]))
base = evaluate(FW); print(f"=== framework-like library ({len(FW)}): {FW}"); show("FW", base)
if not QUICK:
    for n in sorted(set(O) - set(FW)):
        show(f"FW + {n}", evaluate(FW + [n]), base)
    show("ALL", evaluate(sorted(O)), base)
for extra in sys.argv[2:]:
    show(f"FW + {extra}", evaluate(FW + extra.split("+")), base)
