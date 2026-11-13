"""Repeated half-splits inside each outer fold's training-side OOF library: choose the blend on one half with
log loss or a threshold-tailored beta scoring rule, score the cost at the Bayes threshold on the other half.
usage: python p3_halfsplit.py <tag> [reps]"""
import sys, os
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
tag = sys.argv[1]
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 6
sys.argv = [sys.argv[0], tag, "p3"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3_tailored as T

S = {k: T.S[k] for k in ("logloss", "beta12", "beta24")}
out = {k: [] for k in S}
for seed in (0, 1):
    for f in range(5):
        z = np.load(os.path.join(T.OUT, f"{tag}_p3_s{seed}_f{f}.npz"), allow_pickle=True)
        yy, L = T.y[z["tr"]], z["Ltr"]
        for a, b in StratifiedShuffleSplit(reps, test_size=0.5, random_state=100 * seed + f).split(L.T, yy):
            for k, sc in S.items():
                w = T.bagged(sc, yy[a], L[:, a])
                p = T.sigmoid(w @ L[:, b])
                out[k].append(T.cost_hard(yy[b], p) * len(b) / min(T.cfn * yy[b].sum(), T.cfp * (yy[b] == 0).sum()))
        d = np.array(out["beta12"]) - np.array(out["logloss"])
        print(f"s{seed} f{f}: n={len(d)} beta12-logloss mean {d.mean():+.4f} se {d.std() / np.sqrt(len(d)):.4f} "
              f"wins {np.mean(d < 0):.2f} ties {np.mean(d == 0):.2f}", flush=True)
for k in S:
    print(k, f"mean cost {np.mean(out[k]):.4f}")
for k in ("beta12", "beta24"):
    d = np.array(out[k]) - np.array(out["logloss"])
    print(f"{k} - logloss: mean {d.mean():+.4f} se {d.std() / np.sqrt(len(d)):.4f} better {np.mean(d < 0):.2f} worse {np.mean(d > 0):.2f}")
