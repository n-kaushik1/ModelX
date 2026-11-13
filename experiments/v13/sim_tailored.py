"""Threshold-tailored blend selection on synthetic cost problems, scored against the TRUE probabilities.
For each saved library (experiments/v6/logs/simlib/<tag>_<pid>.npz): bagged Caruana on the training OOF logits
with log loss vs beta scoring rules centred on the Bayes threshold, blend applied to the 20k test rows, then
  expected cost  = mean over test rows of the cost of the decision under the true probability (no label noise),
  normalised by the expected cost of the better trivial strategy; regret = minus the same for the true p.
usage: python sim_tailored.py <tag> <pid> [<pid> ...]"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv, args = sys.argv[:2] + ["p3"], sys.argv
import p3_tailored as T

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v6", "logs", "simlib")
tag, pids = args[1], [int(a) for a in args[2:]]
tot = {}
for pid in pids:
    f = os.path.join(D, f"{tag}_{pid}.npz")
    if not os.path.exists(f):
        continue
    z = np.load(f, allow_pickle=True)
    cfp, cfn = float(z["cfp"]), float(z["cfn"])
    t = cfp / (cfp + cfn)
    pte = z["pte"]
    base = min(cfn * pte.sum(), cfp * (1 - pte).sum())

    def ecost(p):
        a = p >= t
        return float(np.sum(np.where(a, cfp * (1 - pte), cfn * pte))) / base

    S = {"logloss": T.S["logloss"]}
    for conc in (12, 24):
        a_, b_ = 1 + t * conc, 1 + (1 - t) * conc
        S[f"beta{conc}"] = T.beta_loss(a_, b_)
    bayes = ecost(pte)
    line = []
    for k, sc in S.items():
        w = T.bagged(sc, z["ytr"], z["Ltr"])
        p = T.sigmoid(w @ z["Lte"])
        r = ecost(p) - bayes
        kl = float(np.mean(pte * np.log(pte / np.clip(p, 1e-9, 1)) + (1 - pte) * np.log((1 - pte) / np.clip(1 - p, 1e-9, 1))))
        tot.setdefault(k, []).append(r)
        line.append(f"{k}: regret {r:.5f} KL {kl:.5f}")
    print(f"{pid:5d} t={t:.3f} " + "  ".join(line), flush=True)
for k, v in tot.items():
    d = np.array(v) - np.array(tot["logloss"])
    print(f"{k:8s} mean expected-cost regret {np.mean(v):.5f}  vs logloss {d.mean():+.5f} (se {d.std() / np.sqrt(len(d)):.5f}, "
          f"better on {np.sum(d < 0)}/{len(d)})")
