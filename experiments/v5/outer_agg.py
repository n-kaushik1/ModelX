"""Aggregate outer folds: python outer_agg.py <tag> [<tag> ...]  (all keys / seeds found)."""
import sys, os, glob, re
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v4"))
from kit import load, folds, auc, ll, norm_cost

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "outer")
Y = {}
for tag in sys.argv[1:]:
    for key in ("p1", "p2", "p3"):
        for seed in sorted({int(re.search(r"_s(\d+)_f", f).group(1)) for f in glob.glob(f"{OUT}/{tag}_{key}_s*_f*.npz")}):
            fs = sorted(glob.glob(f"{OUT}/{tag}_{key}_s{seed}_f*.npz"))
            if key not in Y:
                Y[key] = load(key)[1]
            y = Y[key]
            oof, thr = np.full(len(y), np.nan), np.full(len(y), np.nan)
            per = []
            for f in fs:
                d = np.load(f, allow_pickle=True)
                oof[d["va"]] = d["p"]; thr[d["va"]] = float(d["thr"])
                yv, pv = y[d["va"]], d["p"]
                per.append(auc(yv, pv) if key == "p1" else ll(yv, pv) if key == "p2" else norm_cost(yv, pv, float(d["thr"])))
            done = ~np.isnan(oof)
            yy, pp, tt = y[done], oof[done], thr[done]
            if key == "p1":
                s = f"pooledAUC={auc(yy, pp):.5f}"
            elif key == "p2":
                s = f"pooledLL={ll(yy, pp):.5f}"
            else:
                a = pp >= tt
                cost = np.sum(a & (yy == 0)) + 10 * np.sum(~a & (yy == 1))
                s = f"pooledCost={cost / min(10 * yy.sum(), (yy == 0).sum()):.4f} LL={ll(yy, pp):.5f}"
            print(f"{tag:12s} {key} s{seed} folds={len(fs)} {s} meanfold={np.mean(per):.5f} per=[{' '.join(f'{v:.4f}' for v in per)}]")
