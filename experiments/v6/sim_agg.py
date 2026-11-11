"""Compare tags on the synthetic suite: python sim_agg.py <tagA> <tagB> ...  (regret: lower is better)"""
import sys, os, glob, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
tags = sys.argv[1:]
R = {}
for f in glob.glob(os.path.join(HERE, "logs", "sim", "*.json")):
    r = json.load(open(f))
    R[(r["tag"], r["pid"])] = r
pids = sorted({p for (t, p) in R if t in tags})
print(f"{'pid':>3s} {'problem':14s} {'objective':13s} " + " ".join(f"{t:>12s}" for t in tags))
tot = {t: [] for t in tags}
for p in pids:
    rows = [R.get((t, p)) for t in tags]
    name = next(r for r in rows if r)["name"]; obj = next(r for r in rows if r)["obj"]
    cells = []
    for t, r in zip(tags, rows):
        cells.append(f"{r['regret']:12.5f}" if r else f"{'-':>12s}")
        if r:
            tot[t].append((p, r["regret"]))
    print(f"{p:3d} {name:14s} {obj:13s} " + " ".join(cells))
common = set.intersection(*[{p for p, _ in v} for v in tot.values()]) if tags else set()
for obj in ("roc_auc", "log_loss", "decision_cost"):
    ps = [p for p in common if R[(tags[0], p)]["obj"] == obj]
    if ps:
        print(f"mean regret {obj:13s} (n={len(ps)}): " + " ".join(f"{t}={np.mean([R[(t, p)]['regret'] for p in ps]):.5f}" for t in tags))
for t in tags:
    print(f"{t}: total fit {sum(R[(t, p)]['fit_s'] for p in common):.0f}s, max size {max(R[(t, p)]['size_mb'] for p in common):.1f}MB")
