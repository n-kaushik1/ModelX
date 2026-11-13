"""Per-problem regret of framework tags on the synthetic suite, grouped by problem type, plus log-loss regret
(lower-noise view for every objective).  usage: python cmp.py <tagA> <tagB> [...]"""
import sys, os, glob, json
import numpy as np
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v6", "logs", "sim")
tags = sys.argv[1:]
R = {}
for f in glob.glob(os.path.join(D, "*.json")):
    r = json.load(open(f))
    if r["tag"] in tags:
        R[(r["tag"], r["pid"])] = r
pids = sorted({p for (t, p) in R if all((u, p) in R for u in tags)})
print(f"{'pid':>5s} {'problem':15s} {'obj':9s} " + " ".join(f"{t:>9s}" for t in tags) + "   | LL regret " + " ".join(f"{t:>8s}" for t in tags))
wins = {t: 0 for t in tags}
agg = {}
for p in pids:
    rs = [R[(t, p)] for t in tags]
    reg = [r["regret"] for r in rs]
    llr = [r["ll"] - r["bayes_ll"] for r in rs]
    wins[tags[int(np.argmin(reg))]] += 1
    agg.setdefault(rs[0]["obj"], []).append(reg)
    print(f"{p:5d} {rs[0]['name']:15s} {rs[0]['obj'][:9]:9s} " + " ".join(f"{v:9.5f}" for v in reg)
          + "   |          " + " ".join(f"{v:8.5f}" for v in llr))
for o, v in agg.items():
    v = np.array(v)
    print(f"mean regret {o:13s} n={len(v):2d}: " + " ".join(f"{t}={m:.5f}" for t, m in zip(tags, v.mean(0)))
          + "   paired diff (B-A) mean %.5f sd/sqrt(n) %.5f" % ((v[:, 1] - v[:, 0]).mean(), (v[:, 1] - v[:, 0]).std() / np.sqrt(len(v))) if len(tags) > 1 else "")
print("wins:", wins)
