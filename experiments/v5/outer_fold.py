"""One outer fold of a whole framework: fit on 4/5 -> joblib save/load -> predict the raw 1/5.
usage: python outer_fold.py <solution.py> <tag> <key> <n_jobs> <fold_seed> <fold>
Writes logs/outer/<tag>_<key>_s<seed>_f<fold>.npz (test probabilities, threshold, report, library logits)
so folds can run in parallel; aggregate with outer_agg.py."""
import sys, os, time, json, tempfile
import numpy as np, joblib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v4"))
from kit import load, load_module, folds, auc, ll, norm_cost

sol_path, tag, key = sys.argv[1], sys.argv[2], sys.argv[3]
nj, fseed, fold = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "outer")
os.makedirs(OUT, exist_ok=True)
mod = load_module(sol_path, "solution")
X, y, md = load(key)
tr, va = folds(y, fseed)[fold]
t0 = time.time()
try:
    sol = mod.StudentSolution(n_jobs=nj, keep_oof=True)
except TypeError:
    sol = mod.StudentSolution(n_jobs=nj)
sol.fit(X.iloc[tr].reset_index(drop=True), y[tr], dict(md))
fit_s = time.time() - t0
extra = {}
if getattr(sol, "library_", None) is not None:
    Xte = sol.cleaner_.transform(X.iloc[va].reset_index(drop=True))
    keep = [m for m in sol.library_ if m["models"]]
    if keep:
        extra["Lte"] = np.vstack([np.mean([mod._logit(fm.predict(Xte)) for fm in m["models"]], axis=0) for m in keep])
        extra["Ltr"] = np.vstack([m["Z"].mean(0) for m in keep])
        extra["names"] = np.array([m["spec"].label() for m in keep], dtype=object)
    del sol.library_
    if hasattr(sol, "oof_library_"):
        del sol.oof_library_
path = os.path.join(tempfile.mkdtemp(), "m.joblib")
joblib.dump(sol, path, compress=3)
size_mb = os.path.getsize(path) / 1e6
sol = joblib.load(path)
p = np.asarray(sol.predict_proba(X.iloc[va].reset_index(drop=True), dict(md)), float)
p = p[:, 1] if p.ndim == 2 else p
t = float(sol.get_threshold())
r = getattr(sol, "report_", {})
fs = {"p1": f"AUC={auc(y[va], p):.5f}", "p2": f"LL={ll(y[va], p):.5f}",
      "p3": f"cost={norm_cost(y[va], p, t):.4f} t={t:.4f} LL={ll(y[va], p):.5f}"}[key]
line = (f"{tag} {key} s{fseed} f{fold} {fs} fit={fit_s:.0f}s size={size_mb:.1f}MB "
        f"comb={r.get('combiner', '-')} blend={json.dumps(r.get('blend_weights', {}))[:300]}")
print(line, flush=True)
np.savez(os.path.join(OUT, f"{tag}_{key}_s{fseed}_f{fold}.npz"), p=p, va=va, tr=tr, thr=t, line=line,
         report=json.dumps({k: v for k, v in r.items() if k != "search"}, default=str), **extra)
with open(os.path.join(OUT, "..", "outer.log"), "a") as f:
    f.write(line + "\n")
