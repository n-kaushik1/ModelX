"""Outer 5-fold CV of a whole framework (fit on 4/5 -> joblib save/load -> predict raw 1/5).
usage: python eval_outer.py <solution.py> <tag> <key> [n_jobs] [fold_seed] [folds e.g. 0,1,2,3,4]
Saves OOF probabilities + per-fold thresholds to logs/outer_<tag>_<key>_s<seed>.npz and appends a line to
logs/outer.log.  Pooled metric: P1 AUC over all OOF rows, P2 log loss, P3 normalized cost where each fold uses
its own learned threshold (costs summed over folds, normalised by the pooled trivial baseline)."""
import sys, os, time, json, tempfile
import numpy as np, joblib
from kit import *

sol_path, tag, key = sys.argv[1], sys.argv[2], sys.argv[3]
nj = int(sys.argv[4]) if len(sys.argv) > 4 else 16
fseed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
which = [int(i) for i in sys.argv[6].split(",")] if len(sys.argv) > 6 else [0, 1, 2, 3, 4]
mod = load_module(sol_path, "solution")
sys.modules["solution"] = mod
X, y, md = load(key)
out_f = f"{WORK}/logs/outer_{tag}_{key}_s{fseed}.npz"
prev = dict(np.load(out_f, allow_pickle=True)) if os.path.exists(out_f) else {}
oof = prev.get("oof", np.full(len(y), np.nan))
thr = prev.get("thr", np.full(len(y), np.nan))
info = list(prev.get("info", []))
log = Log("outer.log")
for i, (tr, va) in enumerate(folds(y, fseed)):
    if i not in which:
        continue
    t0 = time.time()
    try:
        sol = mod.StudentSolution(n_jobs=nj, keep_oof=True)
    except TypeError:
        sol = mod.StudentSolution(n_jobs=nj)
    sol.fit(X.iloc[tr].reset_index(drop=True), y[tr], dict(md))
    if getattr(sol, "library_", None) is not None:                # per-member logits on the outer test rows
        Xte = sol.cleaner_.transform(X.iloc[va].reset_index(drop=True))
        keep = [m for m in sol.library_ if m["models"]]
        Lte = np.vstack([np.mean([mod._logit(fm.predict(Xte)) for fm in m["models"]], axis=0) for m in keep])
        Ltr = np.vstack([m["Z"].mean(0) for m in keep])
        np.savez(f"{WORK}/logs/lib_{tag}_{key}_s{fseed}_f{i}.npz", Ltr=Ltr, Lte=Lte, tr=tr, va=va,
                 names=np.array([m["spec"].label() for m in keep], dtype=object))
        del sol.library_, sol.oof_library_
    path = os.path.join(tempfile.mkdtemp(), "m.joblib")
    joblib.dump(sol, path, compress=3)
    sol = joblib.load(path)
    p = np.asarray(sol.predict_proba(X.iloc[va].reset_index(drop=True), dict(md)), float)
    p = p[:, 1] if p.ndim == 2 else p
    oof[va] = p
    t = float(sol.get_threshold())
    thr[va] = t
    r = getattr(sol, "report_", {})
    blend = r.get("blend_weights", {})
    fs = {"p1": f"AUC={auc(y[va], p):.5f}", "p2": f"LL={ll(y[va], p):.5f}",
          "p3": f"cost={norm_cost(y[va], p, t):.4f} t={t:.4f} LL={ll(y[va], p):.5f}"}[key]
    line = (f"{tag} {key} s{fseed} fold{i} {fs} fit={time.time()-t0:.0f}s blend={json.dumps(blend)[:300]} "
            f"oofLL={r.get('blend_oof_logloss', float('nan')):.5f}")
    log(line)
    info.append(line)
    np.savez(out_f, oof=oof, thr=thr, info=np.array(info, dtype=object))
if not np.isnan(oof).any():
    if key == "p1":
        s = f"pooled AUC={auc(y, oof):.5f} mean-fold AUC={np.mean([auc(y[va], oof[va]) for _, va in folds(y, fseed)]):.5f}"
    elif key == "p2":
        s = f"pooled LL={ll(y, oof):.5f}"
    else:
        a = oof >= thr
        cost = np.sum(a & (y == 0)) + 10 * np.sum(~a & (y == 1))
        s = (f"pooled normcost={cost / min(10 * y.sum(), (y == 0).sum()):.4f} LL={ll(y, oof):.5f} "
             f"mean-fold cost={np.mean([norm_cost(y[va], oof[va], thr[va][0]) for _, va in folds(y, fseed)]):.4f}")
    log(f"### {tag} {key} s{fseed} {s}")
