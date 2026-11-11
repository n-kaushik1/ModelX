"""Fit a framework on one synthetic problem, joblib round trip, score a 20k-row test set against the truth.
usage: python sim_run.py <solution.py> <tag> <pid> <n_jobs>   -> logs/sim/<tag>_<pid>.json + logs/sim.log"""
import sys, os, time, json, tempfile, importlib.util
import numpy as np, joblib
from sklearn.metrics import roc_auc_score, log_loss
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import simgen

sol_path, tag, pid, nj = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
spec = importlib.util.spec_from_file_location("solution", sol_path)
mod = importlib.util.module_from_spec(spec); sys.modules["solution"] = mod; spec.loader.exec_module(mod)
Xtr, ytr, Xte, yte, pte, md, sp = simgen.make(pid)
t0 = time.time()
sol = mod.StudentSolution(n_jobs=nj).fit(Xtr, ytr, dict(md))
fit_s = time.time() - t0
path = os.path.join(tempfile.mkdtemp(), "m.joblib")
joblib.dump(sol, path, compress=3)
size = os.path.getsize(path) / 1e6
sol = joblib.load(path)
p = np.clip(np.asarray(sol.predict_proba(Xte, dict(md)), float), 1e-15, 1 - 1e-15)
t = float(sol.get_threshold())


def ncost(pp, thr):
    cfp, cfn = md["cost_false_positive"], md["cost_false_negative"]
    a = pp >= thr
    return (cfp * np.sum(a & (yte == 0)) + cfn * np.sum(~a & (yte == 1))) / min(cfn * yte.sum(), cfp * (yte == 0).sum())


res = dict(tag=tag, pid=pid, name=sp["name"], obj=md["objective"], fit_s=round(fit_s, 1), size_mb=round(size, 2),
           auc=roc_auc_score(yte, p), ll=log_loss(yte, p), bayes_auc=roc_auc_score(yte, pte), bayes_ll=log_loss(yte, pte),
           blend=getattr(sol, "report_", {}).get("blend_weights"))
if md["objective"] == "decision_cost":
    bt = md["cost_false_positive"] / (md["cost_false_positive"] + md["cost_false_negative"])
    res.update(thr=t, cost=ncost(p, t), cost_at_bayes_thr=ncost(p, bt), bayes_cost=ncost(pte, bt))
res["regret"] = {"roc_auc": res["bayes_auc"] - res["auc"], "log_loss": res["ll"] - res["bayes_ll"],
                 "decision_cost": res.get("cost", 0) - res.get("bayes_cost", 0)}[md["objective"]]
os.makedirs(os.path.join(HERE, "logs", "sim"), exist_ok=True)
json.dump(res, open(os.path.join(HERE, "logs", "sim", f"{tag}_{pid}.json"), "w"), default=str)
line = (f"{tag:10s} {pid:2d} {sp['name']:14s} {md['objective']:13s} regret={res['regret']:.5f} auc={res['auc']:.4f} "
        f"ll={res['ll']:.4f}" + (f" cost={res['cost']:.4f} thr={t:.3f} costBayesThr={res['cost_at_bayes_thr']:.4f}" if 'cost' in res else "")
        + f" fit={fit_s:.0f}s size={size:.1f}MB")
print(line, flush=True)
with open(os.path.join(HERE, "logs", "sim.log"), "a") as f:
    f.write(line + "\n")
