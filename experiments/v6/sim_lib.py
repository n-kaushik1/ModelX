"""Fit a framework on one synthetic problem and save its library for offline combiner studies:
OOF logits of every library member (training rows), the members' test logits (fold models averaged), the
training labels, the test labels and the true test probabilities.
usage: python sim_lib.py <solution.py> <tag> <pid> <n_jobs>   -> logs/simlib/<tag>_<pid>.npz"""
import sys, os, time, importlib.util
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import simgen

sol_path, tag, pid, nj = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
spec = importlib.util.spec_from_file_location("solution", sol_path)
mod = importlib.util.module_from_spec(spec); sys.modules["solution"] = mod; spec.loader.exec_module(mod)
Xtr, ytr, Xte, yte, pte, md, sp = simgen.make(pid)
t0 = time.time()
sol = mod.StudentSolution(n_jobs=nj, keep_oof=True).fit(Xtr, ytr, dict(md))
keep = [m for m in sol.library_ if m["models"]]
Xc = sol.cleaner_.transform(Xte)
Lte = np.vstack([np.mean([mod._logit(fm.predict(Xc)) for fm in m["models"]], axis=0) for m in keep])
Ltr = np.vstack([m["Z"].mean(0) for m in keep])
names = np.array([m["spec"].label() for m in keep], dtype=object)
os.makedirs(os.path.join(HERE, "logs", "simlib"), exist_ok=True)
np.savez(os.path.join(HERE, "logs", "simlib", f"{tag}_{pid}.npz"), Ltr=Ltr, Lte=Lte, names=names, ytr=ytr, yte=yte,
         pte=pte, cfp=md["cost_false_positive"] or 1, cfn=md["cost_false_negative"] or 1,
         p_final=np.asarray(sol.predict_proba(Xte, dict(md)), float), thr=float(sol.get_threshold()))
print(f"{tag} {pid} {sp['name']} members={len(keep)} fit={time.time() - t0:.0f}s", flush=True)
