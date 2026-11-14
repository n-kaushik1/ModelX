"""Worst-case fit time: 60k rows (the largest size tuned on all rows), 100 numerics, 4 categoricals and a
200-level categorical, decision cost 1:10.  usage: python time_worst.py <solution.py> <n_jobs>"""
import sys, os, time, importlib.util
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v6"))
import simgen
simgen.SPECS4.append(dict(name="worst_case", n=60000, rate=0.08, obj="decision_cost", cost=(1, 10), nn=100, nc=4,
                          sd=1.6, ni=2, informative=30, highcard=200))
spec = importlib.util.spec_from_file_location("solution", sys.argv[1])
mod = importlib.util.module_from_spec(spec); sys.modules["solution"] = mod; spec.loader.exec_module(mod)
Xtr, ytr, Xte, yte, pte, md, sp = simgen.make(3000 + len(simgen.SPECS4) - 1, n_test=5000)
print(sp["name"], Xtr.shape, flush=True)
t0 = time.time()
sol = mod.StudentSolution(n_jobs=int(sys.argv[2]), verbose=True).fit(Xtr, ytr, dict(md))
print("TOTAL FIT SECONDS", round(time.time() - t0), sol.report_.get("final_mode"), sol.report_.get("blend_weights"), flush=True)
