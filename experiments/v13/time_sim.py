"""Verbose timed fit of a framework on one synthetic problem.  usage: python time_sim.py <solution.py> <pid> <n_jobs>"""
import sys, os, time, importlib.util
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v6"))
import simgen
spec = importlib.util.spec_from_file_location("solution", sys.argv[1])
mod = importlib.util.module_from_spec(spec); sys.modules["solution"] = mod; spec.loader.exec_module(mod)
Xtr, ytr, Xte, yte, pte, md, sp = simgen.make(int(sys.argv[2]))
t0 = time.time()
sol = mod.StudentSolution(n_jobs=int(sys.argv[3]), verbose=True).fit(Xtr, ytr, dict(md))
print("TOTAL FIT SECONDS", time.time() - t0, sol.report_.get("final_mode"), sol.report_.get("blend_weights"), flush=True)
