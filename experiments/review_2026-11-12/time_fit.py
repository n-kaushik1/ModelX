import sys, time, os, json
sys.path.insert(0, "submission/v12")
import pandas as pd, numpy as np, joblib
from solution import StudentSolution
pid, nj = sys.argv[1], int(sys.argv[2])
files = {"p1": ("p1_manufacturing_development.csv", "quality_failure", "roc_auc"),
         "p2": ("p2_credit_development.csv", "serious_delinquency", "log_loss"),
         "p3": ("p3_telecom_development.csv", "major_service_disruption", "decision_cost")}
f, t, objn = files[pid]
df = pd.read_csv("data/" + f)
X = df.drop(columns=[t, "record_id"]); y = df[t].astype(int)
cost = objn == "decision_cost"
md = {"objective": objn, "greater_is_better": objn == "roc_auc", "requires_threshold": cost,
      "cost_false_positive": 1 if cost else None, "cost_false_negative": 10 if cost else None, "positive_label": 1}
t0 = time.time()
sol = StudentSolution(verbose=True, n_jobs=nj, keep_oof=True).fit(X, y, md)
print("TOTAL FIT SECONDS", time.time() - t0, "n_jobs", nj)
print(json.dumps({k: v for k, v in sol.summary().items() if k not in ("search",)}, default=str, indent=1))
joblib.dump({"L": sol.oof_library_["L"], "y": sol.oof_library_["y"], "names": sol.oof_library_["names"],
             "threshold": sol.threshold_, "weights": sol.report_.get("blend_weights")}, f"{os.environ['S']}/oof_{pid}_nj{nj}.joblib")
