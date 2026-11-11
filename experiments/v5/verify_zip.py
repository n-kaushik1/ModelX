"""Load a submission zip in a fresh dir (as the leaderboard does), predict 500 dev rows per problem, print the
fitted report and save the predictions.  usage: python verify_zip.py <zip> <data_dir> <out.npz>"""
import sys, os, json, zipfile, tempfile
import numpy as np, pandas as pd, joblib
zp, data, out = sys.argv[1], sys.argv[2], sys.argv[3]
d = tempfile.mkdtemp()
zipfile.ZipFile(zp).extractall(d)
sys.path.insert(0, d)
import solution
print("files:", sorted(os.listdir(d)))
print("manifest:", open(os.path.join(d, "manifest.json")).read().replace("\n", " ")[:400])
print("threshold:", open(os.path.join(d, "p3_threshold.json")).read())
res = {}
for key, f, t in [("p1", "p1_manufacturing_development.csv", "quality_failure"), ("p2", "p2_credit_development.csv", "serious_delinquency"),
                  ("p3", "p3_telecom_development.csv", "major_service_disruption")]:
    m = joblib.load(os.path.join(d, f"{key}_model.joblib"))
    X = pd.read_csv(os.path.join(data, f)).sample(500, random_state=0).drop(columns=[t, "record_id"])
    p = np.asarray(m.predict_proba(X, {}), float)
    res[key] = p
    r = m.report_
    n_search = {}
    for fam, *_ in r.get("search", []):
        n_search[fam] = n_search.get(fam, 0) + 1
    print(f"{key}: searched={n_search} combiner={r.get('combiner')} weights={r.get('blend_weights')} "
          f"calibrated={m.calibrator_ is not None} oofLL={r.get('blend_oof_logloss'):.5f} oof_{m.objective_.metric}={r.get('blend_oof_score'):.5f} "
          f"fit={r.get('fit_seconds'):.0f}s thr={m.get_threshold():.4f} members={len(m.members_)} p=[{p.min():.4f},{p.mean():.4f},{p.max():.4f}]")
    if key == "p3":
        print("   p3 oof cost/1000:", json.dumps(r.get("oof_cost_per_1000")), "bayes", r.get("bayes_threshold"))
np.savez(out, **res)
