"""Oracle test: how much would dropping noisy duplicate columns (copy = original + noise) or no-signal columns
gain?  Repeated 5-fold CV (3 fold seeds), structure detection nested.  usage: python exp_drop.py <key> <n_jobs>"""
import sys, os, time
import numpy as np, lightgbm as lgb
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v4"))
from kit import *

HERE = os.path.dirname(os.path.abspath(__file__))
S = load_module(os.path.join(HERE, "..", "..", "submission", "v5", "solution.py"), "sol_v7")
key, NJ = sys.argv[1], int(sys.argv[2])
X, y, md = load(key)
COPIES = {"p1": ["material_density_estimate", "temperature_sensor_2"],
          "p2": ["utilization_3m_average", "account_balance_last_month", "estimated_debt_burden", "revenue_per_employee"],
          "p3": ["packet_loss_5m", "utilization_15m_average", "latency_p95_ms"]}[key]
logf = open(os.path.join(HERE, "logs", f"exp_drop_{key}.log"), "a")
def log(s):
    print(s, flush=True); logf.write(s + "\n"); logf.flush()

B = dict(learning_rate=0.03, max_depth=2, num_leaves=4, min_child_samples=20, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.8, reg_lambda=1.0)


def run(Xd, model):
    cl = S.DataCleaner().fit(Xd)
    Xc = cl.transform(Xd)
    nums, cats = cl.numeric_, cl.categorical_
    scs, oofs = [], []
    for s in (0, 1, 2):
        oof = np.zeros(len(y))
        for tr, va in folds(y, s):
            Xtr, Xva = Xc.iloc[tr], Xc.iloc[va]
            if model == "lgbB":
                enc = S.TabularEncoder("native").fit(Xtr, nums, cats)
                st = S._detect_structure(Xtr, y[tr], nums, cats, NJ)
                inter, mon = S._constraints_for({"inter_k": 8, "mono": 1}, st, len(enc.feature_names_))
                m = lgb.LGBMClassifier(n_estimators=5000, verbose=-1, n_jobs=NJ, random_state=0, cat_smooth=20,
                                       min_data_per_group=50, interaction_constraints=inter, monotone_constraints=mon,
                                       monotone_constraints_method="advanced", **B)
                Ztr, Zva = enc.transform(Xtr), enc.transform(Xva)
                m.fit(Ztr, y[tr], eval_set=[(Zva, y[va])], categorical_feature=enc.cat_idx_ or "auto",
                      callbacks=[lgb.early_stopping(200, verbose=False)])
                oof[va] = m.predict_proba(Zva)[:, 1]
            else:
                enc = S.TabularEncoder("linear").fit(Xtr, nums, cats)
                g = S.SplineLogit(len(nums), C=0.05, n_knots=7, basis="hinge").fit(enc.transform(Xtr), y[tr])
                oof[va] = g.predict_proba(enc.transform(Xva))[:, 1]
        scs.append(primary(key, y, oof)); oofs.append(oof)
    return np.mean(scs), scs, np.mean([ll(y, o) for o in oofs])


DISTRACT = {"p1": ["ambient_humidity", "batch_weight", "warehouse_storage_days", "shift", "machine_type"],
            "p2": ["legal_structure"], "p3": ["vendor_family", "technology_generation", "weekday_weekend"]}[key]
configs = [("all", [])] + [(f"drop:{c}", [c]) for c in COPIES] + [("drop_copies", COPIES),
           ("drop_originals", {"p1": ["raw_material_density", "process_temperature"],
                               "p2": ["credit_utilization", "average_account_balance"],
                               "p3": ["packet_loss_rate", "throughput_utilization"]}[key]),
           ("drop_distract", DISTRACT), ("drop_copies+distract", COPIES + DISTRACT)]
if len(sys.argv) > 3:
    configs = [c for c in configs if c[0] in sys.argv[3].split(",")]
for model in ("lgbB",):
    for name, cols in configs:
        t0 = time.time()
        m, scs, l = run(X.drop(columns=cols), model)
        log(f"{key} {model:5s} {name:12s} primary={m:.5f} per-seed={[round(v, 5) for v in scs]} LL={l:.5f} {time.time()-t0:.0f}s")
