"""Threshold rules on many synthetic cost problems (fast model: 5-fold bag of a depth-2 booster + Platt on OOF).
Rules: learned (the framework's smoothed-curve + bootstrap-median rule on calibrated OOF), Bayes c_fp/(c_fp+c_fn)
on calibrated probabilities, and their average in logit space.  Scored on a 20k-row test set from the generator.
usage: python exp_threshold_sim.py <first> <last> <n_jobs>"""
import sys, os, json
import numpy as np, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "..", "submission", "v6"))
import simgen, solution as S

first, last, NJ = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
COSTS = [(1, 2), (1, 5), (1, 10), (1, 20), (3, 1), (5, 1)]
out = open(os.path.join(HERE, "logs", "exp_threshold_sim.log"), "a")
for i in range(first, last):
    rng = np.random.default_rng(i)
    base = dict(simgen.SPECS[int(rng.integers(0, len(simgen.SPECS)))])
    cfp, cfn = COSTS[i % len(COSTS)]
    base.update(name=f"thr{i}", n=int(rng.choice([3000, 6000, 10000, 20000])), rate=float(rng.choice([0.03, 0.06, 0.1, 0.2, 0.4])),
                obj="decision_cost", cost=(cfp, cfn), highcard=None)
    simgen.SPECS.append(base)
    Xtr, ytr, Xte, yte, pte, md, sp = simgen.make(len(simgen.SPECS) - 1)
    simgen.SPECS.pop()
    obj = S.Objective(md)
    cl = S.DataCleaner().fit(Xtr); Xc, Xt = cl.transform(Xtr), cl.transform(Xte)
    enc = S.TabularEncoder("native").fit(Xc, cl.numeric_, cl.categorical_)
    Z, Zt = enc.transform(Xc), enc.transform(Xt)
    oof, zt = np.zeros(len(ytr)), np.zeros(len(yte))
    for tr, va in S._folds(ytr, 5, 0):
        m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.03, max_depth=2, num_leaves=4, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, verbose=-1, n_jobs=NJ)
        m.fit(Z[tr], ytr[tr], eval_set=[(Z[va], ytr[va])], callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[va] = m.predict_proba(Z[va])[:, 1]
        zt += S._logit(m.predict_proba(Zt)[:, 1]) / 5
    cal = S.PlattCalibrator().fit(oof, ytr)
    po, pt = cal.predict(oof), cal.predict(S._sigmoid(zt))
    bayes = cfp / (cfp + cfn)
    learned = S.StudentSolution._robust_threshold(ytr, po, obj, np.random.default_rng(42))
    avg = float(S._sigmoid(0.5 * (S._logit(np.array([learned]))[0] + S._logit(np.array([bayes]))[0])))
    def nc(p, t):
        a = p >= t
        return (cfp * np.sum(a & (yte == 0)) + cfn * np.sum(~a & (yte == 1))) / min(cfn * yte.sum(), cfp * (yte == 0).sum())
    r = dict(i=i, cost=(cfp, cfn), n=sp["n"], rate=sp["rate"], learned=learned, bayes=bayes, avg=avg,
             c_learned=nc(pt, learned), c_bayes=nc(pt, bayes), c_avg=nc(pt, avg), c_oracle=nc(pte, bayes))
    print(json.dumps(r), flush=True); out.write(json.dumps(r) + "\n"); out.flush()
