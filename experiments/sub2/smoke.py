import sys, time; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, 'experiments/sub2')
from common import *
from solution import StudentSolution
key = sys.argv[2]; tb = int(sys.argv[3])
X, y, nums, cats = load(key)
from problems import metadata_for
md = metadata_for({'p1':'roc_auc','p2':'log_loss','p3':'decision_cost'}[key])
from sklearn.model_selection import train_test_split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
t0 = time.time()
sol = StudentSolution(time_budget=tb, verbose=True).fit(Xtr.reset_index(drop=True), ytr, md)
p = sol.predict_proba(Xte.reset_index(drop=True), md)
print('fit', round(time.time()-t0), 's'); report('holdout', yte, p)
print('threshold', sol.get_threshold(), 'normcost', norm_cost(yte, p, sol.get_threshold()))
import json; print(json.dumps({k: v for k, v in sol.summary().items() if k not in ('profile',)}, default=str)[:3000])
