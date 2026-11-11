import sys; sys.path.insert(0, 'experiments/sub2')
from common import *
import lightgbm as lgb, pickle, itertools
from scipy.stats import spearmanr
key = sys.argv[1]; NJ = 1
X, y, nums, cats = load(key); Z = native(X, nums, cats); F = folds(y); cols = list(Z.columns)
cat_idx = [cols.index(c) for c in cats]
P_st = dict(learning_rate=0.05, num_leaves=2, min_child_samples=20)
P_d2 = dict(learning_rate=0.03, max_depth=2, num_leaves=4, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
def cv(name, frac=1.0, seeds=(0,), quiet=False, **kw):
    t0 = time.time(); oof = np.zeros(len(y)); its = []
    for s in seeds:
        rng = np.random.default_rng(s)
        for tr, va in F:
            if frac < 1: tr = rng.choice(tr, int(frac*len(tr)), replace=False)
            m = lgb.LGBMClassifier(n_estimators=5000, verbose=-1, n_jobs=NJ, random_state=s, **kw)
            m.fit(Z.iloc[tr], y[tr], eval_set=[(Z.iloc[va], y[va])], callbacks=[lgb.early_stopping(150, verbose=False)], categorical_feature=cat_idx)
            oof[va] += m.predict_proba(Z.iloc[va])[:, 1]/len(seeds); its.append(m.best_iteration_)
    if not quiet: report(f"{name} it={int(np.mean(its))}", y, oof, t0)
    return oof
print(f"===== {key} =====", flush=True)
print("--- learning curve (fraction of training fold) ---")
for fr in (0.25, 0.5, 0.75, 1.0):
    cv(f"stumps frac={fr}", frac=fr, **P_st); cv(f"depth2 frac={fr}", frac=fr, **P_d2)
print("--- additive shapes from a stumps model (full data) ---")
m = lgb.LGBMClassifier(n_estimators=1500, verbose=-1, n_jobs=NJ, random_state=0, **P_st).fit(Z, y, categorical_feature=cat_idx)
imp = m.booster_.feature_importance('gain'); imp = imp/imp.sum()
shapes = {}
for j, c in enumerate(cols):
    if c in cats or imp[j] < 0.002: continue
    x = Z[c].dropna(); qs = np.quantile(x, np.linspace(0.02, 0.98, 25))
    Zq = pd.DataFrame(np.tile(Z.median().values, (25,1)), columns=cols); Zq[c] = qs
    for cc in cats: Zq[cc] = Z[cc].mode()[0]
    pd_ = m.predict(Zq, raw_score=True); rho = spearmanr(qs, pd_).correlation
    d = np.diff(pd_); sign_changes = int(np.sum(np.sign(d[1:]) * np.sign(d[:-1]) < 0))
    rng_ = pd_.max()-pd_.min()
    shapes[c] = (imp[j], rho, sign_changes, rng_)
    print(f"{c:30s} gain={imp[j]:.3f} spearman={rho:+.2f} signchg={sign_changes:2d} range={rng_:.2f} " + ''.join('▁▂▃▄▅▆▇█'[int(7*(v-pd_.min())/(rng_+1e-9))] for v in pd_))
for j, c in enumerate(cols):
    if c in cats: print(f"{c:30s} gain={imp[j]:.3f} (categorical)")
print("--- interaction detection from depth-2 model ---")
m2 = lgb.LGBMClassifier(n_estimators=600, verbose=-1, n_jobs=NJ, random_state=0, **P_d2).fit(Z, y, categorical_feature=cat_idx)
pairs = {}
for t in m2.booster_.dump_model()['tree_info']:
    def walk(node, parents):
        if 'split_feature' not in node: return
        f = node['split_feature']; g = node.get('split_gain', 0)
        for pf in parents:
            if pf != f: pairs[tuple(sorted((pf, f)))] = pairs.get(tuple(sorted((pf, f))), 0) + g
        for ch in ('left_child', 'right_child'):
            if ch in node: walk(node[ch], parents + [f])
    walk(t['tree_structure'], [])
tot = sum(pairs.values()); top = sorted(pairs.items(), key=lambda kv: -kv[1])[:12]
for (a, b), g in top: print(f"  {cols[a]:28s} x {cols[b]:28s} {100*g/tot:5.1f}% of pair gain")
print("--- interaction-constrained models (additive + top-k pairs) ---")
for k in (0, 2, 4, 8):
    cons = [[i] for i in range(len(cols))] + [list(p) for p, g in top[:k]]
    cv(f"constrained d2 top{k}", interaction_constraints=cons, **P_d2)
    cv(f"constrained d3 top{k}", interaction_constraints=cons, learning_rate=0.03, max_depth=3, num_leaves=8, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
print("--- monotone constraints where the additive shape is monotone (|spearman|>0.9, <=2 sign changes) ---")
mono = [0]*len(cols)
for j, c in enumerate(cols):
    if c in shapes and abs(shapes[c][1]) > 0.9 and shapes[c][2] <= 2: mono[j] = int(np.sign(shapes[c][1]))
print("monotone features:", {cols[j]: s for j, s in enumerate(mono) if s})
cv("stumps + monotone", monotone_constraints=mono, monotone_constraints_method='advanced', **P_st)
cv("depth2 + monotone", monotone_constraints=mono, monotone_constraints_method='advanced', **P_d2)
cv("stumps lr0.01", learning_rate=0.01, num_leaves=2, min_child_samples=20)
cv("stumps lr0.05 3 seeds", seeds=(0,1,2), **P_st)
