import sys; sys.path.insert(0, 'experiments/sub2'); sys.path.insert(0, 'submission/sub2')
from common import *
import solution as S
for key in ('p1', 'p2', 'p3'):
    X, y, nums, cats = load(key); Xc = S.DataCleaner().fit(X).transform(X)
    for C, basis in ((0.05, 'hinge'), (0.2, 'hinge'), (0.3, 'spline')):
        t0 = time.time(); oof = np.zeros(len(y))
        for tr, va in folds(y, seed=0):
            spec = S.ModelSpec('gam', dict(C=C, knots=7 if basis == 'hinge' else 8, basis=basis))
            m = S._fit_model(spec, Xc.iloc[tr], y[tr], nums, cats, None, 2); oof[va] = m.predict(Xc.iloc[va])
        report(f'{key} framework GAM {basis} C={C}', y, oof, t0)
