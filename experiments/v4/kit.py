"""Shared helpers for v4 experiments (run with the .venv311 interpreter)."""
import os, sys, time, warnings, importlib.util
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
warnings.filterwarnings("ignore")

WORK = os.path.dirname(os.path.abspath(__file__))
# development CSVs: $DA5401_DATA, else <repo>/data (ModelX layout: experiments/v4/), else the project folder
# three levels up (local layout: submission/v4/experiments/)
_CANDIDATES = [os.environ.get("DA5401_DATA", ""), os.path.join(WORK, "..", "..", "data"),
               os.path.join(WORK, "..", "..", "..")]
DATA = next((d for d in _CANDIDATES if d and os.path.isfile(os.path.join(d, "p1_manufacturing_development.csv"))),
            _CANDIDATES[1])
PROBS = {"p1": ("p1_manufacturing_development.csv", "quality_failure", "roc_auc"),
         "p2": ("p2_credit_development.csv", "serious_delinquency", "log_loss"),
         "p3": ("p3_telecom_development.csv", "major_service_disruption", "decision_cost")}


def metadata_for(objective):
    cost = objective == "decision_cost"
    return {"objective": objective, "greater_is_better": objective == "roc_auc", "requires_threshold": cost,
            "cost_false_positive": 1 if cost else None, "cost_false_negative": 10 if cost else None,
            "positive_label": 1}


def load(key):
    f, t, obj = PROBS[key]
    df = pd.read_csv(os.path.join(DATA, f))
    X = df.drop(columns=["record_id", t])
    y = df[t].astype(int).to_numpy()
    return X, y, metadata_for(obj)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def folds(y, seed=0, k=5):
    return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def clipp(p, e=1e-6):
    return np.clip(np.asarray(p, float), e, 1 - e)


def logit(p):
    p = clipp(p)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def ll(y, p):
    return log_loss(y, clipp(p), labels=[0, 1])


def auc(y, p):
    return roc_auc_score(y, p)


def norm_cost(y, p, t, cfp=1, cfn=10):
    a = p >= t
    return (cfp * np.sum(a & (y == 0)) + cfn * np.sum(~a & (y == 1))) / min(cfn * y.sum(), cfp * (y == 0).sum())


def best_cost(y, p):
    ts = np.unique(np.round(p, 4))
    cs = np.array([norm_cost(y, p, t) for t in ts])
    i = int(np.argmin(cs))
    return cs[i], ts[i]


def platt_cv_cost(y, p, seed=0):
    """Honest P3-style score: Platt-calibrate on 4/5, apply Bayes threshold 1/11 on 1/5 (pooled cost)."""
    from sklearn.linear_model import LogisticRegression
    z = logit(p)[:, None]
    q = np.zeros(len(y))
    for tr, va in folds(y, seed + 7):
        q[va] = LogisticRegression(C=1e6).fit(z[tr], y[tr]).predict_proba(z[va])[:, 1]
    return norm_cost(y, q, 1 / 11)


def summary(key, y, p):
    if key == "p1":
        return f"AUC={auc(y, p):.5f} LL={ll(y, p):.5f}"
    if key == "p2":
        return f"LL={ll(y, p):.5f} AUC={auc(y, p):.5f}"
    bc, bt = best_cost(y, p)
    return f"LL={ll(y, p):.5f} AUC={auc(y, p):.5f} cost@1/11cal={platt_cv_cost(y, p):.4f} bestcost={bc:.4f}"


def primary(key, y, p):
    """Higher is better."""
    return auc(y, p) if key == "p1" else -ll(y, p)


class Log:
    def __init__(self, name):
        self.f = open(os.path.join(WORK, "logs", name), "a", encoding="utf-8")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()
