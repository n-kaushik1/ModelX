"""
DA5401 Data Analytics Lab -- Course Project
Reusable end-to-end framework for binary tabular classification
(framework version 14 = submission v12; v11 = framework 13, v10 = framework 12, v8 = framework 10).

Public interface (as required by the evaluator):

    sol = StudentSolution()
    sol.fit(X, y, metadata)
    p = sol.predict_proba(X_test, metadata)      # 1-D array, P(y = 1)
    t = sol.get_threshold()                      # float decision threshold

The framework contains no dataset-specific logic.  Everything it does is driven
by (a) measured properties of the training data and (b) the objective that is
supplied via ``metadata`` (primary metric, decision costs, identifier columns).

Workflow implemented in ``fit``
    1. profile the data  : drop identifiers / constants, coerce numeric-like
                           strings, split numeric vs categorical, measure
                           missingness, class balance, cardinality, size;
                           date columns become numeric parts.  Measured
                           redundancy: a numeric column that is a noisy copy of
                           another (same units, var(clean) = cov) is dropped,
                           two noisy twin readings get their weighted average.
    2. preprocessing     : model-specific encoders built from the profile
                           (native-NaN + integer codes for gradient boosting,
                           imputed + one-hot for forests, imputed + log /
                           standardise + one-hot (+ hinge basis) for linear
                           models, raw strings for CatBoost).  Unseen / rare
                           categories are mapped to a "rare" bucket.
    3. structure         : (nested in every fold) a quick depth-2 booster ranks
                           interacting feature pairs, a quick additive booster
                           gives monotone directions; boosting candidates are
                           searched with and without these constraints.
    4. candidates        : prior baseline, hinge-basis GAM, LightGBM, XGBoost,
                           CatBoost, ExtraTrees (HistGradientBoosting only as a
                           fallback when LightGBM is missing).
    5. tuning            : per boosting family a *complexity ladder* (depth 1
                           ... 6, trees by early stopping inside each fold) and
                           a fixed number of random refinements around the best
                           rung, all scored by stratified K-fold OOF predictions
                           on the problem's metric (log loss when the metric
                           needs a threshold).  time_budget only guards slow
                           machines.
    6. library           : the best configurations of every family (structured
                           and unconstrained boosters, two GAMs, ...) are refit
                           on fresh folds with a fixed tree count; their k fold
                           models are kept (k-fold bagging) and their OOF
                           predictions drive everything that follows.
    7. combiner          : bagged ensemble selection in logit space (greedy
                           Caruana selection on bootstrap resamples of the OOF
                           rows, weights averaged).
    8. calibration       : a Platt map on the blend, kept only if nested CV
                           shows a log-loss gain (monotone: AUC unchanged).
    9. threshold         : cost metrics: c_fp / (c_fp + c_fn) on the calibrated
                           probabilities, unless the empirical OOF optimum wins
                           out of sample in >= 95 % of bootstrap rounds; other
                           decision metrics (F1, ...): smoothed OOF optimum.
   10. final predictor   : the selected members' fold models, logits averaged
                           (k-fold bagging; forests and very large data: one
                           refit on all rows).
"""

from __future__ import annotations

import contextlib
import os
import re
import time
import warnings
import zlib
from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import SplineTransformer

try:  # optional -- the framework degrades gracefully if LightGBM is missing
    import lightgbm as lgb
    _HAS_LGBM = True
except Exception:  # pragma: no cover
    lgb = None
    _HAS_LGBM = False

try:  # optional -- CatBoost (ordered target statistics for categoricals)
    import catboost as cb
    _HAS_CAT = True
except Exception:  # pragma: no cover
    cb = None
    _HAS_CAT = False

try:  # thread limits for BLAS (scikit-learn dependency); without it the linear fits simply run unlimited
    from threadpoolctl import threadpool_limits as _tp_limits
except Exception:  # pragma: no cover
    _tp_limits = None


def _blas_limit(n=4):
    """Cap BLAS threads for the small linear-model fits: with all cores (the default) an lbfgs logistic
    regression on a few thousand rows is several times slower, and badly so on a shared machine."""
    if _tp_limits is None:
        return contextlib.nullcontext()
    try:
        return _tp_limits(limits=n, user_api="blas")
    except Exception:
        return contextlib.nullcontext()


try:  # optional -- XGBoost
    import xgboost as xgb
    _HAS_XGB = True
except Exception:  # pragma: no cover
    xgb = None
    _HAS_XGB = False

warnings.filterwarnings("ignore")

_EPS = 1e-4
def _available_cpus():
    """CPUs this process may actually use (respects container / affinity limits)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        return max(1, os.cpu_count() or 1)


_N_CPU = _available_cpus()


# =============================================================================
# 1. Metadata / objective handling
# =============================================================================
def _flatten(obj, prefix="", out=None, depth=0):
    """Flatten nested mappings into {'a_b_c': value} with lower-case keys."""
    out = {} if out is None else out
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            key = f"{prefix}_{k}" if prefix else str(k)
            if isinstance(v, Mapping) and depth < 3:
                _flatten(v, key.lower(), out, depth + 1)
            else:
                out[key.lower()] = v
    return out


def _to_dict(metadata):
    if metadata is None:
        return {}
    if isinstance(metadata, Mapping):
        return _flatten(metadata)
    try:
        return _flatten(vars(metadata))
    except Exception:
        return {}


def _tokens(key):
    return set(re.split(r"[^a-z0-9]+", key.lower()))


def _norm_metric(name):
    s = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
    if "cost" in s or "expected_loss" in s:
        return "cost"
    if ("pr" in s.split("_") and "auc" in s) or "average_precision" in s or s in ("ap", "auprc"):
        return "average_precision"
    if "auc" in s or "roc" in s or "gini" in s:
        return "roc_auc"
    if "log" in s or "cross_entropy" in s or "entropy" in s or "nll" in s:
        return "log_loss"
    if "brier" in s:
        return "brier"
    if "balanced_acc" in s:
        return "balanced_accuracy"
    if "f1" in s or s == "f_score":
        return "f1"
    if "acc" in s:
        return "accuracy"
    return None


class Objective:
    """Evaluation objective decoded from metadata (with safe defaults)."""

    THRESHOLD_METRICS = ("cost", "f1", "accuracy", "balanced_accuracy")

    def __init__(self, metadata):
        md = _to_dict(metadata)
        self.fp_cost, self.fn_cost = None, None
        self.metric = None
        self.id_columns, self.target_name = [], None

        metric_keys = ("metric", "primary_metric", "eval_metric", "evaluation_metric", "scoring",
                       "objective", "task_metric", "score")
        for k in metric_keys:
            for key, v in md.items():
                if (key == k or key.endswith("_" + k)) and isinstance(v, str):
                    m = _norm_metric(v)
                    if m:
                        self.metric = m
                        break
            if self.metric:
                break

        for key, v in md.items():
            if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
                continue
            tk = _tokens(key)
            if "cost" in tk or "penalty" in tk or "loss" in tk or "weight" in tk or "fp" in tk or "fn" in tk:
                if "fp" in tk or ("false" in tk and "positive" in tk):
                    self.fp_cost = float(v)
                elif "fn" in tk or ("false" in tk and "negative" in tk):
                    self.fn_cost = float(v)
        if self.fp_cost is not None or self.fn_cost is not None:
            self.fp_cost = 1.0 if self.fp_cost is None else self.fp_cost
            self.fn_cost = 1.0 if self.fn_cost is None else self.fn_cost
            if self.metric is None:
                self.metric = "cost"
        if self.metric == "cost" and self.fp_cost is None:
            self.fp_cost = self.fn_cost = 1.0

        for key, v in md.items():
            tk = _tokens(key)
            if ("id" in tk or "ids" in tk or "identifier" in tk or "identifiers" in tk) and "columns" in tk or key in (
                    "id_col", "id_cols", "id_column", "id_columns", "identifier", "identifiers", "id"):
                if isinstance(v, str):
                    self.id_columns.append(v)
                elif isinstance(v, (list, tuple, set)):
                    self.id_columns += [c for c in v if isinstance(c, str)]
            if key in ("target", "target_col", "target_column", "target_name", "label", "label_column"):
                if isinstance(v, str):
                    self.target_name = v
        # explicit flags used by the leaderboard metadata
        self.requires_threshold = bool(md.get("requires_threshold", False)) if not isinstance(
            md.get("requires_threshold"), str) else md["requires_threshold"].strip().lower() in ("1", "true", "yes")
        self.positive_label = md.get("positive_label", None)
        if self.requires_threshold and self.metric not in self.THRESHOLD_METRICS:
            # a decision is required: use the stated costs, or plain accuracy (equal costs) if none are given
            self.metric = "cost"
            if self.fp_cost is None:
                self.fp_cost = self.fn_cost = 1.0
        if self.metric is None:
            self.metric = "log_loss"          # proper scoring rule: a safe default
            self.metric_was_default = True
        else:
            self.metric_was_default = False

    # ---- score helpers (all "higher is better") ------------------------------
    @property
    def uses_threshold(self):
        return self.metric in self.THRESHOLD_METRICS

    @property
    def tune_metric(self):
        """Metric used during hyper-parameter search (threshold-free)."""
        return "log_loss" if self.uses_threshold or self.metric in ("brier",) else self.metric

    def describe(self):
        d = {"metric": self.metric, "metric_defaulted": self.metric_was_default}
        if self.metric == "cost":
            d.update(fp_cost=self.fp_cost, fn_cost=self.fn_cost)
        return d


def _clip(p):
    return np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)


def _score_free(metric, y, p):
    """Threshold-free metrics, oriented so that larger is better."""
    if len(np.unique(y)) < 2:
        return 0.0
    if metric == "roc_auc":
        return roc_auc_score(y, p)
    if metric == "average_precision":
        return average_precision_score(y, p)
    if metric == "log_loss":
        return -log_loss(y, _clip(p), labels=[0, 1])
    if metric == "brier":
        return -brier_score_loss(y, p)
    raise ValueError(metric)


def _conf_curve(y, p):
    """Confusion counts for every distinct threshold (positive iff p >= thr),
    preceded by the "nobody is positive" option."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-p, kind="mergesort")
    ps, ys = p[order], y[order]
    last = np.r_[ps[1:] != ps[:-1], True]            # last index of each tie group
    cum = np.cumsum(ys)
    tp = cum[last]
    fp = (np.arange(1, len(ys) + 1) - cum)[last]
    thr = ps[last]
    P, N = y.sum(), len(y) - y.sum()
    thr = np.r_[np.nextafter(thr[0], np.inf), thr]
    tp, fp = np.r_[0, tp], np.r_[0, fp]
    return thr, tp, fp, P - tp, N - fp


def _threshold_metric_from_counts(metric, tp, fp, fn, tn, obj):
    n = tp + fp + fn + tn
    if metric == "cost":
        return -(obj.fp_cost * fp + obj.fn_cost * fn) / n          # per-sample cost, negated
    if metric == "accuracy":
        return (tp + tn) / n
    if metric == "balanced_accuracy":
        return 0.5 * (tp / np.maximum(tp + fn, 1) + tn / np.maximum(tn + fp, 1))
    if metric == "f1":
        return 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    raise ValueError(metric)


def _best_threshold(y, p, obj, smooth_window=0):
    """Return (threshold, best_score).  Optionally smooths the score curve."""
    thr, tp, fp, fn, tn = _conf_curve(y, p)
    sc = _threshold_metric_from_counts(obj.metric, tp, fp, fn, tn, obj)
    if smooth_window and len(sc) > smooth_window:
        k = np.ones(smooth_window) / smooth_window
        h = smooth_window // 2
        sm = np.convolve(np.pad(sc, (h, smooth_window - 1 - h), mode="edge"), k, mode="valid")
        i = int(np.argmax(sm))
    else:
        i = int(np.argmax(sc))
    t = thr[i]
    if i + 1 < len(thr):                              # midpoint to the next lower distinct value
        t = 0.5 * (thr[i] + thr[i + 1])
    return float(t), float(sc[i])


def _selection_score(obj, y, p):
    """Score used to pick between models / blends (higher = better)."""
    if obj.uses_threshold:
        _, s = _best_threshold(y, p, obj)
        return s
    return _score_free(obj.metric, y, p)


def _tuning_score(obj, y, p):
    return _score_free(obj.tune_metric, y, p)


# =============================================================================
# 2. Data profiling and cleaning
# =============================================================================
_ID_NAME = re.compile(r"(^|_)(id|ids|uuid|guid|identifier)($|_)|record_?id", re.I)


def _is_id_name(name):
    n = str(name).strip().lower()
    return bool(_ID_NAME.search(n))


class DataCleaner:
    """Learns which columns to keep and how to type them; applies it to new data."""

    def __init__(self, extra_ids=(), target_name=None):
        self.extra_ids = set(extra_ids)
        self.target_name = target_name

    def fit(self, X):
        X = self._as_df(X)
        n = len(X)
        self.dropped_ = {"id": [], "constant": [], "empty": [], "unique_text": [], "unsupported": []}
        self.numeric_, self.categorical_, self.coerced_ = [], [], []
        self.dates_ = {}                                   # date column -> derived numeric columns
        for c in X.columns:
            s = X[c]
            if c in self.extra_ids or c == self.target_name:
                self.dropped_["id"].append(c)
                continue
            # an identifier-like *name* only drops a column that also behaves like an identifier (mostly
            # unique values); a low-cardinality "machine_id" or "region_id" is a real categorical
            if _is_id_name(c) and (s.nunique(dropna=True) >= 0.5 * max(1, s.notna().sum()) or n < 50):
                self.dropped_["id"].append(c)
                continue
            if pd.api.types.is_timedelta64_dtype(s):
                s = s.dt.total_seconds()
            if pd.api.types.is_datetime64_any_dtype(s) or self._looks_like_date(s):
                self._add_date(c, s)
                continue
            if s.isna().all():
                self.dropped_["empty"].append(c)
                continue
            if s.dropna().nunique() <= 1 and s.isna().sum() == 0:
                self.dropped_["constant"].append(c)
                continue
            if pd.api.types.is_bool_dtype(s):
                self.categorical_.append(c)
                continue
            if pd.api.types.is_numeric_dtype(s):
                self.numeric_.append(c)
                continue
            # text-like column: numeric strings -> numeric; unique text -> identifier
            conv = pd.to_numeric(s, errors="coerce")
            nn = s.notna().sum()
            # numeric text, possibly with a few placeholder tokens ("n/a", "?", "-") that become missing values
            bad = s[s.notna() & conv.isna()].astype(str).str.strip()
            if nn > 0 and (conv.notna().sum() >= 0.95 * nn or (conv.notna().sum() >= 0.8 * nn and bad.nunique() <= 5)):
                self.numeric_.append(c)
                self.coerced_.append(c)
                continue
            if n >= 50 and s.nunique(dropna=True) >= 0.9 * nn:
                self.dropped_["unique_text"].append(c)
                continue
            self.categorical_.append(c)
        self.columns_ = self.numeric_ + self.categorical_
        return self

    # -- date / time columns -> numeric features ----------------------------------
    _DATE_PARTS = ("days", "month", "dow", "hour")

    @staticmethod
    def _looks_like_date(s):
        """Text column whose values parse as dates (and are not plain numbers)."""
        if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
            return False
        v = s.dropna().astype(str)
        if len(v) < 5:
            return False
        v = v.sample(min(500, len(v)), random_state=0)
        if pd.to_numeric(v, errors="coerce").notna().mean() > 0.5:
            return False
        full_date = r"(\d{4}[-/.]\d{1,2}[-/.]\d{1,2})|(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})"
        if not v.str.contains(full_date, regex=True).mean() > 0.9:
            return False
        return pd.to_datetime(v, errors="coerce", format="mixed").notna().mean() > 0.95

    @staticmethod
    def _to_datetime(s):
        if pd.api.types.is_datetime64_any_dtype(s):
            d = s
        else:
            try:
                d = pd.to_datetime(s.astype("object").where(s.notna(), None), errors="coerce", format="mixed")
            except Exception:
                d = pd.Series(pd.NaT, index=s.index)
        try:
            if getattr(d.dt, "tz", None) is not None:
                d = d.dt.tz_convert(None)
        except Exception:
            pass
        return d

    def _add_date(self, c, s):
        d = self._to_datetime(s)
        if d.notna().sum() == 0:
            self.dropped_["unsupported"].append(c)
            return
        parts = {"days": (d - pd.Timestamp("1970-01-01")).dt.total_seconds() / 86400.0,
                 "month": d.dt.month, "dow": d.dt.dayofweek, "hour": d.dt.hour}
        keep = [k for k, v in parts.items() if v.dropna().nunique() > 1]
        if not keep:
            self.dropped_["constant"].append(c)
            return
        self.dates_[c] = keep
        self.numeric_ += [f"{c}__{k}" for k in keep]

    def set_redundancy(self, drop, twins):
        """Apply the measured redundancy (see _find_redundant): drop noisy copies, add twin averages."""
        self.redundant_ = list(drop)
        self.twins_ = list(twins)
        self.dropped_["redundant_copy"] = list(drop)
        self.numeric_ = [c for c in self.numeric_ if c not in drop] + [t["name"] for t in twins]
        self.columns_ = self.numeric_ + self.categorical_
        return self

    @staticmethod
    def _as_df(X):
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X)

    def transform(self, X):
        X = self._as_df(X)
        out = pd.DataFrame(index=X.index)
        derived = {}
        for c, keep in getattr(self, "dates_", {}).items():
            if c in X:
                d = self._to_datetime(X[c])
                vals = {"days": (d - pd.Timestamp("1970-01-01")).dt.total_seconds() / 86400.0,
                        "month": d.dt.month, "dow": d.dt.dayofweek, "hour": d.dt.hour}
            else:
                vals = {}
            for k in keep:
                derived[f"{c}__{k}"] = vals[k].astype(float) if k in vals else np.nan
        for t in getattr(self, "twins_", []):                    # noise-weighted average of two twin readings
            xa = pd.to_numeric(X[t["a"]], errors="coerce").to_numpy(dtype=float) if t["a"] in X else np.full(len(X), np.nan)
            xb = pd.to_numeric(X[t["b"]], errors="coerce").to_numpy(dtype=float) if t["b"] in X else np.full(len(X), np.nan)
            derived[t["name"]] = np.where(np.isnan(xa), xb, np.where(np.isnan(xb), xa, t["wa"] * xa + t["wb"] * xb))
        for c in self.numeric_:
            if c in derived:
                out[c] = derived[c]
            elif c in X:
                col = X[c]
                if pd.api.types.is_timedelta64_dtype(col):
                    col = col.dt.total_seconds()
                out[c] = pd.to_numeric(col, errors="coerce").astype(float)
            else:
                out[c] = np.nan
        for c in self.categorical_:
            out[c] = X[c].astype("object").where(X[c].notna(), None) if c in X else None
            out[c] = out[c].map(lambda v: None if v is None else str(v))
        out = out.replace([np.inf, -np.inf], np.nan)
        return out


def _noise_shares(a, b):
    """Measurement-error decomposition of two columns in the same units.  If b = a + e with e independent of a,
    then var(a) = cov(a, b): the share (var(x) - cov) / var(x) is ~0 for the clean column and > 0 for the noisy
    one; two noisy readings of a hidden quantity both have a positive share.  Computed over all rows (restricting
    the rows to a range of the noisy column would bias its share towards zero)."""
    c = float(np.cov(a, b)[0, 1])
    va, vb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    return [(va - c) / va if va > 0 else np.nan, (vb - c) / vb if vb > 0 else np.nan]


def _find_redundant(Xc, num_cols, min_corr=0.9, max_rows=50000, seed=0):
    """Measured redundancy among numeric columns (no names, no target involved).
    Pairs with |correlation| >= ``min_corr`` that are in the same units (equal means within 0.1 sd, sd ratio within
    0.8-1.25) are decomposed with ``_noise_shares``:
      * noisy copy  (one share <= 1/3 of the other, the larger >= 0.005): the noisier column is dropped -- it carries
        no information the clean one lacks and only competes for splits;
      * twin readings (both shares >= 0.008, within a factor 2): both are kept and their noise-weighted average is
        added as a feature (a better estimate of the hidden quantity, which trees cannot form themselves);
      * anything else is left alone.
    Returns (columns to drop, twin specs)."""
    drop, twins = [], []
    rng = np.random.default_rng(seed)
    D = Xc[num_cols]
    if len(D) > max_rows:
        D = D.iloc[rng.choice(len(D), max_rows, replace=False)]
    nun = {c: D[c].nunique() for c in num_cols}
    cols = [c for c in num_cols if nun[c] >= 5]
    if len(cols) < 2:
        return drop, twins
    C = D[cols].corr()
    pairs = sorted(((abs(C.loc[a, b]), a, b) for i, a in enumerate(cols) for b in cols[i + 1:]
                    if np.isfinite(C.loc[a, b]) and abs(C.loc[a, b]) >= min_corr and max(nun[a], nun[b]) > 20),
                   reverse=True)
    for _, a, b in pairs:
        if a in drop or b in drop:
            continue
        xa, xb = D[a].to_numpy(dtype=float), D[b].to_numpy(dtype=float)
        ok = np.isfinite(xa) & np.isfinite(xb)
        if ok.sum() < 200:
            continue
        xa, xb = xa[ok], xb[ok]
        sa_, sb_ = xa.std(), xb.std()
        if not (sa_ > 0 and abs(xa.mean() - xb.mean()) <= 0.1 * sa_ and 0.8 <= sb_ / sa_ <= 1.25):
            continue                                                    # not the same units: undecidable
        na, nb = _noise_shares(xa, xb)
        if not (np.isfinite(na) and np.isfinite(nb)):
            continue
        lo, hi = min(na, nb), max(na, nb)
        if hi >= 0.005 and lo <= hi / 3:
            drop.append(b if nb > na else a)
        elif lo >= 0.008 and lo >= 0.5 * hi:
            wa, wb = 1.0 / na, 1.0 / nb
            twins.append({"name": f"{a}__{b}__avg", "a": a, "b": b, "wa": wa / (wa + wb), "wb": wb / (wa + wb)})
    return drop, twins


def profile_data(Xc, y, cleaner):
    """Measured data characteristics that drive the framework's decisions."""
    n, p = Xc.shape
    prior = float(np.mean(y))
    prof = {
        "n_rows": n, "n_features": p,
        "n_numeric": len(cleaner.numeric_), "n_categorical": len(cleaner.categorical_),
        "positive_rate": prior,
        "minority_count": int(min(y.sum(), n - y.sum())),
        "imbalanced": bool(min(prior, 1 - prior) < 0.15),
        "missing_rate_overall": float(Xc.isna().mean().mean()) if p else 0.0,
        "n_cols_with_missing": int((Xc.isna().sum() > 0).sum()),
        "max_cardinality": int(max([Xc[c].nunique() for c in cleaner.categorical_] or [0])),
        "dropped": {k: v for k, v in cleaner.dropped_.items() if v},
    }
    if cleaner.numeric_:
        sk = Xc[cleaner.numeric_].skew().abs()
        prof["n_skewed_numeric"] = int((sk > 1).sum())
    return prof


# =============================================================================
# 3. Preprocessing (one encoder, three output modes)
# =============================================================================
class TabularEncoder:
    """
    mode = "native"  : numerics untouched (NaN kept); categoricals -> integer codes
                       (NaN for missing) so that GBMs can use native handling.
    mode = "imputed" : median imputation + missing indicators; one-hot categoricals.
    mode = "linear"  : as "imputed" + log transform of skewed columns, standardise,
                       clip to +-5 sd.
    """

    def __init__(self, mode, max_levels=30, min_freq=0.005, max_onehot=25):
        self.mode = mode
        self.max_levels = max_levels
        self.min_freq = min_freq
        self.max_onehot = max_onehot

    def fit(self, Xc, num_cols, cat_cols):
        self.num_cols, self.cat_cols = list(num_cols), list(cat_cols)
        n = len(Xc)
        thr = max(5, int(self.min_freq * n))
        self.levels_ = {}
        for c in self.cat_cols:
            vc = Xc[c].value_counts()
            keep = [v for v in vc.index if vc[v] >= thr][: self.max_levels]
            self.levels_[c] = keep
        self.freq_ = {c: Xc[c].value_counts(normalize=True).to_dict() for c in self.cat_cols}
        self.onehot_cols_ = [c for c in self.cat_cols if len(self.levels_[c]) <= self.max_onehot]
        self.ordinal_cols_ = [c for c in self.cat_cols if c not in self.onehot_cols_]
        self.median_ = Xc[self.num_cols].median().fillna(0.0) if self.num_cols else pd.Series(dtype=float)
        self.na_cols_ = [c for c in self.num_cols if Xc[c].isna().any()][:100]
        self.log_cols_ = {}
        if self.mode == "linear" and self.num_cols:
            for c in self.num_cols:
                x = Xc[c].dropna()
                if len(x) > 20 and abs(x.skew()) > 1.0 and x.nunique() > 10:
                    self.log_cols_[c] = bool((x >= 0).all())
            Z = self._numeric_block(Xc, fitting=True)
            self.mu_, self.sd_ = Z.mean(0), Z.std(0)
            self.sd_[self.sd_ < 1e-12] = 1.0
        self.feature_names_ = self._names()
        self.cat_idx_ = self._cat_idx()
        return self

    # -- helpers --------------------------------------------------------------
    def _cat_idx(self):
        base = len(self.num_cols)
        if self.mode == "frame":
            return [base + i for i in range(len(self.cat_cols))]
        if self.mode != "native":
            return []
        return [base + i for i, c in enumerate(self.cat_cols) if len(self.levels_[c]) + 1 <= 30]

    def _names(self):
        names = list(self.num_cols)
        if self.mode in ("native", "frame"):
            return names + [f"{c}__code" for c in self.cat_cols]
        names += [f"{c}__isna" for c in self.na_cols_]
        for c in self.onehot_cols_:
            names += [f"{c}={l}" for l in self.levels_[c]] + [f"{c}=__rare__", f"{c}=__na__"]
        names += [f"{c}__code" for c in self.ordinal_cols_]
        names += [f"{c}__freq" for c in self.cat_cols if c in self.ordinal_cols_]
        return names

    def _map_cat(self, s, c):
        lv = self.levels_[c]
        idx = {v: i for i, v in enumerate(lv)}
        rare = len(lv)
        return s.map(lambda v: np.nan if v is None or (isinstance(v, float) and np.isnan(v))
                     else float(idx.get(v, rare)))

    def _numeric_block(self, Xc, fitting=False):
        Z = Xc[self.num_cols].to_numpy(dtype=float).copy() if self.num_cols else np.zeros((len(Xc), 0))
        if self.mode in ("native", "frame"):
            return Z
        med = self.median_.to_numpy()
        for j, c in enumerate(self.num_cols):
            m = np.isnan(Z[:, j])
            Z[m, j] = med[j]
        if self.mode == "linear":
            for j, c in enumerate(self.num_cols):
                if c in self.log_cols_:
                    z = Z[:, j]
                    Z[:, j] = np.log1p(np.maximum(z, 0)) if self.log_cols_[c] else np.sign(z) * np.log1p(np.abs(z))
        return Z

    def transform(self, Xc):
        n = len(Xc)
        Z = self._numeric_block(Xc)
        if self.mode == "frame":                       # CatBoost: numerics with NaN + string categories
            out = pd.DataFrame(Z, columns=[f"n{j}" for j in range(Z.shape[1])])
            for j, c in enumerate(self.cat_cols):
                known = set(self.levels_[c])
                out[f"c{j}"] = [("__na__" if v is None or (isinstance(v, float) and v != v)
                                 else (v if v in known else "__rare__")) for v in Xc[c].tolist()]
            return out
        if self.mode == "native":
            cats = [self._map_cat(Xc[c], c).to_numpy(dtype=float)[:, None] for c in self.cat_cols]
            return np.hstack([Z] + cats) if cats else Z
        if self.mode == "linear" and self.num_cols:
            Z = np.clip((Z - self.mu_) / self.sd_, -5, 5)
        blocks = [Z]
        if self.na_cols_:
            blocks.append(Xc[self.na_cols_].isna().to_numpy(dtype=float))
        for c in self.onehot_cols_:
            lv = self.levels_[c]
            codes = self._map_cat(Xc[c], c).to_numpy()
            oh = np.zeros((n, len(lv) + 2))
            isna = np.isnan(codes)
            ok = ~isna
            oh[np.where(ok)[0], codes[ok].astype(int)] = 1.0
            oh[isna, len(lv) + 1] = 1.0
            blocks.append(oh)
        for c in self.ordinal_cols_:
            codes = self._map_cat(Xc[c], c).to_numpy()
            blocks.append(np.where(np.isnan(codes), -1.0, codes)[:, None])
        for c in self.ordinal_cols_:
            blocks.append(Xc[c].map(self.freq_[c]).fillna(0.0).to_numpy(dtype=float)[:, None])
        return np.hstack(blocks)


# =============================================================================
# 4. Candidate model families
# =============================================================================
class ModelSpec:
    """A (family, hyper-parameters) pair that can be fitted on a cleaned frame."""

    def __init__(self, family, params, seed=0):
        self.family, self.params, self.seed = family, dict(params), seed

    def label(self):
        return f"{self.family}({', '.join(f'{k}={self._fmt(v)}' for k, v in self.params.items())})"

    @staticmethod
    def _fmt(v):
        return f"{v:.4g}" if isinstance(v, float) else str(v)


_FAMILY_MODE = {"prior": "imputed", "logreg": "linear", "gam": "linear", "hgb": "native", "lgbm": "native",
                "rf": "imputed", "et": "imputed", "cat": "frame", "xgb": "native", "hyb": "native"}
_BOOSTERS = ("lgbm", "xgb", "cat", "hyb")
_FORESTS = ("et", "rf")
_MAX_TREES = {"lgbm": 8000, "xgb": 4000, "cat": 3000, "hyb": 8000}
_PATIENCE = {"lgbm": 150, "xgb": 150, "cat": 100, "hyb": 150}


class SplineLogit:
    """Logistic regression on a piecewise-linear (hinge) basis of the (standardised) numeric columns,
    i.e. a regularised additive model (GAM).  Each numeric column with enough distinct values gets the
    linear term plus max(x - knot, 0) at ``n_knots`` quantile knots; ``basis="spline"`` uses cubic
    B-splines instead.  Hinges match threshold- and U-shaped effects better than smooth splines did.
    ``pairs`` (numeric column index pairs, from the measured structure) adds the product of the two
    standardised columns: a bilinear interaction term, i.e. a GA2M with the few interactions the data asks
    for.  Trees can only approximate such a product with many splits; on the public credit problem two
    product terms improved the GAM's log loss by 0.0016 and the ensemble's by 0.0008."""

    def __init__(self, n_numeric, C=0.3, n_knots=8, basis="hinge", pairs=None, cat_slopes=None, penalty="l2",
                 pair_hinge=0, cat_hinge=0):
        self.n_numeric, self.C, self.n_knots, self.basis = n_numeric, C, n_knots, basis
        # kinked interactions: for the first `pair_hinge` numeric pairs, max(a - k, 0) * b and a * max(b - k, 0)
        # at four quantile knots; for the first `cat_hinge` category pairs, a category-specific hinge per level
        self.pair_hinge, self.cat_hinge = int(pair_hinge), int(cat_hinge)
        # "l1": sparse GAM -- every design column standardised and an L1 penalty, so only a few hinges survive:
        # piecewise-linear shapes with a few sharp kinks (L1 on the kink basis = trend filtering)
        self.penalty = penalty
        self.pairs = [tuple(int(i) for i in p) for p in (pairs or [])]
        # (numeric column, one-hot columns of a category): the numeric column's slope differs per category level
        self.cat_slopes = [(int(n), [int(j) for j in oh]) for n, oh in (cat_slopes or [])]

    def _extra(self, Z, fitting=False):
        pairs, cslopes = getattr(self, "pairs", []), getattr(self, "cat_slopes", [])
        ph, ch = pairs[:getattr(self, "pair_hinge", 0)], cslopes[:getattr(self, "cat_hinge", 0)]
        if fitting:
            used = sorted({j for p in ph for j in p} | {n for n, _ in ch})
            self.ik_ = {j: np.quantile(Z[:, j], [0.2, 0.4, 0.6, 0.8]) for j in used}
        B = [np.clip(Z[:, a] * Z[:, b], -10, 10)[:, None] for a, b in pairs]
        B += [Z[:, oh] * Z[:, [n]] for n, oh in cslopes]
        for a, b in ph:
            B += [np.clip(np.maximum(Z[:, a] - u, 0) * Z[:, b], -10, 10)[:, None] for u in self.ik_[a]]
            B += [np.clip(Z[:, a] * np.maximum(Z[:, b] - v, 0), -10, 10)[:, None] for v in self.ik_[b]]
        for n, oh in ch:
            B += [Z[:, oh] * np.maximum(Z[:, [n]] - u, 0) for u in self.ik_[n]]
        return np.hstack(B) if B else None

    def _design(self, Z, fitting=False):
        if fitting:
            nun = [len(np.unique(Z[:, j])) for j in range(self.n_numeric)]
            self.spline_cols_ = [j for j in range(self.n_numeric) if nun[j] > 10]
            if self.spline_cols_:
                if self.basis == "spline":
                    self.spline_ = SplineTransformer(n_knots=self.n_knots, degree=3, knots="quantile",
                                                     include_bias=False).fit(Z[:, self.spline_cols_])
                else:
                    if getattr(self, "penalty", "l2") == "l1":   # sparse GAM: interior knots only (tail hinges rest
                        qs = np.linspace(0, 1, self.n_knots + 2)[1:-1]   # on few rows and add noise under L1)
                    else:
                        qs = np.linspace(0.5 / self.n_knots, 1 - 0.5 / self.n_knots, self.n_knots)
                    self.knots_ = np.array([np.quantile(Z[:, j], qs) for j in self.spline_cols_])   # (cols, knots)
        blocks = [Z]
        if self.spline_cols_:
            if self.basis == "spline":
                blocks.append(self.spline_.transform(Z[:, self.spline_cols_]))
            else:
                S = Z[:, self.spline_cols_]
                blocks.append(np.maximum(S[:, :, None] - self.knots_[None, :, :], 0.0).reshape(len(Z), -1))
        E = self._extra(Z, fitting=fitting)
        if E is not None:                                   # interaction terms, standardised on the training rows
            if fitting:
                self.emu_, self.esd_ = E.mean(0), E.std(0) + 1e-9
            blocks.append((E - self.emu_) / self.esd_)
        return np.hstack(blocks)

    def fit(self, Z, y):
        D = self._design(np.asarray(Z, dtype=float), fitting=True)
        with _blas_limit():
            if getattr(self, "penalty", "l2") == "l1":
                self.dmu_, self.dsd_ = D.mean(0), D.std(0) + 1e-9
                self.lr_ = LogisticRegression(penalty="l1", C=self.C, solver="liblinear", max_iter=2000).fit(
                    (D - self.dmu_) / self.dsd_, y)
            else:
                self.lr_ = LogisticRegression(C=self.C, max_iter=3000, solver="lbfgs").fit(D, y)
        return self

    def predict_proba(self, Z):
        D = self._design(np.asarray(Z, dtype=float))
        if getattr(self, "penalty", "l2") == "l1":
            D = (D - self.dmu_) / self.dsd_
        return self.lr_.predict_proba(D)


class _PairEncoder:
    """Native (booster) and linear (GAM) encodings of the same rows, for the GAM + booster hybrid."""

    def __init__(self, native, linear):
        self.native, self.linear = native, linear

    def transform(self, Xc):
        return self.native.transform(Xc), self.linear.transform(Xc)


class GamBoost:
    """Hybrid: a sparse GAM gives the base logit, a small booster models its residual (init score = GAM logit).
    The GAM captures the piecewise-linear shapes and kinked interactions with few parameters; the booster adds
    what the GAM cannot express (e.g. very steep walls of a U-shape)."""

    def __init__(self, gam, booster):
        self.gam, self.booster = gam, booster

    def predict_proba(self, ZZ):
        Zn, Zl = ZZ
        z = _logit(self.gam.predict_proba(Zl)[:, 1]) + self.booster.predict_proba(Zn, raw_score=True)
        p = _sigmoid(z)
        return np.c_[1 - p, p]


def _gam_terms(P, Xc, y, num_cols, cat_cols, n_jobs, enc_linear):
    """(numeric pairs, category slopes) requested by GAM parameters P, from the measured structure."""
    prs, cslopes = None, None
    if P.get("pairs") or P.get("cpairs"):
        st = _detect_structure(Xc, y, num_cols, cat_cols, n_jobs)
        nn = len(num_cols)
        if st:
            prs = [tuple(q) for q in st["pairs"] if q[0] < nn and q[1] < nn][: int(P.get("pairs", 0))]
            cslopes = []
            for q in st["pairs"]:
                if len(cslopes) >= int(P.get("cpairs", 0)):
                    break
                if (q[0] < nn) == (q[1] < nn):
                    continue
                ni, ci = (q[0], q[1]) if q[0] < nn else (q[1], q[0])
                cname = cat_cols[ci - nn]
                oh = [k for k, f in enumerate(enc_linear.feature_names_) if f.startswith(f"{cname}=")]
                if oh:
                    cslopes.append((ni, oh))
    return prs, cslopes


class FittedModel:
    def __init__(self, spec, enc, est, best_iter=None):
        self.spec, self.enc, self.est, self.best_iter = spec, enc, est, best_iter

    def predict(self, Xc):
        Z = self.enc.transform(Xc)
        p = self.est.predict_proba(Z)[:, 1]
        return np.asarray(p, dtype=float)


def _leaves_for(depth, leaves):
    """Number of leaves consistent with a depth limit."""
    return int(max(2, min(leaves, 2 ** depth)))


def _fit_model(spec, Xc, y, num_cols, cat_cols, obj, n_jobs, eval_data=None):
    """Fit one candidate.  ``eval_data=(Xv, yv)`` enables early stopping for the boosting
    families (used during tuning only); the number of trees found is stored on the result."""
    fam, P, seed = spec.family, spec.params, spec.seed
    mode = _FAMILY_MODE[fam]
    # CatBoost copes with high-cardinality categoricals itself, so keep (many) more levels for it
    enc = TabularEncoder(mode, max_levels=1000 if mode == "frame" else 30).fit(Xc, num_cols, cat_cols)
    Z = enc.transform(Xc)
    Zv = yv = None
    if eval_data is not None and fam in _BOOSTERS:
        Zv, yv = enc.transform(eval_data[0]), eval_data[1]
    cw = P.get("class_weight")
    best_iter = None
    inter, mono = None, None
    if fam in ("lgbm", "xgb", "hgb", "cat") and (P.get("inter_k") or P.get("mono")):
        struct = _detect_structure(Xc, y, num_cols, cat_cols, n_jobs)
        inter, mono = _constraints_for(P, struct, Z.shape[1])
        if fam == "cat":
            inter = None                                   # CatBoost: monotone constraints only
    if fam == "prior":
        est = DummyClassifier(strategy="prior").fit(Z, y)
    elif fam == "logreg":
        with _blas_limit():
            est = LogisticRegression(C=P["C"], class_weight=cw, max_iter=2000, solver="lbfgs").fit(Z, y)
    elif fam == "gam":
        prs, cslopes = _gam_terms(P, Xc, y, num_cols, cat_cols, n_jobs, enc)   # measured interaction terms
        est = SplineLogit(len(num_cols), C=P["C"], n_knots=P["knots"], basis=P.get("basis", "hinge"),
                          pairs=prs, cat_slopes=cslopes, penalty=P.get("penalty", "l2"),
                          pair_hinge=P.get("phinge", 0), cat_hinge=P.get("chinge", 0)).fit(Z, y)
    elif fam == "hgb":
        est = HistGradientBoostingClassifier(
            learning_rate=P["lr"], max_iter=P.get("n_estimators", 600), max_depth=P["depth"],
            max_leaf_nodes=_leaves_for(P["depth"], P["leaves"]), min_samples_leaf=P["min_leaf"],
            l2_regularization=P["l2"], max_features=P["colsample"], max_bins=255, class_weight=cw,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, scoring="loss",
            random_state=seed, categorical_features=enc.cat_idx_ if enc.cat_idx_ else None,
            interaction_cst=inter, monotonic_cst=mono).fit(Z, y)
    elif fam == "lgbm":
        n_trees = P.get("n_estimators") or _MAX_TREES[fam]
        est = lgb.LGBMClassifier(
            n_estimators=n_trees, learning_rate=P["lr"], num_leaves=_leaves_for(P["depth"], P["leaves"]),
            max_depth=P["depth"] if P["depth"] <= 8 else -1, min_child_samples=P["min_leaf"],
            subsample=P["subsample"], subsample_freq=1, colsample_bytree=P["colsample"],
            reg_lambda=P["l2"], reg_alpha=P.get("l1", 0.0), class_weight=cw, random_state=seed,
            n_jobs=n_jobs, verbose=-1, cat_smooth=20, min_data_per_group=50,
            # piecewise-linear leaves (not in the search: they helped P1 / P2 slightly but hurt the cost metric
            # around the decision threshold in the ensemble study)
            **({"linear_tree": True, "linear_lambda": 1.0} if P.get("linear") else {}),
            **({"interaction_constraints": inter} if inter else {}),
            **({"monotone_constraints": mono, "monotone_constraints_method": "advanced"} if mono else {}))
        kw = dict(categorical_feature=enc.cat_idx_ if enc.cat_idx_ else "auto")
        if Zv is not None:
            pat = int(_PATIENCE[fam] * max(1.0, 0.03 / P["lr"]))        # slower learning rates need more patience
            est.fit(Z, y, eval_set=[(Zv, yv)], callbacks=[lgb.early_stopping(pat, verbose=False)], **kw)
            best_iter = int(est.best_iteration_ or n_trees)
        else:
            est.fit(Z, y, **kw)
    elif fam == "cat":
        n_trees = P.get("n_estimators") or _MAX_TREES[fam]
        ckw = dict(iterations=n_trees, learning_rate=P["lr"], depth=P["depth"], l2_leaf_reg=P["l2"],
                   random_strength=P["random_strength"], bagging_temperature=P["bagging_temperature"],
                   random_seed=seed, thread_count=n_jobs, verbose=0, allow_writing_files=False)
        if cw:
            ckw["auto_class_weights"] = "Balanced"
        if enc.cat_idx_:
            ckw["cat_features"] = enc.cat_idx_
        if Zv is not None:
            ckw.update(od_type="Iter", od_wait=_PATIENCE[fam])
        if mono:
            ckw["monotone_constraints"] = [int(m) if j < len(num_cols) else 0 for j, m in enumerate(mono)]
        est = cb.CatBoostClassifier(**ckw)
        if Zv is not None:
            est.fit(Z, y, eval_set=(Zv, yv), use_best_model=True)
            best_iter = int(est.get_best_iteration() or 0) + 1
        else:
            est.fit(Z, y)
    elif fam == "xgb":
        n_trees = P.get("n_estimators") or _MAX_TREES[fam]
        spw = float((len(y) - y.sum()) / max(y.sum(), 1)) if cw else 1.0
        est = xgb.XGBClassifier(
            n_estimators=n_trees, learning_rate=P["lr"], max_depth=P["depth"], min_child_weight=P["mcw"],
            subsample=P["subsample"], colsample_bytree=P["colsample"], reg_lambda=P["l2"],
            reg_alpha=P.get("l1", 0.0), scale_pos_weight=spw, tree_method="hist", n_jobs=n_jobs,
            random_state=seed, verbosity=0, eval_metric="logloss",
            early_stopping_rounds=_PATIENCE[fam] if Zv is not None else None,
            **({"interaction_constraints": str([[int(i) for i in g] for g in inter])} if inter else {}),
            **({"monotone_constraints": "(" + ",".join(str(int(m)) for m in mono) + ")"} if mono else {}))
        if Zv is not None:
            est.fit(Z, y, eval_set=[(Zv, yv)], verbose=False)
            best_iter = int(est.best_iteration) + 1
        else:
            est.fit(Z, y)
    elif fam == "hyb":
        encl = TabularEncoder("linear", max_levels=30).fit(Xc, num_cols, cat_cols)
        Zl = encl.transform(Xc)
        prs, cslopes = _gam_terms(P, Xc, y, num_cols, cat_cols, n_jobs, encl)
        gam = SplineLogit(len(num_cols), C=P["C"], n_knots=P["knots"], basis="hinge", pairs=prs, cat_slopes=cslopes,
                          penalty="l1", pair_hinge=P.get("phinge", 0), cat_hinge=P.get("chinge", 0)).fit(Zl, y)
        g_tr = _logit(gam.predict_proba(Zl)[:, 1])
        n_trees = P.get("n_estimators") or _MAX_TREES[fam]
        booster = lgb.LGBMClassifier(
            n_estimators=n_trees, learning_rate=P["lr"], num_leaves=_leaves_for(P["depth"], P["leaves"]),
            max_depth=P["depth"], min_child_samples=P["min_leaf"], subsample=P["subsample"], subsample_freq=1,
            colsample_bytree=P["colsample"], reg_lambda=P["l2"], random_state=seed, n_jobs=n_jobs, verbose=-1,
            cat_smooth=20, min_data_per_group=50)
        kw = dict(categorical_feature=enc.cat_idx_ if enc.cat_idx_ else "auto", init_score=g_tr)
        if Zv is not None:
            g_v = _logit(gam.predict_proba(encl.transform(eval_data[0]))[:, 1])
            pat = int(_PATIENCE[fam] * max(1.0, 0.03 / P["lr"]))
            booster.fit(Z, y, eval_set=[(Zv, yv)], eval_init_score=[g_v],
                        callbacks=[lgb.early_stopping(pat, verbose=False)], **kw)
            best_iter = int(booster.best_iteration_ or n_trees)
        else:
            booster.fit(Z, y, **kw)
        est = GamBoost(gam, booster)
        enc = _PairEncoder(enc, encl)
    elif fam in ("rf", "et"):
        cls = RandomForestClassifier if fam == "rf" else ExtraTreesClassifier
        # leaves per tree are bounded (<= ~8000), which limits the size of a forest refit on a large table
        est = cls(n_estimators=P["n_estimators"], min_samples_leaf=max(P["min_leaf"], int(np.ceil(len(y) / 8000))),
                  max_features=P["max_features"], class_weight=cw, n_jobs=n_jobs,
                  random_state=seed).fit(Z, y)
    else:
        raise ValueError(fam)
    return FittedModel(spec, enc, est, best_iter)


# ---- structure detection (nested: computed on whatever training rows a model is fitted on) ------
_STRUCT_CACHE = {}


def _struct_key(Xc, y):
    """Cheap fingerprint of a training subset (same rows => same structure, so folds are cached)."""
    y = np.ascontiguousarray(np.asarray(y, dtype=np.int8))
    first = Xc.iloc[:, 0].to_numpy()
    try:
        fb = np.ascontiguousarray(first.astype(float)).tobytes()
    except Exception:
        fb = str(first[:50].tolist()).encode()
    return (len(Xc), Xc.shape[1], zlib.crc32(y.tobytes()), zlib.crc32(fb))


def _pd_shape(model, Zmed, j, qs):
    Zq = np.tile(Zmed, (len(qs), 1))
    Zq[:, j] = qs
    return model.predict(Zq, raw_score=True)


def _detect_structure(Xc, y, num_cols, cat_cols, n_jobs, max_pairs=32):
    """Measured structure of the training data, used as model constraints:
         pairs : feature-index pairs ranked by the split gain they share in a quick depth-2 boosting
                 model (which pairwise interactions the data asks for);
         mono  : +1 / -1 / 0 per feature, from the partial dependence of a quick additive (depth-1)
                 boosting model: a feature is monotone if its additive shape is nearly monotone.
       Feature indices refer to the native layout (numerics first, then categorical codes), which is
       the layout of every boosting encoder here.  Returns None if LightGBM is unavailable."""
    if not _HAS_LGBM:
        return None
    key = _struct_key(Xc, y)
    if key in _STRUCT_CACHE:
        return _STRUCT_CACHE[key]
    enc = TabularEncoder("native", max_levels=30).fit(Xc, num_cols, cat_cols)
    Z = enc.transform(Xc)
    p = Z.shape[1]
    out = {"pairs": [], "mono": [0] * p}
    cats = enc.cat_idx_ if enc.cat_idx_ else "auto"
    try:
        n_pre = min(300, max(60, len(y) // 20))
        m2 = lgb.LGBMClassifier(n_estimators=n_pre, learning_rate=0.05, num_leaves=4, max_depth=2,
                                min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                verbose=-1, n_jobs=n_jobs, random_state=0).fit(Z, y, categorical_feature=cats)
        gains = {}
        for tree in m2.booster_.dump_model()["tree_info"]:
            stack = [(tree["tree_structure"], ())]
            while stack:
                node, parents = stack.pop()
                if "split_feature" not in node:
                    continue
                f, g = node["split_feature"], float(node.get("split_gain", 0.0))
                for pf in parents:
                    if pf != f:
                        k = (min(pf, f), max(pf, f))
                        gains[k] = gains.get(k, 0.0) + g
                for ch in ("left_child", "right_child"):
                    if ch in node:
                        stack.append((node[ch], parents + (f,)))
        out["pairs"] = [list(k) for k, _ in sorted(gains.items(), key=lambda kv: -kv[1])[:max_pairs]]
    except Exception:
        pass
    try:
        m1 = lgb.LGBMClassifier(n_estimators=min(400, max(80, len(y) // 15)), learning_rate=0.1, num_leaves=2,
                                min_child_samples=20, verbose=-1, n_jobs=n_jobs, random_state=0
                                ).fit(Z, y, categorical_feature=cats)
        imp = m1.booster_.feature_importance("split")
        Zmed = np.nanmedian(Z, axis=0)
        Zmed = np.where(np.isfinite(Zmed), Zmed, 0.0)
        for j in range(len(num_cols)):                       # numerics only
            if imp[j] == 0:
                continue
            x = Z[:, j]
            x = x[np.isfinite(x)]
            if len(np.unique(x)) < 3:
                continue
            qs = np.quantile(x, np.linspace(0.02, 0.98, 25))
            s = _pd_shape(m1, Zmed, j, qs)
            if s.max() - s.min() < 1e-9:
                continue
            rho = pd.Series(qs).corr(pd.Series(s), method="spearman")
            d = np.diff(s)
            sign_changes = int(np.sum(np.sign(d[1:]) * np.sign(d[:-1]) < 0))
            if np.isfinite(rho) and abs(rho) > 0.9 and sign_changes <= 2:
                out["mono"][j] = int(np.sign(rho))
    except Exception:
        pass
    _STRUCT_CACHE[key] = out
    return out


def _constraints_for(P, struct, n_features):
    """(interaction groups or None, monotone list or None) requested by params P."""
    if not struct:
        return None, None
    k = int(P.get("inter_k", 0) or 0)
    inter = None
    if k > 0 and struct["pairs"]:
        inter = [[i] for i in range(n_features)] + [list(pr) for pr in struct["pairs"][:k]]
    mono = None
    if P.get("mono") and any(struct["mono"]):
        mono = list(struct["mono"][:n_features]) + [0] * max(0, n_features - len(struct["mono"]))
    return inter, mono


# ---- search spaces -----------------------------------------------------------
# Stage 1 of the search is a "complexity ladder": one configuration per tree depth, with the
# number of trees chosen by early stopping.  Stage 2 refines the other parameters at random
# around the best depth.  Depth 1 = additive model, depth 2 = pairwise interactions, ...
_LADDER = {
    # "inter_k" = number of measured feature pairs the trees may combine (0 = unconstrained),
    # "mono" = impose the measured monotone directions.  Both come from _detect_structure.
    "lgbm": [dict(depth=1, leaves=2, lr=0.05, min_leaf=20, subsample=1.0, colsample=1.0, l2=0.0),
             dict(depth=2, leaves=4, lr=0.03, min_leaf=20, subsample=0.8, colsample=0.8, l2=1.0),
             dict(depth=2, leaves=4, lr=0.03, min_leaf=20, subsample=0.8, colsample=0.8, l2=1.0, inter_k=8, mono=1),
             dict(depth=2, leaves=4, lr=0.03, min_leaf=20, subsample=0.8, colsample=0.8, l2=1.0, inter_k=24, mono=1),
             dict(depth=2, leaves=4, lr=0.01, min_leaf=20, subsample=0.8, colsample=0.8, l2=1.0, inter_k=8, mono=1),
             dict(depth=3, leaves=8, lr=0.03, min_leaf=20, subsample=0.8, colsample=0.8, l2=1.0),
             dict(depth=3, leaves=8, lr=0.03, min_leaf=20, subsample=0.8, colsample=0.8, l2=1.0, inter_k=16, mono=1),
             dict(depth=4, leaves=15, lr=0.03, min_leaf=40, subsample=0.8, colsample=0.7, l2=1.0),
             dict(depth=6, leaves=31, lr=0.02, min_leaf=40, subsample=0.8, colsample=0.7, l2=3.0)],
    "xgb": [dict(depth=1, lr=0.05, mcw=5.0, subsample=1.0, colsample=1.0, l2=1.0),
            dict(depth=2, lr=0.03, mcw=5.0, subsample=0.8, colsample=0.8, l2=1.0),
            dict(depth=2, lr=0.03, mcw=5.0, subsample=0.8, colsample=0.8, l2=1.0, inter_k=8, mono=1),
            dict(depth=3, lr=0.03, mcw=5.0, subsample=0.8, colsample=0.7, l2=1.0),
            dict(depth=3, lr=0.03, mcw=5.0, subsample=0.8, colsample=0.7, l2=1.0, inter_k=16, mono=1),
            dict(depth=4, lr=0.03, mcw=5.0, subsample=0.8, colsample=0.7, l2=1.0),
            dict(depth=6, lr=0.02, mcw=10.0, subsample=0.8, colsample=0.6, l2=3.0)],
    # CatBoost: symmetric trees; the monotone depth-3 rung was the most useful blend member on 2 of 3 public
    # problems in repeated CV (directions come from _detect_structure, so it is a no-op without monotone effects)
    "cat": [dict(depth=3, lr=0.03, l2=5.0, random_strength=1.0, bagging_temperature=1.0, mono=1),
            dict(depth=4, lr=0.05, l2=5.0, random_strength=1.0, bagging_temperature=1.0),
            dict(depth=6, lr=0.05, l2=3.0, random_strength=1.0, bagging_temperature=1.0)],
}


def _ladder(family, prof, obj):
    """Stage-1 configurations of a family (cheapest / simplest first)."""
    if family in _LADDER:
        return [dict(c, class_weight=None) for c in _LADDER[family]]
    if family == "hyb":
        return [dict(C=0.1, knots=15, pairs=4, cpairs=2, phinge=4, chinge=2, depth=2, leaves=4, lr=0.01, min_leaf=20,
                     subsample=0.8, colsample=0.8, l2=1.0, class_weight=None),
                dict(C=0.05, knots=15, pairs=6, cpairs=2, phinge=6, chinge=2, depth=2, leaves=4, lr=0.01, min_leaf=20,
                     subsample=0.8, colsample=0.8, l2=1.0, class_weight=None)]
    if family == "gam":
        return [dict(C=0.05, knots=7, basis="hinge", class_weight=None), dict(C=0.2, knots=7, basis="hinge", class_weight=None),
                dict(C=0.05, knots=7, basis="hinge", pairs=2, class_weight=None),
                dict(C=0.05, knots=7, basis="hinge", pairs=4, class_weight=None),
                dict(C=0.05, knots=7, basis="hinge", pairs=4, cpairs=2, class_weight=None),
                dict(C=0.05, knots=7, basis="hinge", pairs=2, cpairs=1, class_weight=None),
                dict(C=0.1, knots=15, basis="hinge", pairs=4, cpairs=2, penalty="l1", phinge=4, chinge=2,
                     class_weight=None),
                dict(C=0.1, knots=15, basis="hinge", pairs=6, cpairs=2, penalty="l1", phinge=6, chinge=2,
                     class_weight=None),
                dict(C=0.05, knots=15, basis="hinge", pairs=6, cpairs=2, penalty="l1", phinge=6, chinge=2,
                     class_weight=None)]
    if family == "logreg":
        return [dict(C=c, class_weight=None) for c in (0.03, 0.3)]
    if family == "hgb":
        base = [dict(depth=d, leaves=2 ** d, lr=0.05 if d < 3 else 0.03, min_leaf=30, l2=1.0, colsample=0.8,
                     class_weight=None) for d in (1, 2, 3, 5)]
        return base[:2] + [dict(base[1], inter_k=8, mono=1)] + base[2:]
    if family in ("et", "rf"):
        return [dict(n_estimators=300, min_leaf=ml, max_features=mf, class_weight=None) for ml, mf in ((5, 0.4), (15, 0.5))]
    return [{}]


def _sample_params(family, rng, best, obj=None, prof=None):
    """Random draw around the best stage-1 configuration (``best``) of the family."""
    weights = [None]
    if prof and prof["imbalanced"] and obj is not None and obj.metric in ("roc_auc", "average_precision", "f1",
                                                                          "balanced_accuracy"):
        weights = [None, "balanced"]           # class weighting only where ranking/labels matter
    lu = lambda a, b: float(np.exp(rng.uniform(np.log(a), np.log(b))))
    cw = rng.choice(weights)
    if family in ("lgbm", "xgb", "cat", "hgb"):
        d0 = int(best.get("depth", 3))
        lo, hi = (1, 8) if family != "cat" else (2, 8)
        depth = int(np.clip(d0 + rng.integers(-1, 2), lo, hi))
    if family in ("lgbm", "xgb", "hgb"):
        # structure flags: stay close to what won stage 1 (structured or not), but keep exploring
        if best.get("inter_k") or best.get("mono"):
            struct = dict(inter_k=int(rng.choice([4, 8, 12, 16, 24, 32])) if depth > 1 else 0,
                          mono=int(rng.random() < 0.8))
        else:
            struct = dict(inter_k=int(rng.choice([0, 0, 8, 16])) if depth > 1 else 0, mono=int(rng.random() < 0.3))
    if family == "lgbm":
        return dict(depth=depth, leaves=int(rng.integers(2, 2 ** min(depth, 6) + 1)), lr=lu(0.015, 0.08),
                    min_leaf=int(lu(5, 200)), subsample=float(rng.uniform(0.5, 1.0)),
                    colsample=float(rng.uniform(0.3, 1.0)), l2=lu(1e-3, 30), l1=lu(1e-3, 3), class_weight=cw,
                    **struct)
    if family == "xgb":
        return dict(depth=depth, lr=lu(0.015, 0.08), mcw=lu(1.0, 50.0), subsample=float(rng.uniform(0.5, 1.0)),
                    colsample=float(rng.uniform(0.3, 1.0)), l2=lu(1e-2, 30.0), l1=lu(1e-3, 3), class_weight=cw,
                    **struct)
    if family == "cat":
        return dict(depth=depth, lr=lu(0.025, 0.08), l2=lu(1.0, 30.0), random_strength=lu(0.2, 5.0),
                    bagging_temperature=float(rng.uniform(0.0, 1.5)), class_weight=cw,
                    mono=int(rng.random() < (0.7 if best.get("mono") else 0.3)))
    if family == "hgb":
        return dict(depth=depth, leaves=int(rng.integers(2, 2 ** min(depth, 6) + 1)), lr=lu(0.02, 0.1),
                    min_leaf=int(lu(5, 200)), l2=lu(1e-2, 30), colsample=float(rng.uniform(0.4, 1.0)),
                    class_weight=cw, **struct)
    if family == "gam":
        pairs = int(rng.choice([2, 4, 6] if best.get("pairs") else [0, 0, 2, 4]))
        cpairs = int(rng.choice([1, 2, 2, 4] if best.get("cpairs") else [0, 0, 1, 2]))
        if rng.random() < (0.6 if best.get("penalty") == "l1" else 0.3):      # sparse (L1) hinge GAM
            pairs, cpairs = max(pairs, 2), max(cpairs, 1)
            return dict(C=lu(0.05, 0.3), knots=int(rng.choice([12, 15, 20, 25])), basis="hinge", pairs=pairs,
                        cpairs=cpairs, penalty="l1", phinge=int(rng.choice([2, 4, pairs])),
                        chinge=int(rng.choice([0, 1, cpairs])), class_weight=None)
        return dict(C=lu(0.02, 1.0), knots=int(rng.choice([5, 7, 9, 12])),
                    basis=str(rng.choice(["hinge", "hinge", "spline"])), pairs=pairs, cpairs=cpairs, class_weight=None)
    if family == "logreg":
        return dict(C=lu(1e-3, 10), class_weight=cw)
    if family in ("rf", "et"):
        return dict(n_estimators=300, min_leaf=int(rng.integers(3, 40)),
                    max_features=float(rng.choice([0.2, 0.3, 0.5, 0.7])),
                    class_weight=rng.choice(["balanced_subsample" if family == "rf" else "balanced"] + [None] * 3)
                    if weights != [None] else None)
    return {}


# =============================================================================
# 5. Cross-validation helpers
# =============================================================================
def _folds(y, n_splits, seed):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def _oof_predict(spec, Xc, y, num_cols, cat_cols, folds, obj, n_jobs, early_stop=False):
    """OOF predictions; with ``early_stop`` the boosting families stop on the held-out fold and
    the mean best iteration is returned as well (used for tuning only)."""
    oof = np.zeros(len(y))
    iters = []
    for tr, va in folds:
        ev = (Xc.iloc[va], y[va]) if early_stop else None
        m = _fit_model(spec, Xc.iloc[tr], y[tr], num_cols, cat_cols, obj, n_jobs, eval_data=ev)
        oof[va] = m.predict(Xc.iloc[va])
        if m.best_iter is not None:
            iters.append(m.best_iter)
    return oof, (int(round(float(np.mean(iters)))) if iters else None)


def _logit(p):
    p = _clip(p)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class PlattCalibrator:
    """Sigmoid map on the logit of the blended probability (2 parameters)."""

    def fit(self, p, y):
        z = self._z(p)
        with _blas_limit():
            self.lr_ = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
        return self

    @staticmethod
    def _z(p):
        return _logit(p)[:, None]

    def predict(self, p):
        return self.lr_.predict_proba(self._z(p))[:, 1]


# =============================================================================
# 6. The framework entry point
# =============================================================================
class StudentSolution:
    """
    Parameters
    ----------
    time_budget : safety net (seconds) for the tuning phase on slow machines -- the search itself is a fixed
                  number of configurations; 600 s lets it finish in full on a laptop-class CPU (the fresh-fold re-scoring and the final
                  refit are extra).  Small datasets finish well within it.
    n_splits    : number of stratified CV folds.
    random_state: global seed.
    verbose     : print progress.
    """

    def __init__(self, time_budget=600, n_splits=5, n_repeats=1, bag_splits=5, random_state=42, verbose=False,
                 n_jobs=None, keep_oof=False):
        self.time_budget = time_budget
        self.keep_oof = keep_oof
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.bag_splits = bag_splits
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs if n_jobs else _N_CPU
        self.threshold_ = 0.5
        self.report_ = {}

    # ------------------------------------------------------------------ utils
    def _log(self, *a):
        if self.verbose:
            print("[framework]", *a, flush=True)

    @staticmethod
    def _blend_score(obj, y, logits):
        """Score of a blend given as a (members, n) logit matrix (higher = better).  Threshold
        metrics are optimised through log loss, a proper scoring rule that is far less noisy
        than the best-threshold cost on a finite OOF sample."""
        p = _sigmoid(logits.mean(0)) if logits.ndim == 2 else _sigmoid(logits)
        metric = obj.metric if obj.metric in ("roc_auc", "average_precision") else "log_loss"
        return _score_free(metric, y, p)

    # -------------------------------------------------------------------- fit
    def fit(self, X, y, metadata=None):
        t0 = time.time()
        _STRUCT_CACHE.clear()
        rng = np.random.default_rng(self.random_state)
        obj = Objective(metadata)
        self.objective_ = obj
        y = np.asarray(y).ravel()
        classes = np.unique(y[~pd.isna(y)])
        pos = obj.positive_label
        if pos is not None and pos in classes.tolist() and not set(classes.tolist()) <= {0, 1, False, True}:
            y = (y == pos).astype(int)                          # metadata says which label is "positive"
        elif not set(classes.tolist()) <= {0, 1, False, True}:  # map arbitrary two-class labels to {0,1}
            y = (y == classes[-1]).astype(int)
        y = y.astype(int)

        X = pd.DataFrame(X).reset_index(drop=True)
        self.cleaner_ = DataCleaner(obj.id_columns, obj.target_name).fit(X)
        Xc = self.cleaner_.transform(X)
        # measured redundancy: drop noisy copies of other columns, add averages of twin readings
        try:
            drop, twins = _find_redundant(Xc, self.cleaner_.numeric_, seed=self.random_state)
        except Exception as e:                                             # never let this kill fit
            self._log("redundancy check skipped", repr(e))
            drop, twins = [], []
        if drop or twins:
            self.cleaner_.set_redundancy(drop, twins)
            Xc = self.cleaner_.transform(X)
        self.report_["redundant_dropped"] = list(drop)
        self.report_["twin_features"] = [t["name"] for t in twins]
        num_cols, cat_cols = self.cleaner_.numeric_, self.cleaner_.categorical_
        prof = profile_data(Xc, y, self.cleaner_)
        self.profile_ = prof
        self._log("objective:", obj.describe())
        self._log("profile:", {k: v for k, v in prof.items() if k != "dropped"}, "dropped:", prof["dropped"])

        n = len(y)
        self.prior_ = float(np.clip(y.mean(), _EPS, 1 - _EPS))
        self.calibrator_ = None
        if Xc.shape[1] == 0 or len(np.unique(y)) < 2 or min(y.sum(), n - y.sum()) < 4:
            self.degenerate_ = True                       # nothing to learn: predict the prior
            self.threshold_ = 0.5
            return self
        self.degenerate_ = False

        # sub-sample for tuning on very large data (final fit still uses everything)
        if n > 60000:
            sub = rng.choice(n, 40000, replace=False)
            Xt, yt = Xc.iloc[sub].reset_index(drop=True), y[sub]
        else:
            Xt, yt = Xc, y
        n_splits = int(max(2, min(self.n_splits, yt.sum(), len(yt) - yt.sum())))
        folds_a = _folds(yt, n_splits, self.random_state)
        folds_b = _folds(yt, n_splits, self.random_state + 101)        # fresh folds for the final comparison

        # ---- baseline --------------------------------------------------------
        base = np.full(len(yt), yt.mean())
        self.report_["baseline_prior"] = {
            obj.tune_metric: _tuning_score(obj, yt, base),
            obj.metric: _selection_score(obj, yt, base)}

        # ---- candidate families ------------------------------------------------
        # the hinge GAM is used when there are numerics but not so many that its design matrix explodes
        families = (["gam" if 0 < len(num_cols) <= 60 else "logreg"]
                    + (["hyb"] if _HAS_LGBM and 0 < len(num_cols) <= 60 else [])
                    + (["lgbm"] if _HAS_LGBM else ["hgb"])
                    + (["xgb"] if _HAS_XGB else []) + (["cat"] if _HAS_CAT else []) + ["et"])
        # Search effort is a fixed number of configurations per family (ladder + random refinement), so
        # the result does not depend on how fast the machine is.  time_budget is only a safety net: on a
        # slow machine each family may spend its share of it on the refinement (unused time rolls over).
        share = {"gam": 0.15, "hyb": 0.12, "logreg": 0.05, "lgbm": 0.40, "hgb": 0.40, "xgb": 0.24, "cat": 0.24, "et": 0.06}
        max_iter = {"gam": 11, "hyb": 2, "logreg": 4, "lgbm": 32, "hgb": 20, "xgb": 18, "cat": 4, "et": 3}
        tuned = {}                                                       # family -> list of (score, spec, n_trees)
        budget_end = t0
        search_log = []
        for fam in families:
            # every family gets its own time window (unused time still rolls over), so one slow family can
            # never starve the ones after it; fam_hard caps the random refinements (the ladder always runs)
            fam_start = time.time()
            budget_end = max(budget_end, fam_start) + self.time_budget * share[fam]
            fam_hard = fam_start + 2.5 * self.time_budget * max(share[fam], 0.1)
            # own random stream per family: the search does not change when another family is
            # added or a library is missing
            frng = np.random.default_rng([self.random_state, zlib.crc32(fam.encode())])
            results = []
            ladder = _ladder(fam, prof, obj)
            for it in range(max_iter[fam]):
                now = time.time()
                if it < len(ladder):
                    # the ladder always runs (unless hopelessly over time) so every family has a
                    # sensible model; the refinement is what the time box controls
                    # ladder rungs are never cut by the family's time window: the validated library members come
                    # from them, so only the random refinements absorb time pressure on a slow machine (a far-out
                    # guard only stops a hopelessly slow run)
                    if it > 0 and now > t0 + 6 * self.time_budget:
                        break
                    params = ladder[it]
                else:
                    # the cheap linear families always get a few refinements (their time share is tiny, so a
                    # pure time box would make the result depend on the machine); the rest stop at their share
                    min_refine = {"gam": 2, "logreg": 2}.get(fam, 0)
                    if not results or now > fam_hard or (now > budget_end and it >= len(ladder) + min_refine):
                        break
                    params = _sample_params(fam, frng, best=max(results, key=lambda r: r[0])[1].params,
                                            obj=obj, prof=prof)
                spec = ModelSpec(fam, params, seed=self.random_state)
                try:
                    oof, n_trees = _oof_predict(spec, Xt, yt, num_cols, cat_cols, folds_a, obj, self.n_jobs,
                                                early_stop=True)
                    sc = _tuning_score(obj, yt, oof)
                except Exception as e:                                   # a failing config must never kill fit
                    self._log("config failed", spec.label(), repr(e))
                    continue
                results.append((sc, spec, n_trees))
                search_log.append((fam, sc, n_trees, spec.label()))
                self._log(f"{fam:5s} it{it:02d} {obj.tune_metric}={sc:.5f} trees={n_trees} t={time.time() - t0:5.0f}s")
            if results:
                results.sort(key=lambda r: -r[0])
                tuned[fam] = results
        self.report_["search"] = search_log
        if not tuned:                                                    # every candidate failed
            self.degenerate_ = True
            self.threshold_ = 0.5
            return self

        # ---- library: re-score the selected configurations as k-fold *bagged* members ----------
        # Each library configuration is refit on fresh folds with its tree count fixed (no early-stopping
        # optimism).  Its out-of-fold (OOF) predictions drive the blend, the calibration and the threshold,
        # and -- when the tuning used every row -- its k fold models ARE the final predictor (their logits
        # are averaged).  In repeated outer CV this k-fold bagging beat a single refit on all rows on all
        # three public problems, and it keeps the deployed predictor consistent with the OOF predictions
        # that the calibration and threshold were fitted on.
        bag = len(yt) == n
        R = max(1, int(getattr(self, "n_repeats", 1)))
        k_bag = int(max(2, min(getattr(self, "bag_splits", n_splits), yt.sum(), len(yt) - yt.sum())))
        fold_sets = [_folds(yt, k_bag, self.random_state + 101 + 7919 * r) for r in range(R)]
        members = []
        for fam, (sc0, spec0, n_trees) in self._library(tuned):
            params = dict(spec0.params)
            if n_trees is not None:
                params["n_estimators"] = int(max(20, n_trees))
            Zr = np.zeros((R, len(yt)))
            models = []
            try:
                for r, fs in enumerate(fold_sets):
                    for k, (tr, va) in enumerate(fs):
                        spec = ModelSpec(fam, params, seed=self.random_state + 1000 * r + k)
                        m = _fit_model(spec, Xt.iloc[tr], yt[tr], num_cols, cat_cols, obj, self.n_jobs)
                        Zr[r, va] = _logit(m.predict(Xt.iloc[va]))
                        if bag and fam not in _FORESTS:
                            models.append(m)
            except Exception as e:
                self._log("re-score failed", fam, repr(e))
                continue
            spec = ModelSpec(fam, params, seed=self.random_state)
            members.append({"spec": spec, "models": models, "Z": Zr})
            oof = _sigmoid(Zr.mean(0))
            self._log(f"library {fam:5s} {obj.metric}={_selection_score(obj, yt, oof):.5f} "
                      f"{obj.tune_metric}={_tuning_score(obj, yt, oof):.5f} {spec.label()[:70]} t={time.time() - t0:.0f}s")
        if not members:
            self.degenerate_ = True
            self.threshold_ = 0.5
            return self
        L = np.vstack([m["Z"].mean(0) for m in members])                 # members x n, OOF logits

        def _mname(s):
            return s.family + ("+struct" if (s.params.get("inter_k") or s.params.get("mono") or s.params.get("pairs")
                                             or s.params.get("cpairs")) else "")
        self.report_["family_scores"] = {}
        for m_, z in zip(members, L):
            nm = _mname(m_["spec"])
            s_ = _selection_score(obj, yt, _sigmoid(z))
            self.report_["family_scores"][nm] = max(s_, self.report_["family_scores"].get(nm, -np.inf))

        if self.keep_oof:                                                 # analysis only (not used to predict)
            self.oof_library_ = {"L": L, "y": yt, "names": [m_["spec"].label() for m_ in members]}
            self.library_ = members

        # ---- combiner: bagged ensemble selection ------------------------------------------------
        # Greedy (Caruana) selection with replacement, repeated on bootstrap resamples of the OOF rows
        # and the weights averaged (Caruana et al. 2006, "bagged ensemble selection").  A single greedy
        # pass over-fits the OOF rows it is run on; averaging over resamples keeps the members that are
        # useful on most of them.  On the outer-CV holdout of the three public problems it was first or
        # second on every problem, while a single Caruana pass, a logistic stack (free, possibly negative
        # weights) and equal weights + Platt were each last on at least one (equal weights is also unsafe
        # when a weak family is in the library).
        weights = self._bagged_caruana(obj, yt, L)
        self.intercept_ = 0.0
        self.combiner_ = "bagged_selection"
        blend = _sigmoid(weights @ L)
        self.report_["combiner"] = self.combiner_
        self.report_["blend_weights"] = {}
        for m_, w in zip(members, weights):
            if w != 0:
                nm = _mname(m_["spec"])
                self.report_["blend_weights"][nm] = round(self.report_["blend_weights"].get(nm, 0.0) + float(w), 4)
        self.report_["blend_oof_score"] = float(_selection_score(obj, yt, blend))
        self.report_["blend_oof_logloss"] = float(-_score_free("log_loss", yt, blend))

        # ---- optional calibration of the blend (monotone => ranking preserved) ----------------
        try:
            inner = _folds(yt, n_splits, self.random_state + 202)
            cal_oof = np.zeros(len(yt))
            for tr, va in inner:
                cal_oof[va] = PlattCalibrator().fit(blend[tr], yt[tr]).predict(blend[va])
            ll_raw, ll_cal = -_score_free("log_loss", yt, blend), -_score_free("log_loss", yt, cal_oof)
            self.report_["logloss_blend_raw"], self.report_["logloss_blend_calibrated"] = ll_raw, ll_cal
            if ll_cal < ll_raw - 1e-5:
                self.calibrator_ = PlattCalibrator().fit(blend, yt)
                blend = self.calibrator_.predict(blend)
        except Exception as e:
            self._log("calibration skipped", repr(e))

        # ---- threshold ------------------------------------------------------------
        if obj.uses_threshold:
            self.threshold_ = self._decision_threshold(yt, blend, obj, rng)
            thr, tp, fp, fn, tn = _conf_curve(yt, blend)
            sc = _threshold_metric_from_counts(obj.metric, tp, fp, fn, tn, obj)
            self.report_["oof_best_score_at_optimum"] = float(sc.max())
            i = int(np.argmin(np.abs(thr - self.threshold_)))
            self.report_["oof_score_at_chosen_threshold"] = float(sc[i])
            self.report_["oof_score_at_0.5"] = float(sc[int(np.argmin(np.abs(thr - 0.5)))])
            if obj.metric == "cost":
                cost_fn = lambda t: (obj.fp_cost * ((blend >= t) & (yt == 0)).sum()
                                     + obj.fn_cost * ((blend < t) & (yt == 1)).sum())
                self.report_["oof_cost_per_1000"] = {
                    "chosen_threshold": 1000 * cost_fn(self.threshold_) / len(yt),
                    "bayes_threshold": 1000 * cost_fn(obj.fp_cost / (obj.fp_cost + obj.fn_cost)) / len(yt),
                    "predict_all_negative": 1000 * obj.fn_cost * yt.sum() / len(yt),
                    "predict_all_positive": 1000 * obj.fp_cost * (len(yt) - yt.sum()) / len(yt)}
                self.report_["bayes_threshold"] = obj.fp_cost / (obj.fp_cost + obj.fn_cost)
        else:
            self.threshold_ = 0.5

        # ---- final predictor --------------------------------------------------------
        self.members_ = []                                               # [(coef, [fitted models])]
        # k-fold bagging: a selected member contributes its fold models (logits averaged).  Forests are
        # bagged ensembles already, so they (and every member when tuning ran on a subsample) are refit
        # once on ALL rows instead -- this also keeps the saved model small.
        grow = (1.0 + 0.5 / max(k_bag - 1, 1)) * min(1.5, (n / len(yt)) ** 0.5)
        for m_, w in zip(members, weights):
            if w == 0:
                continue
            if m_["models"]:
                self.members_.append((float(w), list(m_["models"])))
            else:
                spec = m_["spec"]
                params = dict(spec.params)
                if not bag and spec.family in _BOOSTERS and params.get("n_estimators"):
                    params["n_estimators"] = int(round(params["n_estimators"] * grow))
                try:
                    fm = _fit_model(ModelSpec(spec.family, params, seed=self.random_state), Xc, y, num_cols,
                                    cat_cols, obj, self.n_jobs)
                except Exception as e:
                    self._log("refit failed", spec.label(), repr(e))
                    continue
                self.members_.append((float(w), [fm]))
        if not self.members_:
            self.degenerate_ = True
            return self
        tw = float(sum(w for w, _ in self.members_))                    # convex weights sum to one
        self.members_ = [(w / tw, ms) for w, ms in self.members_]
        self.report_["final_mode"] = "kfold_bag" if bag else "refit"
        self.report_["selected"] = sorted({ms[0].spec.label() for _, ms in self.members_})
        self.report_["fit_seconds"] = time.time() - t0
        self._log("done in %.0fs; blend=%s threshold=%.4f" % (self.report_["fit_seconds"],
                                                                self.report_["blend_weights"], self.threshold_))
        return self

    # ------------------------------------------------------------ library / blend
    @staticmethod
    def _library(tuned):
        """Configurations offered to the ensemble: per family the best one and, for the boosting families,
        the best of the other kind (structured vs unconstrained, they make different errors) plus the
        runner-up.  Two GAM settings (cheap, and a different inductive bias from trees)."""
        quota = {"lgbm": 3, "hgb": 3, "xgb": 2, "cat": 2, "gam": 4, "hyb": 2, "logreg": 1, "et": 1, "rf": 1}
        structured = lambda r: bool(r[1].params.get("inter_k") or r[1].params.get("mono") or r[1].params.get("pairs")
                                    or r[1].params.get("cpairs"))
        lib = []
        for fam, res in tuned.items():
            picks = [res[0]]
            if fam in ("lgbm", "xgb", "hgb", "gam"):       # GAM: with vs without measured pair terms
                other = [r for r in res if structured(r) != structured(res[0])]
                if other:
                    picks.append(other[0])
            if fam == "gam":                               # and the best GAM of the other penalty (L1 vs L2)
                l1 = lambda r: r[1].params.get("penalty") == "l1"
                other = [r for r in res if l1(r) != l1(res[0]) and all(r is not p for p in picks)]
                if other:
                    picks.append(other[0])
            for r in res[1:]:
                if len(picks) >= quota.get(fam, 1):
                    break
                if all(r is not p for p in picks):
                    picks.append(r)
            lib += [(fam, r) for r in picks[:quota.get(fam, 1)]]
        return lib

    def _bagged_caruana(self, obj, y, L, n_bags=25, min_weight=0.02):
        """Bagged ensemble selection: Caruana selection on ``n_bags`` bootstrap resamples of the OOF rows,
        weights averaged; members below ``min_weight`` are dropped (they cost model size and fit time for
        no measurable gain) and the rest renormalised to a convex combination."""
        if len(L) == 1:
            return np.ones(1)
        rng = np.random.default_rng([self.random_state, 2006])
        n, w = len(y), np.zeros(len(L))
        done = 0
        for _ in range(n_bags):
            idx = rng.integers(0, n, n)
            if len(np.unique(y[idx])) < 2:
                continue
            w += self._caruana(obj, y[idx], L[:, idx])
            done += 1
        if done == 0:
            return self._caruana(obj, y, L)
        w /= done
        w[w < min_weight] = 0.0
        if w.sum() <= 0:
            return self._caruana(obj, y, L)
        return w / w.sum()

    def _caruana(self, obj, y, L, max_steps=40):
        """Greedy forward selection with replacement on the OOF logit matrix L (members x n)."""
        counts = np.zeros(len(L))
        singles = [self._blend_score(obj, y, L[i]) for i in range(len(L))]
        best_i = int(np.argmax(singles))
        counts[best_i] = 1
        cur = singles[best_i]
        for _ in range(max_steps - 1):
            sc = [self._blend_score(obj, y, ((counts + np.eye(len(L))[i]) / (counts.sum() + 1)) @ L)
                  for i in range(len(L))]
            j = int(np.argmax(sc))
            if sc[j] > cur + 1e-7:
                cur = sc[j]
                counts[j] += 1
            else:
                break
        return counts / counts.sum()

    # ----------------------------------------------------------- threshold
    def _decision_threshold(self, y, p, obj, rng, n_boot=400):
        """Decision threshold for the calibrated OOF probabilities ``p``.
        Cost metrics: the decision-theoretic threshold c_fp / (c_fp + c_fn) -- optimal when the probabilities are
        calibrated, which the blend + Platt step is fitted for.  The empirical alternative (OOF cost minimiser) is
        used only if it is better *out of sample*: in each bootstrap round it is fitted on the in-bag rows and both
        thresholds are scored on the out-of-bag rows; it must win in at least 95 % of the rounds (evidence that the
        probabilities are miscalibrated around the decision boundary).  On 30 synthetic cost problems the empirical
        rule alone cost 0.0017 more (normalized) than the Bayes rule.  Other decision metrics (F1, accuracy, ...)
        have no closed form here and use the empirical rule."""
        t_emp = self._robust_threshold(y, p, obj, rng)
        if obj.metric != "cost":
            return t_emp
        t_bayes = float(obj.fp_cost / (obj.fp_cost + obj.fn_cost))
        cost = lambda yy, pp, t: obj.fp_cost * np.sum((pp >= t) & (yy == 0)) + obj.fn_cost * np.sum((pp < t) & (yy == 1))
        n, wins, done = len(y), 0, 0
        sw = max(5, n // 400)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            oob = np.ones(n, dtype=bool)
            oob[idx] = False
            if oob.sum() < 30 or y[oob].sum() == 0 or y[idx].sum() == 0:
                continue
            t_b, _ = _best_threshold(y[idx], p[idx], obj, smooth_window=sw)
            wins += int(cost(y[oob], p[oob], t_b) < cost(y[oob], p[oob], t_bayes))
            done += 1
        rate = wins / done if done else 0.0
        self.report_["threshold_rule"] = {"bayes": t_bayes, "empirical": t_emp, "empirical_oob_win_rate": rate}
        return t_emp if rate >= 0.95 else t_bayes

    @staticmethod
    def _robust_threshold(y, p, obj, rng, n_boot=200):
        """Median of bootstrap optima + average of the smoothed optimum (reduces curve noise)."""
        t_full, _ = _best_threshold(y, p, obj, smooth_window=max(5, len(y) // 400))
        ts = []
        n = len(y)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            t, _ = _best_threshold(y[idx], p[idx], obj, smooth_window=max(5, n // 400))
            ts.append(t)
        return float(np.clip(0.5 * (np.median(ts) + t_full), 1e-4, 1 - 1e-4))

    # ---------------------------------------------------------- inference
    def predict_proba(self, X, metadata=None):
        X = pd.DataFrame(X).reset_index(drop=True)
        if getattr(self, "degenerate_", False) or not getattr(self, "members_", None):
            return np.full(len(X), self.prior_)
        Xc = self.cleaner_.transform(X)
        z = np.full(len(Xc), float(getattr(self, "intercept_", 0.0)))
        for w, models in self.members_:
            zm = np.zeros(len(Xc))
            for m in models:                                     # k-fold bag: average of fold-model logits
                pm = m.predict(Xc)
                zm += _logit(np.where(np.isfinite(pm), pm, self.prior_))
            z += w * zm / len(models)                            # members are combined in logit space
        p = _sigmoid(z)
        if self.calibrator_ is not None:
            p = self.calibrator_.predict(p)
        p = np.where(np.isfinite(p), p, self.prior_)            # never return NaN / inf
        return _clip(p)

    def get_threshold(self):
        return float(self.threshold_)

    # ------------------------------------------------------------- reporting
    def summary(self):
        r = dict(self.report_)
        r.pop("search", None)
        return {"objective": self.objective_.describe(), "profile": self.profile_, **r,
                "threshold": self.threshold_}
