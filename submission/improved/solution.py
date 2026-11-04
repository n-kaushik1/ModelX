"""
DA5401 Data Analytics Lab -- Course Project
Reusable end-to-end framework for binary tabular classification.

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
                           missingness, class balance, cardinality, size.
    2. preprocessing     : three model-specific encoders are built from the
                           profile (native-NaN for gradient boosting, imputed +
                           one-hot for forests, imputed + log/standardise +
                           one-hot for linear models).  Unseen / rare categories
                           are mapped to a "rare" bucket; the fitted encoder is
                           always re-applied to test data.
    3. candidates        : baseline (prior), logistic regression, histogram
                           gradient boosting, LightGBM and XGBoost (if
                           installed), ExtraTrees / RandomForest; CatBoost is
                           added only when a categorical has > 30 levels
                           (where its ordered target statistics help -- it was
                           slower and no better on low-cardinality data).
    4. validation        : stratified K-fold out-of-fold (OOF) predictions.
    5. tuning            : a default config plus a time-boxed random search per
                           family, scored with the problem's metric (or a
                           proper scoring rule when the metric needs a
                           threshold).
    6. selection         : best config per family is re-scored on fresh folds
                           and combined by greedy (Caruana) ensemble selection
                           on the metric -- a single model is chosen if blending
                           does not help.
    7. calibration       : a Platt (sigmoid) map fitted on OOF predictions is
                           kept only if nested CV shows a log-loss gain.  It is
                           monotone, so ranking metrics (AUC) are unaffected.
    8. threshold         : when the metric is decision based (cost, F1, ...)
                           the threshold is chosen on calibrated OOF
                           predictions, taking the median over bootstrap
                           resamples for stability.
    9. refit             : every selected member is refit on all supplied data
                           (several seeds for stochastic learners).
"""

from __future__ import annotations

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

try:  # optional -- XGBoost
    import xgboost as xgb
    _HAS_XGB = True
except Exception:  # pragma: no cover
    xgb = None
    _HAS_XGB = False

warnings.filterwarnings("ignore")

_EPS = 1e-4
_N_CPU = max(1, os.cpu_count() or 1)


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
        for c in X.columns:
            s = X[c]
            if c in self.extra_ids or _is_id_name(c) or c == self.target_name:
                self.dropped_["id"].append(c)
                continue
            if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_timedelta64_dtype(s):
                self.dropped_["unsupported"].append(c)
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
            if nn > 0 and conv.notna().sum() >= 0.95 * nn:
                self.numeric_.append(c)
                self.coerced_.append(c)
                continue
            if n >= 50 and s.nunique(dropna=True) >= 0.9 * nn:
                self.dropped_["unique_text"].append(c)
                continue
            self.categorical_.append(c)
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
        for c in self.numeric_:
            out[c] = pd.to_numeric(X[c], errors="coerce").astype(float) if c in X else np.nan
        for c in self.categorical_:
            out[c] = X[c].astype("object").where(X[c].notna(), None) if c in X else None
            out[c] = out[c].map(lambda v: None if v is None else str(v))
        out = out.replace([np.inf, -np.inf], np.nan)
        return out


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


_FAMILY_MODE = {"prior": "imputed", "logreg": "linear", "hgb": "native", "lgbm": "native",
                "rf": "imputed", "et": "imputed", "cat": "frame", "xgb": "native"}


class FittedModel:
    def __init__(self, spec, enc, est, kind="sk"):
        self.spec, self.enc, self.est, self.kind = spec, enc, est, kind

    def predict(self, Xc):
        Z = self.enc.transform(Xc)
        p = self.est.predict_proba(Z)[:, 1]
        return np.asarray(p, dtype=float)


def _fit_model(spec, Xc, y, num_cols, cat_cols, obj, n_jobs):
    fam, P, seed = spec.family, spec.params, spec.seed
    mode = _FAMILY_MODE[fam]
    # CatBoost copes with high-cardinality categoricals itself, so keep (many) more levels for it
    enc = TabularEncoder(mode, max_levels=1000 if mode == "frame" else 30).fit(Xc, num_cols, cat_cols)
    Z = enc.transform(Xc)
    cw = P.get("class_weight")
    if fam == "prior":
        est = DummyClassifier(strategy="prior").fit(Z, y)
    elif fam == "logreg":
        est = LogisticRegression(C=P["C"], class_weight=cw, max_iter=2000, solver="lbfgs").fit(Z, y)
    elif fam == "hgb":
        est = HistGradientBoostingClassifier(
            learning_rate=P["lr"], max_iter=P.get("max_iter", 600), max_leaf_nodes=P["leaves"],
            min_samples_leaf=P["min_leaf"], l2_regularization=P["l2"], max_features=P["colsample"],
            max_bins=255, class_weight=cw, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=25, scoring="loss", random_state=seed,
            categorical_features=enc.cat_idx_ if enc.cat_idx_ else None).fit(Z, y)
    elif fam == "lgbm":
        est = lgb.LGBMClassifier(
            n_estimators=P["n_estimators"], learning_rate=P["lr"], num_leaves=P["leaves"],
            min_child_samples=P["min_leaf"], subsample=P["subsample"], subsample_freq=1,
            colsample_bytree=P["colsample"], reg_lambda=P["l2"], class_weight=cw,
            random_state=seed, n_jobs=n_jobs, verbose=-1, cat_smooth=20, min_data_per_group=50,
        ).fit(Z, y, categorical_feature=enc.cat_idx_ if enc.cat_idx_ else "auto")
    elif fam == "cat":
        est = cb.CatBoostClassifier(
            iterations=P["iterations"], learning_rate=P["lr"], depth=P["depth"], l2_leaf_reg=P["l2"],
            random_strength=P["random_strength"], bagging_temperature=P["bagging_temperature"],
            auto_class_weights="Balanced" if cw else None, random_seed=seed, thread_count=n_jobs,
            verbose=0, allow_writing_files=False, cat_features=enc.cat_idx_ if enc.cat_idx_ else None,
        ).fit(Z, y)
    elif fam == "xgb":
        spw = float((len(y) - y.sum()) / max(y.sum(), 1)) if cw else 1.0
        est = xgb.XGBClassifier(
            n_estimators=P["n_estimators"], learning_rate=P["lr"], max_depth=P["depth"],
            min_child_weight=P["mcw"], subsample=P["subsample"], colsample_bytree=P["colsample"],
            reg_lambda=P["l2"], scale_pos_weight=spw, tree_method="hist", n_jobs=n_jobs,
            random_state=seed, verbosity=0).fit(Z, y)
    elif fam in ("rf", "et"):
        cls = RandomForestClassifier if fam == "rf" else ExtraTreesClassifier
        est = cls(n_estimators=P["n_estimators"], min_samples_leaf=P["min_leaf"],
                  max_features=P["max_features"], class_weight=cw, n_jobs=n_jobs,
                  random_state=seed).fit(Z, y)
    else:
        raise ValueError(fam)
    return FittedModel(spec, enc, est)


def _sample_params(family, rng, default=False, obj=None, prof=None):
    """Default config (first call) or a random draw from the family search space."""
    weights = [None]
    if prof and prof["imbalanced"] and obj is not None and obj.metric in ("roc_auc", "average_precision", "f1",
                                                                          "balanced_accuracy"):
        weights = [None, "balanced"]           # class weighting only where ranking/labels matter
    lu = lambda a, b: float(np.exp(rng.uniform(np.log(a), np.log(b))))
    if family == "logreg":
        return {"C": 0.1 if default else lu(1e-3, 10), "class_weight": None if default else rng.choice(weights)}
    if family == "hgb":
        if default:
            return dict(lr=0.06, leaves=15, min_leaf=40, l2=1.0, colsample=1.0, class_weight=None)
        return dict(lr=lu(0.02, 0.15), leaves=int(rng.integers(4, 48)), min_leaf=int(rng.integers(10, 150)),
                    l2=lu(0.01, 30), colsample=float(rng.uniform(0.4, 1.0)), class_weight=rng.choice(weights))
    if family == "lgbm":
        if default:
            return dict(n_estimators=300, lr=0.04, leaves=15, min_leaf=40, subsample=0.8, colsample=0.8,
                        l2=1.0, class_weight=None)
        return dict(n_estimators=int(rng.integers(100, 700)), lr=lu(0.015, 0.1), leaves=int(rng.integers(4, 48)),
                    min_leaf=int(rng.integers(10, 150)), subsample=float(rng.uniform(0.5, 1.0)),
                    colsample=float(rng.uniform(0.3, 1.0)), l2=lu(0.01, 30), class_weight=rng.choice(weights))
    if family == "cat":
        if default:
            return dict(iterations=500, lr=0.06, depth=5, l2=3.0, random_strength=1.0,
                        bagging_temperature=1.0, class_weight=None)
        return dict(iterations=int(rng.integers(250, 800)), lr=lu(0.03, 0.12), depth=int(rng.integers(3, 8)),
                    l2=lu(1.0, 30.0), random_strength=lu(0.2, 5.0),
                    bagging_temperature=float(rng.uniform(0.0, 1.5)), class_weight=rng.choice(weights))
    if family == "xgb":
        if default:
            return dict(n_estimators=400, lr=0.04, depth=4, mcw=5.0, subsample=0.8, colsample=0.8, l2=1.0,
                        class_weight=None)
        return dict(n_estimators=int(rng.integers(150, 800)), lr=lu(0.015, 0.1), depth=int(rng.integers(2, 9)),
                    mcw=lu(1.0, 30.0), subsample=float(rng.uniform(0.5, 1.0)),
                    colsample=float(rng.uniform(0.3, 1.0)), l2=lu(0.01, 30.0), class_weight=rng.choice(weights))
    if family in ("rf", "et"):
        if default:
            return dict(n_estimators=300, min_leaf=5, max_features=0.4, class_weight=None)
        return dict(n_estimators=300, min_leaf=int(rng.integers(1, 40)),
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


def _oof_predict(spec, Xc, y, num_cols, cat_cols, folds, obj, n_jobs):
    oof = np.zeros(len(y))
    for tr, va in folds:
        m = _fit_model(spec, Xc.iloc[tr], y[tr], num_cols, cat_cols, obj, n_jobs)
        oof[va] = m.predict(Xc.iloc[va])
    return oof


class PlattCalibrator:
    """Sigmoid map on the logit of the blended probability (2 parameters)."""

    def fit(self, p, y):
        z = self._z(p)
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
        return self

    @staticmethod
    def _z(p):
        p = _clip(p)
        return np.log(p / (1 - p))[:, None]

    def predict(self, p):
        return self.lr_.predict_proba(self._z(p))[:, 1]


# =============================================================================
# 6. The framework entry point
# =============================================================================
class StudentSolution:
    """
    Parameters
    ----------
    time_budget : seconds allowed for the tuning/validation phase (the final
                  refit is extra).  Small datasets finish well within it.
    n_splits    : number of stratified CV folds.
    random_state: global seed.
    verbose     : print progress.
    """

    def __init__(self, time_budget=150, n_splits=5, random_state=42, verbose=False, n_jobs=None):
        self.time_budget = time_budget
        self.n_splits = n_splits
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs if n_jobs else _N_CPU
        self.threshold_ = 0.5
        self.report_ = {}

    # ------------------------------------------------------------------ utils
    def _log(self, *a):
        if self.verbose:
            print("[framework]", *a, flush=True)

    # -------------------------------------------------------------------- fit
    def fit(self, X, y, metadata=None):
        t0 = time.time()
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
        num_cols, cat_cols = self.cleaner_.numeric_, self.cleaner_.categorical_
        prof = profile_data(Xc, y, self.cleaner_)
        self.profile_ = prof
        self._log("objective:", obj.describe())
        self._log("profile:", {k: v for k, v in prof.items() if k != "dropped"}, "dropped:", prof["dropped"])

        n = len(y)
        if Xc.shape[1] == 0 or len(np.unique(y)) < 2:    # degenerate: predict the prior
            self.degenerate_ = True
            self.prior_ = float(np.clip(y.mean(), _EPS, 1 - _EPS))
            self.threshold_ = 0.5
            return self
        self.degenerate_ = False
        self.prior_ = float(np.clip(y.mean(), _EPS, 1 - _EPS))

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

        # ---- candidate families (ordered by expected usefulness / cost) --------
        # CatBoost is slow (tens of seconds per fit) and was not better than LightGBM/XGBoost on
        # low-cardinality data; its ordered target statistics pay off when categoricals have many levels.
        use_cat = _HAS_CAT and prof["max_cardinality"] > 30
        families = (["logreg"] + (["lgbm"] if _HAS_LGBM else []) + (["xgb"] if _HAS_XGB else [])
                    + (["cat"] if use_cat else []) + ["hgb", "et", "rf"])
        # Fractions of time_budget per family.  They are absolute (not renormalised), so an optional
        # family adds search time instead of starving the others.
        share = {"logreg": 0.10, "lgbm": 0.30, "xgb": 0.25, "cat": 0.30, "hgb": 0.20, "et": 0.15, "rf": 0.15}
        if not _HAS_LGBM:
            share["hgb"] += 0.30
        max_iter = {"logreg": 8, "lgbm": 14, "xgb": 12, "cat": 4, "hgb": 10, "et": 5, "rf": 4}
        tuned = {}                                                       # family -> list of (score, spec)
        budget_end = t0
        search_log = []
        for fam in families:
            budget_end += self.time_budget * share[fam]
            # own random stream per family: the search does not change when another family is
            # added or a library is missing
            frng = np.random.default_rng([self.random_state, zlib.crc32(fam.encode())])
            results = []
            for it in range(max_iter[fam]):
                if it > 0 and time.time() > budget_end:
                    break
                params = _sample_params(fam, frng, default=(it == 0), obj=obj, prof=prof)
                spec = ModelSpec(fam, params, seed=self.random_state)
                try:
                    oof = _oof_predict(spec, Xt, yt, num_cols, cat_cols, folds_a, obj, self.n_jobs)
                    sc = _tuning_score(obj, yt, oof)
                except Exception as e:                                   # a failing config must never kill fit
                    self._log("config failed", spec.label(), repr(e))
                    continue
                results.append((sc, spec))
                search_log.append((fam, sc, spec.label()))
                self._log(f"{fam:7s} it{it:02d} {obj.tune_metric}={sc:.5f} t={time.time() - t0:5.0f}s")
            if results:
                results.sort(key=lambda r: -r[0])
                tuned[fam] = results
        self.report_["search"] = search_log

        # ---- re-score best config of each family on fresh folds ----------------
        members, oofs = [], []
        for fam, res in tuned.items():
            spec = res[0][1]
            oof = _oof_predict(spec, Xt, yt, num_cols, cat_cols, folds_b, obj, self.n_jobs)
            members.append(spec)
            oofs.append(oof)
            self._log(f"final-fold {fam:7s} {obj.metric}={_selection_score(obj, yt, oof):.5f}")
        O = np.vstack(oofs)
        member_scores = {s.family: _selection_score(obj, yt, o) for s, o in zip(members, oofs)}
        self.report_["family_scores"] = member_scores

        # ---- greedy ensemble selection (with replacement) -----------------------
        counts = np.zeros(len(members))
        best_i = int(np.argmax([_selection_score(obj, yt, o) for o in oofs]))
        counts[best_i] = 1
        cur = O[best_i].copy()
        cur_score = _selection_score(obj, yt, cur)
        for k in range(2, 13):
            cand = [((cur * (k - 1) + O[i]) / k) for i in range(len(members))]
            sc = [_selection_score(obj, yt, c) for c in cand]
            j = int(np.argmax(sc))
            if sc[j] > cur_score + 1e-5:
                cur, cur_score = cand[j], sc[j]
                counts[j] += 1
            else:
                break
        weights = counts / counts.sum()
        blend = np.tensordot(weights, O, axes=1)
        self.report_["blend_weights"] = {m.family: float(w) for m, w in zip(members, weights) if w > 0}
        self.report_["blend_oof_score"] = float(cur_score)

        # ---- optional calibration (monotone => ranking preserved) ----------------
        self.calibrator_ = None
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
            self.threshold_ = self._robust_threshold(yt, blend, obj, rng)
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

        # ---- refit chosen members on ALL data (several seeds for stochastic ones) --
        self.models_, self.model_weights_ = [], []
        for spec, w in zip(members, weights):
            if w <= 0:
                continue
            seeds = [self.random_state] if spec.family in ("logreg", "cat") else \
                [self.random_state + 17 * s for s in range(3 if n < 40000 else 1)]
            for s in seeds:
                sp = ModelSpec(spec.family, spec.params, seed=s)
                self.models_.append(_fit_model(sp, Xc, y, num_cols, cat_cols, obj, self.n_jobs))
                self.model_weights_.append(w / len(seeds))
        self.report_["selected"] = [m.spec.label() for m in self.models_]
        self.report_["fit_seconds"] = time.time() - t0
        self._log("done in %.0fs; blend=%s threshold=%.4f" % (self.report_["fit_seconds"],
                                                                self.report_["blend_weights"], self.threshold_))
        return self

    # ----------------------------------------------------------- threshold
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
        if getattr(self, "degenerate_", False):
            return np.full(len(X), self.prior_)
        Xc = self.cleaner_.transform(X)
        p = np.zeros(len(Xc))
        for m, w in zip(self.models_, self.model_weights_):
            p += w * m.predict(Xc)
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
