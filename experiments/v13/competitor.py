"""A strong 'typical competitor' framework for the synthetic suite (development reference only, never submitted):
Optuna (TPE) tuned LightGBM, 5-fold CV with early stopping on the objective, final model = the 5 fold models
averaged, Platt calibration on the out-of-fold logits for probability / decision objectives, and for decision
costs the threshold that minimises the out-of-fold cost.  COMP_TRIALS sets the number of Optuna trials (40)."""
import os, warnings
import numpy as np, pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


class StudentSolution:
    def __init__(self, n_jobs=4, n_trials=None, random_state=0):
        self.n_jobs, self.random_state = n_jobs, random_state
        self.n_trials = n_trials or int(os.environ.get("COMP_TRIALS", 40))

    def _prep(self, X):
        X = pd.DataFrame(X).copy()
        out = pd.DataFrame(index=X.index)
        for c in self.num_:
            out[c] = pd.to_numeric(X[c], errors="coerce") if c in X else np.nan
        for c in self.cat_:
            v = X[c].astype("object").where(X[c].notna(), None) if c in X else pd.Series([None] * len(X))
            out[c] = pd.Categorical(v.astype(str).where(v.notna(), None), categories=self.levels_[c])
        return out

    def fit(self, X, y, metadata):
        md = dict(metadata or {})
        self.obj_ = md.get("objective", "log_loss")
        self.cfp_ = md.get("cost_false_positive") or 1
        self.cfn_ = md.get("cost_false_negative") or 1
        X = pd.DataFrame(X).reset_index(drop=True)
        y = np.asarray(y).astype(int)
        self.num_ = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        self.cat_ = [c for c in X.columns if c not in self.num_]
        self.levels_ = {c: sorted(X[c].dropna().astype(str).unique().tolist()) for c in self.cat_}
        Xp = self._prep(X)
        folds = list(StratifiedKFold(5, shuffle=True, random_state=self.random_state).split(Xp, y))
        auc_obj = self.obj_ == "roc_auc"

        def cv(params, keep=False):
            oof = np.zeros(len(y)); models = []
            for tr, va in folds:
                dtr = lgb.Dataset(Xp.iloc[tr], y[tr]); dva = lgb.Dataset(Xp.iloc[va], y[va])
                m = lgb.train(params, dtr, 5000, valid_sets=[dva],
                              callbacks=[lgb.early_stopping(100, verbose=False)])
                oof[va] = m.predict(Xp.iloc[va], num_iteration=m.best_iteration)
                if keep:
                    models.append(m)
            return oof, models

        base = dict(objective="binary", metric="auc" if auc_obj else "binary_logloss", verbose=-1,
                    num_threads=self.n_jobs, seed=self.random_state, subsample_freq=1)

        def objective(trial):
            prm = dict(base,
                       learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                       num_leaves=trial.suggest_int("num_leaves", 4, 96, log=True),
                       max_depth=trial.suggest_categorical("max_depth", [-1, 3, 4, 6, 8]),
                       min_child_samples=trial.suggest_int("min_child_samples", 5, 200, log=True),
                       subsample=trial.suggest_float("subsample", 0.5, 1.0),
                       colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
                       reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
                       reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True))
            oof, _ = cv(prm)
            return roc_auc_score(y, oof) if auc_obj else -log_loss(y, np.clip(oof, 1e-7, 1 - 1e-7))

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.n_trials)
        self.params_ = dict(base, **study.best_params)
        oof, self.models_ = cv(self.params_, keep=True)
        self.platt_ = None
        if not auc_obj:
            self.platt_ = LogisticRegression(C=1e4).fit(_logit(oof)[:, None], y)
            oof = self.platt_.predict_proba(_logit(oof)[:, None])[:, 1]
        self.threshold_ = 0.5
        if self.obj_ == "decision_cost":
            ts = np.unique(np.round(oof, 4))
            cost = [self.cfp_ * np.sum((oof >= t) & (y == 0)) + self.cfn_ * np.sum((oof < t) & (y == 1)) for t in ts]
            self.threshold_ = float(ts[int(np.argmin(cost))])
        self.report_ = {"blend_weights": {"optuna_lgbm": 1.0}}
        return self

    def predict_proba(self, X, metadata=None):
        Xp = self._prep(X)
        p = np.mean([m.predict(Xp, num_iteration=m.best_iteration) for m in self.models_], axis=0)
        if self.platt_ is not None:
            p = self.platt_.predict_proba(_logit(p)[:, None])[:, 1]
        return p

    def get_threshold(self):
        return self.threshold_
