"""The five Q1 estimators, written as plain functions with one return type.

Conventions used everywhere:
  * Every model is cross-fitted with folds grouped by search, so rows from one search
    never sit on both sides of a fold.
  * Standard errors come from each estimator's influence function (its per-row score),
    summed within search before squaring. That is a cluster-robust standard error.
  * The estimand is the average treatment effect on the rows that survive overlap
    trimming, so every estimator below runs on the same trimmed rows.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm

Z95 = 1.959963984540054


class OverlapError(RuntimeError):
    """Raised when trimming leaves a fold without treated or control rows."""


@dataclass
class EstimateResult:
    """Common return type. Effects are in outcome units (probability for clicks)."""
    name: str
    estimate: float
    se: float
    n_rows: int
    n_groups: int
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ci_low(self) -> float:
        return self.estimate - Z95 * self.se

    @property
    def ci_high(self) -> float:
        return self.estimate + Z95 * self.se

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(ci_low=self.ci_low, ci_high=self.ci_high)
        return d


# ---------------------------------------------------------------- helpers

def group_folds(groups: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Assign every group (search) to one of k folds at random. Returns a fold id per row."""
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    fold_of_group = pd.Series(rng.permutation(len(uniq)) % k, index=uniq)
    return fold_of_group.loc[groups].to_numpy()


def fold_splits(folds: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    idx = np.arange(len(folds))
    return [(idx[folds != f], idx[folds == f]) for f in np.unique(folds)]


def cluster_se(influence: np.ndarray, groups: np.ndarray) -> float:
    """Cluster-robust standard error of a mean of scores.

    `influence` is each row's centred contribution (score minus the estimate). Summing
    within a search before squaring lets rows of one search be correlated."""
    n = len(influence)
    sums = pd.Series(influence).groupby(groups).sum().to_numpy()
    g = len(sums)
    return float(np.sqrt((sums ** 2).sum() * g / (g - 1)) / n)


def lgbm_params(cfg: dict, seed: int) -> dict:
    p = dict(cfg["models"]["lgbm"])
    p["random_state"] = seed
    return p


def impute_for_linear(X: pd.DataFrame, medians: pd.Series | None = None) -> pd.DataFrame:
    """Linear models need complete data. Missing indicators already exist as columns,
    so a median fill is enough here; LightGBM sees the raw missing values instead."""
    medians = X.median() if medians is None else medians
    return X.fillna(medians).fillna(0.0)


# ---------------------------------------------------------------- nuisance models

def crossfit_propensity(X: pd.DataFrame, t: np.ndarray, folds: np.ndarray, cfg: dict, seed: int):
    """Out-of-fold P(treated | covariates). Returns the scores and the fitted fold models."""
    e = np.zeros(len(t))
    models = []
    for tr, te in fold_splits(folds):
        m = lgb.LGBMClassifier(**lgbm_params(cfg, seed))
        m.fit(X.iloc[tr], t[tr])
        e[te] = m.predict_proba(X.iloc[te])[:, 1]
        models.append(m)
    return e, models


def crossfit_outcome(X: pd.DataFrame, y: np.ndarray, t: np.ndarray, folds: np.ndarray, cfg: dict, seed: int,
                     return_models: bool = False):
    """Out-of-fold predictions of the outcome under treatment (m1) and control (m0).
    Separate models per arm, so the treatment effect can vary with covariates.
    With return_models=True the fold models are returned too (used for the post-hoc
    population-matched comparison in the benchmark module)."""
    m1 = np.zeros(len(y))
    m0 = np.zeros(len(y))
    models = {1: [], 0: []}
    for tr, te in fold_splits(folds):
        for arm, target in ((1, m1), (0, m0)):
            rows = tr[t[tr] == arm]
            m = lgb.LGBMRegressor(**lgbm_params(cfg, seed))
            m.fit(X.iloc[rows], y[rows])
            target[te] = m.predict(X.iloc[te])
            models[arm].append(m)
    if set(np.unique(y)) <= {0, 1}:  # predicted probabilities must stay in [0, 1]
        m1, m0 = np.clip(m1, 0, 1), np.clip(m0, 0, 1)
    return (m1, m0, models) if return_models else (m1, m0)


def save_propensity(models: list, covariates: list[str], trim: tuple[float, float], path: Path) -> None:
    """Saved so the benchmark module can trim the experimental arm with the same rule."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "covariates": covariates, "trim": list(trim)}, path)


def save_outcome_models(models: dict, covariates: list[str], path: Path) -> None:
    """Saved so the benchmark module can carry the observational effect over to the
    randomly ranked arm's hotels (covariates only)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models1": models[1], "models0": models[0], "covariates": covariates}, path)


def score_effect(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    """Predicted effect m1(x) - m0(x) per row, averaged over the fold models. Each prediction is
    clipped to [0, 1], so this is for a binary outcome (clicks), the only place it is used."""
    X = df[bundle["covariates"]]
    m1 = np.clip(np.mean([m.predict(X) for m in bundle["models1"]], axis=0), 0, 1)
    m0 = np.clip(np.mean([m.predict(X) for m in bundle["models0"]], axis=0), 0, 1)
    return m1 - m0


def score_propensity(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    """Average of the fold models: each was trained on 4/5 of the observational searches."""
    X = df[bundle["covariates"]]
    return np.mean([m.predict_proba(X)[:, 1] for m in bundle["models"]], axis=0)


# ---------------------------------------------------------------- estimators

def naive(y: np.ndarray, t: np.ndarray, groups: np.ndarray, name: str = "naive") -> EstimateResult:
    """Difference in means. In the observational arm this mixes the position effect with
    the ranker's choice of better hotels for the top, so it shows the size of confounding."""
    p = t.mean()
    mu1, mu0 = y[t == 1].mean(), y[t == 0].mean()
    infl = t * (y - mu1) / p - (1 - t) * (y - mu0) / (1 - p)
    return EstimateResult(name, float(mu1 - mu0), cluster_se(infl, groups), len(y), len(np.unique(groups)),
                          {"mean_treated": float(mu1), "mean_control": float(mu0)})


def regression(y: np.ndarray, t: np.ndarray, X: pd.DataFrame, groups: np.ndarray):
    """Linear probability model with cluster-robust errors (statsmodels).
    The treatment coefficient reads directly in outcome units. The fitted model is
    returned too, because the sensitivity analysis is computed on it."""
    Xl = impute_for_linear(X)
    Xl = Xl.loc[:, Xl.std() > 0]  # constant columns (e.g. a fixed page size) add nothing and break the fit
    design = sm.add_constant(pd.concat([pd.Series(t, name="treatment", index=Xl.index), Xl], axis=1),
                             has_constant="add")
    fit = sm.OLS(y, design).fit(cov_type="cluster", cov_kwds={"groups": groups})
    res = EstimateResult("regression", float(fit.params["treatment"]), float(fit.bse["treatment"]),
                         len(y), len(np.unique(groups)))
    return res, fit, design


def ipw(y: np.ndarray, t: np.ndarray, e: np.ndarray, groups: np.ndarray) -> EstimateResult:
    """Inverse propensity weighting, normalised (Hajek) so weights sum to one per arm.
    The standard error treats the propensities as known, which is slightly conservative."""
    w1, w0 = t / e, (1 - t) / (1 - e)
    mu1 = (w1 * y).sum() / w1.sum()
    mu0 = (w0 * y).sum() / w0.sum()
    infl = w1 * (y - mu1) / w1.mean() - w0 * (y - mu0) / w0.mean()
    return EstimateResult("ipw", float(mu1 - mu0), cluster_se(infl, groups), len(y), len(np.unique(groups)),
                          {"max_weight": float(max(w1.max(), w0.max()))})


def aipw(y: np.ndarray, t: np.ndarray, e: np.ndarray, m1: np.ndarray, m0: np.ndarray,
         groups: np.ndarray) -> EstimateResult:
    """Doubly robust: outcome-model prediction plus an inverse-weighted correction of its
    errors. Consistent if either the propensity or the outcome model is right."""
    psi = m1 - m0 + t * (y - m1) / e - (1 - t) * (y - m0) / (1 - e)
    ate = psi.mean()
    return EstimateResult("aipw", float(ate), cluster_se(psi - ate, groups), len(y), len(np.unique(groups)),
                          {"mean_control": float(np.mean(m0))})


def dml(y: np.ndarray, t: np.ndarray, X: pd.DataFrame, folds: np.ndarray, groups: np.ndarray,
        cfg: dict, seed: int) -> EstimateResult:
    """Double machine learning (EconML LinearDML): regress the outcome residual on the
    treatment residual. With a constant effect this is the ATE; when effects vary it is a
    variance-weighted average, which is why AIPW, not DML, is the primary estimate."""
    from econml.dml import LinearDML

    model = LinearDML(
        model_y=lgb.LGBMRegressor(**lgbm_params(cfg, seed)),
        model_t=lgb.LGBMClassifier(**lgbm_params(cfg, seed)),
        discrete_treatment=True, cv=fold_splits(folds), allow_missing=True, random_state=seed,
    )
    model.fit(y, t, X=None, W=X, cache_values=True)
    y_res, t_res = (np.asarray(r).ravel() for r in model.residuals_[:2])
    theta = float((y_res * t_res).sum() / (t_res ** 2).sum())
    infl = (y_res - theta * t_res) * t_res / np.mean(t_res ** 2)
    return EstimateResult("dml", theta, cluster_se(infl, groups), len(y), len(np.unique(groups)),
                          {"econml_estimate": float(np.ravel(model.intercept_)[0])})


# ---------------------------------------------------------------- full pipeline

def estimate_all(df: pd.DataFrame, treatment: str, outcome: str, covariates: list[str], group: str,
                 cfg: dict, trim: tuple[float, float] | None = None, folds_k: int | None = None,
                 seed: int | None = None, propensity_path: Path | None = None, outcome_path: Path | None = None,
                 which: tuple[str, ...] = ("naive", "regression", "ipw", "aipw", "dml")) -> dict[str, Any]:
    """Run the estimators on one dataset.

    Steps: cross-fit the propensity model on all rows, trim rows outside the overlap range,
    then run every estimator on the trimmed rows."""
    seed = cfg["seed"] if seed is None else seed
    # Sort (and de-duplicate) the covariates: LightGBM samples columns by position, so the same
    # set in a different order would otherwise give slightly different estimates.
    covariates = sorted(dict.fromkeys(covariates))
    trim = tuple(cfg["overlap"]["trim"]) if trim is None else tuple(trim)
    folds_k = folds_k or cfg["models"]["cv_folds"]
    y_all = df[outcome].to_numpy(float)
    t_all = df[treatment].to_numpy(int)
    g_all = df[group].to_numpy()
    X_all = df[covariates]

    folds_all = group_folds(g_all, folds_k, seed)
    e_all, prop_models = crossfit_propensity(X_all, t_all, folds_all, cfg, seed)
    keep = (e_all >= trim[0]) & (e_all <= trim[1])
    if propensity_path is not None:
        save_propensity(prop_models, covariates, trim, propensity_path)

    y, t, g, e, folds = y_all[keep], t_all[keep], g_all[keep], e_all[keep], folds_all[keep]
    for f in np.unique(folds_all):
        tf = t[folds == f]
        if (tf == 1).sum() < 20 or (tf == 0).sum() < 20:
            raise OverlapError(f"after trimming to {trim}, fold {f} has {(tf == 1).sum()} treated and "
                               f"{(tf == 0).sum()} control rows ({1 - keep.mean():.0%} of rows trimmed)")
    X = X_all[keep].reset_index(drop=True)
    out: dict[str, Any] = {
        "trim": list(trim), "trimmed_share": float(1 - keep.mean()), "n_before_trim": int(len(keep)),
        "keep": keep, "propensity": e_all, "folds": folds, "estimates": {},
    }
    if "naive" in which:
        out["estimates"]["naive_untrimmed"] = naive(y_all, t_all, g_all, "naive_untrimmed")
        out["estimates"]["naive"] = naive(y, t, g)
    if "regression" in which:
        res, fit, design = regression(y, t, X, g)
        out["estimates"]["regression"] = res
        out["ols_fit"], out["ols_design"] = fit, design
    if "ipw" in which:
        out["estimates"]["ipw"] = ipw(y, t, e, g)
    if "aipw" in which:
        m1, m0, outcome_models = crossfit_outcome(X, y, t, folds, cfg, seed, return_models=True)
        out["estimates"]["aipw"] = aipw(y, t, e, m1, m0, g)
        if outcome_path is not None:
            save_outcome_models(outcome_models, covariates, outcome_path)
    if "dml" in which:
        out["estimates"]["dml"] = dml(y, t, X, folds, g, cfg, seed)
    return out


def primary_only(df: pd.DataFrame, treatment: str, outcome: str, covariates: list[str], group: str,
                 cfg: dict, **kw) -> EstimateResult:
    """Re-estimate with the primary estimator only. Used by the refuters."""
    name = cfg["models"]["primary_estimator"]
    return estimate_all(df, treatment, outcome, covariates, group, cfg, which=(name,), **kw)["estimates"][name]
