"""Diagnostics and refutation checks (Section 4.6).

Every check returns a dict with the measured value, the rule, and `passed`
(True / False, or None for checks that are reported but not pass/fail).
Thresholds come from config.yaml. Code makes every pass/fail decision; the LLM only
explains them afterwards.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .estimators import Z95, EstimateResult, OverlapError, cluster_se, estimate_all, impute_for_linear, primary_only


# ---------------------------------------------------------------- balance

MIN_OBSERVED = 300  # SMDs on fewer observed rows per group are noise, so they are skipped


def smd(df: pd.DataFrame, cols: list[str], treat: np.ndarray, weight: np.ndarray | None = None,
        min_observed: int = 2) -> dict[str, float]:
    """Standardised mean differences (treated minus control) over the pooled unweighted SD."""
    w = np.ones(len(df)) if weight is None else weight
    t = treat == 1
    out = {}
    for c in cols:
        x = df[c].to_numpy(float)
        ok = ~np.isnan(x)
        a, b = ok & t, ok & ~t
        if a.sum() < max(2, min_observed) or b.sum() < max(2, min_observed):
            continue
        sd = np.sqrt((x[a].var() + x[b].var()) / 2)
        if sd == 0:
            continue
        out[c] = float((np.average(x[a], weights=w[a]) - np.average(x[b], weights=w[b])) / sd)
    return out


def overlap(est: dict[str, Any], cfg: dict) -> dict[str, Any]:
    e = est["propensity"]
    t_share = est["trimmed_share"]
    limit = cfg["overlap"]["max_trimmed_share"]
    return {"check": "overlap", "trimmed_share": t_share, "trim": est["trim"],
            "propensity_quantiles": {q: float(np.quantile(e, q)) for q in (0.01, 0.05, 0.5, 0.95, 0.99)},
            "rule": f"share of rows outside {est['trim']} <= {limit}", "passed": bool(t_share <= limit)}


def balance(df_kept: pd.DataFrame, covariates: list[str], t: np.ndarray, e: np.ndarray, cfg: dict) -> dict[str, Any]:
    """SMDs after weighting with trimmed, stabilised inverse propensity weights."""
    p = t.mean()
    w = np.where(t == 1, p / e, (1 - p) / (1 - e))
    before = smd(df_kept, covariates, t, min_observed=MIN_OBSERVED)
    after = smd(df_kept, covariates, t, w, min_observed=MIN_OBSERVED)
    skipped = sorted(set(covariates) - set(after))
    limit = cfg["refuters"]["balance_max_smd"]
    if not after:  # no covariate has enough observed rows in both groups
        return {"check": "balance", "max_abs_smd_after": None, "worst_covariate": None, "smd_before": before,
                "smd_after": after, "failing": [], "skipped_too_few_observed": skipped,
                "rule": f"all |SMD| after weighting < {limit}", "passed": False}
    worst = max(after, key=lambda k: abs(after[k]))
    return {"check": "balance", "max_abs_smd_after": abs(after[worst]), "worst_covariate": worst,
            "smd_before": before, "smd_after": after,
            "failing": sorted(k for k, v in after.items() if abs(v) >= limit),
            "skipped_too_few_observed": skipped,
            "rule": f"all |SMD| after weighting < {limit}", "passed": bool(abs(after[worst]) < limit)}


# ---------------------------------------------------------------- re-estimation checks

def placebo(df: pd.DataFrame, treatment: str, outcome: str, covariates: list[str], group: str,
            cfg: dict) -> dict[str, Any]:
    """Permute the treatment across the whole sample and re-estimate; repeat for several
    permutations and average. Any 'effect' should vanish. This checks the code, not confounding.

    Design (post-hoc, 20 Nov 2026): the average of k permutation estimates is judged against
    +/- 1.96 single-run standard errors. Averaging removes most of one permutation's chance
    noise while the interval keeps single-run width, so chance false alarms become rare; a real
    leak bigger than about two standard errors is still caught. With k = 1 this is the original
    single-permutation rule.

    The spec shuffled within each search; that leaks part of the true effect when searches have
    different numbers of treated rows, so permutations are across the whole sample."""
    k = int(cfg["refuters"].get("placebo_reps", 1))
    ests, ses = [], []
    for i in range(k):
        rng = np.random.default_rng(cfg["seed"] + 1 + 1000 * i)
        d = df.copy()
        d[treatment] = rng.permutation(d[treatment].to_numpy())
        try:
            r = primary_only(d, treatment, outcome, covariates, group, cfg, folds_k=cfg["models"]["refuter_cv_folds"])
        except OverlapError as err:
            return {"check": "placebo", "estimate": None, "ci": [None, None], "error": str(err),
                    "rule": "placebo re-estimate must be possible", "passed": False}
        ests.append(r.estimate)
        ses.append(r.se)
    mean, se1 = float(np.mean(ests)), float(np.mean(ses))
    lo, hi = mean - Z95 * se1, mean + Z95 * se1
    rule = ("95% interval includes zero" if k == 1 else
            f"average of {k} permutation estimates within +/-1.96 single-run standard errors of zero")
    return {"check": "placebo", "estimate": mean, "ci": [lo, hi], "estimates": ests, "single_run_se": se1,
            "rule": rule, "passed": bool(lo <= 0 <= hi)}


def random_common_cause(df: pd.DataFrame, treatment: str, outcome: str, covariates: list[str], group: str,
                        cfg: dict, main: EstimateResult) -> dict[str, Any]:
    """Add a pure-noise covariate and re-estimate with the same settings as the main run."""
    rng = np.random.default_rng(cfg["seed"] + 2)
    d = df.copy()
    d["random_noise"] = rng.normal(size=len(d))
    limit = cfg["refuters"]["random_common_cause_max_shift_in_se"]
    try:
        r = primary_only(d, treatment, outcome, covariates + ["random_noise"], group, cfg)
    except OverlapError as err:
        # the re-estimate lost overlap: the estimate is not stable to an irrelevant covariate
        return {"check": "random_common_cause", "estimate": None, "shift_in_se": None, "error": str(err),
                "rule": f"estimate moves by < {limit} standard errors", "passed": False}
    shift = abs(r.estimate - main.estimate) / main.se
    return {"check": "random_common_cause", "estimate": r.estimate, "shift_in_se": float(shift),
            "rule": f"estimate moves by < {limit} standard errors", "passed": bool(shift < limit)}


def subset_stability(df: pd.DataFrame, treatment: str, outcome: str, covariates: list[str], group: str,
                     cfg: dict, main: EstimateResult) -> dict[str, Any]:
    """Re-estimate on random half-samples of searches (primary estimator only)."""
    rc = cfg["refuters"]
    rng = np.random.default_rng(cfg["seed"] + 3)
    groups = df[group].unique()
    ests = []
    for i in range(rc["subset_n"]):
        pick = rng.choice(groups, size=int(len(groups) * rc["subset_fraction"]), replace=False)
        d = df[df[group].isin(pick)]
        try:
            r = primary_only(d, treatment, outcome, covariates, group, cfg,
                             folds_k=cfg["models"]["refuter_cv_folds"], seed=cfg["seed"] + 100 + i)
            ests.append(r.estimate)
        except OverlapError:
            ests.append(None)  # a half-sample that loses overlap counts as outside the interval
    inside = float(np.mean([(x is not None and main.ci_low <= x <= main.ci_high) for x in ests]))
    return {"check": "subset_stability", "estimates": ests, "share_inside": inside,
            "rule": f"at least {rc['subset_min_share_inside']:.0%} of {rc['subset_n']} half-sample estimates "
                    "inside the full-sample 95% interval",
            "passed": bool(inside >= rc["subset_min_share_inside"])}


def held_out_covariate(df_kept: pd.DataFrame, treatment_values: np.ndarray, covariates: list[str], held_out: str,
                       groups: np.ndarray) -> dict[str, Any]:
    """Leave one attribute out and estimate the 'effect' of the treatment on it.
    Reported only: the ranker uses hotel attributes, so a nonzero value is expected.
    Columns derived from the held-out attribute (its missing flag, its value relative to the
    search) are left out with it."""
    from .features import DERIVED_FROM
    dropped = {held_out, *DERIVED_FROM.get(held_out, [])}
    others = [c for c in covariates if c not in dropped]
    y = df_kept[held_out].to_numpy(float)
    ok = ~np.isnan(y)
    X = sm.add_constant(pd.concat([pd.Series(treatment_values, name="treatment", index=df_kept.index),
                                   impute_for_linear(df_kept[others])], axis=1), has_constant="add")[ok]
    # On rows where the held-out value exists, some columns (e.g. its own missing flag) are
    # constant; drop them so the regression is well defined.
    X = X.loc[:, (X.std() > 0) | (X.columns == "const")]
    fit = sm.OLS(y[ok], X).fit(cov_type="cluster", cov_kwds={"groups": groups[ok]})
    sd = float(np.nanstd(y))
    return {"check": "held_out_covariate", "covariate": held_out, "also_left_out": sorted(dropped - {held_out}),
            "estimate": float(fit.params["treatment"]),
            "estimate_in_sd": float(fit.params["treatment"]) / sd if sd else None,
            "ci": [float(fit.conf_int().loc["treatment", 0]), float(fit.conf_int().loc["treatment", 1])],
            "rule": "reported, not pass or fail", "passed": None}


# ---------------------------------------------------------------- sensitivity

def _partial_r2(t_stat: float, dof: float) -> float:
    return t_stat ** 2 / (t_stat ** 2 + dof)


def robustness_value(t_stat: float, dof: float, q: float = 1.0) -> float:
    """Cinelli & Hazlett (2020): the partial R2 a confounder needs with both treatment and
    outcome to remove a share q of the estimate."""
    f = q * abs(t_stat) / np.sqrt(dof)
    return float(0.5 * (np.sqrt(f ** 4 + 4 * f ** 2) - f ** 2))


def ovb_bound(estimate: float, se: float, dof: float, r2dxj_x: float, r2yxj_dx: float,
              kd: float = 1.0, ky: float = 1.0) -> dict[str, float]:
    """Adjusted estimate if a hidden confounder were kd (ky) times as strong as covariate j
    in explaining the treatment (outcome). Closed form from Cinelli & Hazlett (2020)."""
    r2dz_x = kd * r2dxj_x / (1 - r2dxj_x)
    if r2dz_x >= 1:
        return {"r2dz_x": float(r2dz_x), "r2yz_dx": float("nan"), "bias": float("inf"), "adjusted_estimate": float("nan")}
    r2zxj_xd = kd * r2dxj_x ** 2 / ((1 - kd * r2dxj_x) * (1 - r2dxj_x))
    r2yz_dx = ((np.sqrt(ky) + np.sqrt(r2zxj_xd)) / np.sqrt(1 - r2zxj_xd)) ** 2 * r2yxj_dx / (1 - r2yxj_dx)
    r2yz_dx = min(r2yz_dx, 1.0)
    bias = se * np.sqrt(dof) * np.sqrt(r2yz_dx * r2dz_x / (1 - r2dz_x))
    adj = np.sign(estimate) * (abs(estimate) - bias)
    return {"r2dz_x": float(r2dz_x), "r2yz_dx": float(r2yz_dx), "bias": float(bias), "adjusted_estimate": float(adj)}


def e_value(rr: float) -> float | None:
    """VanderWeele & Ding E-value for a risk ratio. Defined only for a positive ratio: an effect
    that would push the implied rate to zero or below has no risk-ratio reading, so None."""
    if not np.isfinite(rr) or rr <= 0:
        return None
    rr = rr if rr >= 1 else 1 / rr
    return float(rr + np.sqrt(rr * (rr - 1)))


def sensitivity(y: np.ndarray, t: np.ndarray, X: pd.DataFrame, primary: EstimateResult, cfg: dict,
                groups: np.ndarray | None = None) -> dict[str, Any]:
    """Omitted-variable bias analysis on the linear regression (Cinelli & Hazlett, 2020).

    Each observed covariate in turn is the benchmark: how far would a hidden confounder as strong
    as it move the estimate? Fails if the strongest one reaches zero.

    The bounds and the robustness value are algebraic identities between variances and partial
    R2 values, computed from the classical regression; clustering does not change them, and
    plugging clustered t-statistics into them would be wrong. Clustering matters only for the
    uncertainty around the adjusted estimate, so with `groups` a cluster-robust 95% interval for
    the adjusted estimate is added (Cinelli & Hazlett's adjusted standard error, scaled from the
    cluster-robust standard error)."""
    Xl = impute_for_linear(X)
    Xl = Xl.loc[:, Xl.std() > 0]
    design = sm.add_constant(pd.concat([pd.Series(t, name="treatment", index=Xl.index), Xl], axis=1), has_constant="add")
    out_fit = sm.OLS(y, design).fit()
    trt_fit = sm.OLS(t.astype(float), sm.add_constant(Xl, has_constant="add")).fit()
    est, se, dof = float(out_fit.params["treatment"]), float(out_fit.bse["treatment"]), float(out_fit.df_resid)
    kd, ky = cfg["refuters"]["sensitivity_kd"], cfg["refuters"]["sensitivity_ky"]
    bounds = {}
    for c in Xl.columns:
        bounds[c] = ovb_bound(est, se, dof, _partial_r2(trt_fit.tvalues[c], trt_fit.df_resid),
                              _partial_r2(out_fit.tvalues[c], dof), kd, ky)
    strongest = max(bounds, key=lambda c: bounds[c]["bias"])
    b = bounds[strongest]
    adj = b["adjusted_estimate"]
    t_stat = est / se
    res = {"check": "sensitivity", "ols_estimate": est, "ols_se_classical": se,
           "partial_r2_treatment_outcome": _partial_r2(t_stat, dof),
           "robustness_value_q1": robustness_value(t_stat, dof),
           "strongest_benchmark_covariate": strongest, "bound_at_strongest": b, "bounds": bounds}
    if groups is not None and np.isfinite(adj):
        se_cl = float(sm.OLS(y, design).fit(cov_type="cluster", cov_kwds={"groups": groups}).bse["treatment"])
        se_adj = se_cl * np.sqrt((1 - b["r2yz_dx"]) / (1 - b["r2dz_x"]) * dof / (dof - 1))
        res.update(ols_se_cluster=se_cl, adjusted_ci_cluster=[float(adj - Z95 * se_adj), float(adj + Z95 * se_adj)])
    # E-value on the primary estimate, on the risk-ratio scale
    p0 = primary.extra.get("mean_control")
    ev = ev_ci = None
    if p0 and p0 > 0:
        ev = e_value((p0 + primary.estimate) / p0)
        lo = primary.ci_low if primary.estimate > 0 else primary.ci_high
        if ev is not None:
            ev_ci = 1.0 if np.sign(lo) != np.sign(primary.estimate) else e_value((p0 + lo) / p0)
    passed = bool(np.isfinite(adj) and np.sign(adj) == np.sign(est) and abs(adj) > 0)
    res.update(e_value=ev, e_value_ci=ev_ci,
               rule=f"effect keeps its sign with a hidden confounder kd={kd}, ky={ky} times the strongest covariate",
               passed=passed)
    return res


# ---------------------------------------------------------------- run all

def run_all(df: pd.DataFrame, est: dict[str, Any], treatment: str, outcome: str, covariates: list[str],
            group: str, held_out: str | None, cfg: dict) -> dict[str, dict[str, Any]]:
    """All checks for one estimation run. `est` is the output of estimate_all on `df`."""
    primary = est["estimates"][cfg["models"]["primary_estimator"]]
    keep = est["keep"]
    kept = df[keep].reset_index(drop=True)
    t = kept[treatment].to_numpy(int)
    y = kept[outcome].to_numpy(float)
    g = kept[group].to_numpy()
    e = est["propensity"][keep]
    out = {
        "overlap": overlap(est, cfg),
        "balance": balance(kept, covariates, t, e, cfg),
        "placebo": placebo(df, treatment, outcome, covariates, group, cfg),
        "random_common_cause": random_common_cause(df, treatment, outcome, covariates, group, cfg, primary),
        "subset_stability": subset_stability(df, treatment, outcome, covariates, group, cfg, primary),
        "sensitivity": sensitivity(y, t, kept[covariates], primary, cfg, groups=g),
    }
    if held_out and held_out in covariates:
        out["held_out_covariate"] = held_out_covariate(kept, t, covariates, held_out, g)
    return out


__all__ = ["run_all", "smd", "overlap", "balance", "placebo", "random_common_cause", "subset_stability",
           "held_out_covariate", "sensitivity", "ovb_bound", "robustness_value", "e_value", "cluster_se",
           "estimate_all"]
