"""Phase 3 acceptance: each estimator recovers a known effect on a small synthetic dataset,
and the naive estimator recovers it only when there is no confounding."""
import numpy as np
import pandas as pd

from causallab import estimators as E
from causallab.synthetic import trap_scenarios


def _no_confounding(n=8000, seed=3):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"g": np.repeat(np.arange(n // 4), 4), "x": rng.normal(size=n)})
    d["t"] = (rng.random(n) < 0.4).astype(int)
    p0 = 1 / (1 + np.exp(-(-1.0 + 0.5 * d["x"])))
    p1 = 1 / (1 + np.exp(-(-1.0 + 0.5 * d["x"] + 0.5)))
    d["y"] = (rng.random(n) < np.where(d["t"] == 1, p1, p0)).astype(int)
    return d, float(np.mean(p1 - p0))


def test_adjusted_estimators_recover_confounded_effect(cfg):
    sc = trap_scenarios(n=12000, seed=2)[0]  # clean confounding
    out = E.estimate_all(sc.df, sc.treatment, sc.outcome, sc.reference_adjustment, sc.group, cfg)
    naive = out["estimates"]["naive"].estimate
    assert naive - sc.true_effect > 0.03, "the scenario should bias the naive comparison"
    for name in ("ipw", "aipw", "dml"):
        r = out["estimates"][name]
        assert abs(r.estimate - sc.true_effect) < 0.025, f"{name} missed: {r.estimate:.3f} vs {sc.true_effect:.3f}"


def test_naive_recovers_effect_without_confounding():
    d, truth = _no_confounding()
    r = E.naive(d["y"].to_numpy(float), d["t"].to_numpy(), d["g"].to_numpy())
    assert r.ci_low < truth < r.ci_high


def test_dml_matches_econml(cfg):
    sc = trap_scenarios(n=6000, seed=1)[0]
    out = E.estimate_all(sc.df, sc.treatment, sc.outcome, sc.reference_adjustment, sc.group, cfg, which=("dml",))
    r = out["estimates"]["dml"]
    assert abs(r.estimate - r.extra["econml_estimate"]) < 1e-6


def test_cluster_se_larger_when_rows_are_correlated():
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(500), 10)
    shared = np.repeat(rng.normal(size=500), 10)
    infl = shared + 0.2 * rng.normal(size=g.size)
    infl = infl - infl.mean()
    clustered = E.cluster_se(infl, g)
    iid = E.cluster_se(infl, np.arange(g.size))
    assert clustered > 2 * iid


def test_folds_never_split_a_group():
    g = np.repeat(np.arange(300), 7)
    folds = E.group_folds(g, 5, seed=1)
    assert (pd.Series(folds).groupby(g).nunique() == 1).all()
    for tr, te in E.fold_splits(folds):
        assert not set(g[tr]) & set(g[te])
