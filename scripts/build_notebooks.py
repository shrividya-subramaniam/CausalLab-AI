"""Build the seven teaching notebooks in notebooks/. Run: python scripts/build_notebooks.py

Each notebook: a markdown cell before every code cell that introduces a concept (what it is,
why it is needed here, what output to expect), runs top to bottom on its own, and ends with
check-your-understanding questions and answers.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"

SETUP = '''import warnings
warnings.filterwarnings("ignore")
import json
import numpy as np
import pandas as pd
from causallab.config import load_config, load_spec, results_dir

cfg = load_config()
SYNTHETIC = False   # True = run on the synthetic Expedia-like file (no Kaggle data needed)
pd.set_option("display.width", 140)'''

SETUP_MD = ("## Setup\n\nImports and configuration. Every threshold and setting comes from `config.yaml`. "
            "Set `SYNTHETIC = True` to run on the synthetic file built by `python -m causallab prepare --synthetic`.")


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


def quiz(qas):
    body = "## Check your understanding\n\n"
    for i, (q, a) in enumerate(qas, 1):
        body += f"**{i}. {q}**\n\n<details><summary>Answer</summary>\n\n{a}\n\n</details>\n\n"
    return md(body)


NOTEBOOKS = {}

# ---------------------------------------------------------------- 01
NOTEBOOKS["01_eda"] = [
    md("""# 01 · Exploratory data analysis

**Goal:** check every fact the spec marks *verify in EDA* against the real file, and see why
each one shapes the design.

You will see: the size of the file, the share of randomly ranked searches, the selection on
clicks, how bookings differ by arm, the reserved positions, zero-coded values and price outliers.

Firewall note: this notebook shows click rates per **arm**, but never click rates by **position**
in the randomly ranked arm, because that would reveal the benchmark before the specification is fixed."""),
    md(SETUP_MD), code(SETUP),
    md("""## The Phase 1 checks

**What:** one query per *verify in EDA* item, run with DuckDB directly on the full Parquet file
(about 10 million rows) without loading it into memory.

**Why:** the spec says the data wins. Each finding below changed a design decision.

**Expect:** about 9.9 million rows, roughly 30% randomly ranked searches, zero searches without a
click, and four reserved positions."""),
    code('''from causallab.eda import run_eda
eda = run_eda(cfg, SYNTHETIC, save=False)  # computed here, not written over results/
for line in eda["findings"]:
    print("-", line)'''),
    md("""## Rows by position and arm

**What:** how many rows sit at each position, separately for algorithm-ranked (observational) and
randomly ranked (experimental) searches.

**Why:** positions that almost never appear in either arm are reserved for other listing types.
Comparing top-3 with a control group that includes them would compare different kinds of listing.

**Expect:** sharp dips at positions 5, 11, 17 and 23 in both arms."""),
    code('''pos = pd.DataFrame(eda["rows_by_position"]).set_index("position")
ax = pos.plot(kind="bar", figsize=(11, 3.5), title="Rows by position (dips = reserved slots)")
ax.set_ylabel("rows");'''),
    md("""## Selection: every search has a click

**What:** the share of searches with zero clicks, and the share with a booking, per arm.

**Why:** Expedia only released searches with at least one click, so every rate in this project is
*conditional on the search producing a click*. Bookings were sampled very differently in the two
arms, which is why click, not booking, is the primary outcome.

**Expect:** 0% zero-click searches in both arms; roughly 94% vs 13% of searches with a booking."""),
    code('''pd.DataFrame(eda["searches"]).assign(arm=lambda d: d.random_bool.map({0: "observational", 1: "experimental"}))'''),
    md("""## Building the sample and the features

**What:** keep 50.5% of searches (`sample_fraction` in `config.yaml`; whole searches, chosen by a hash of `srch_id`), then build features.

**Why:** rows in a search are not independent, so we always sample, split and bootstrap by search.
Within-search features (price rank, price relative to the page median, quality relative to the
page, number of results) are computed on the **whole page** before any position filter, because
the user saw the whole page and the ranker scored all of it.

**Expect:** about 5 million rows in about 200,000 searches across both arms. The cell prints the
algorithm-ranked arm only (about 3.5 million rows in 140,000 searches). Zero-coded states get their own flags.
The features file is built only if it is missing (`python -m causallab prepare` rebuilds it)."""),
    code('''from causallab import data, features
if not SYNTHETIC:
    sample = data.build_sample(cfg=cfg)          # skips if the file already exists
else:
    sample = data.sample_path(cfg, True)
fpath = data.features_path(cfg, SYNTHETIC)
if not fpath.exists():  # built once; rebuilding here would rewrite the file every result comes from
    features.build_features(sample, fpath, cfg)
obs = data.load_arm("observational", cfg, SYNTHETIC)
print(f"observational rows: {len(obs):,}  searches: {obs.srch_id.nunique():,}")
obs[["star_zero", "review_zero", "review_missing", "hist_price_zero", "loc2_missing", "affinity_missing"]].mean().round(3)'''),
    md("""## The Q1 analysis rows

**What:** positions 1-3 (treated) against positions 4-10 without the reserved slot 5 (control).

**Why:** keeping the control group to positions 4-10 means both arms compare the same mix of
positions; position 25 would be a very different control from position 4.

**Expect:** about 1.2 million rows, about a third of them treated."""),
    code('''q1 = features.q1_rows(obs)
print(f"Q1 rows: {len(q1):,}; treated share {q1.top3.mean():.1%}; positions used: {sorted(q1.position.unique())}")'''),
    quiz([
        ("Why is the sample drawn by search rather than by row?",
         "Rows of one search share the user, the query and the page, so they are correlated. Sampling rows would split searches and break the within-search features; standard errors must also be clustered by search."),
        ("Why are within-search features computed before positions 11+ are dropped?",
         "The user saw, and the ranker scored, the whole page. A price that is cheap relative to a 30-hotel page may look expensive relative to the top 10 only."),
        ("What does 'every search has at least one click' imply for the estimates?",
         "All click rates are conditional on the search producing a click. The selection also differs between arms because ranking changes which searches get a click, and no reweighting can undo selection on the outcome."),
        ("Why does a 0 star rating get its own indicator before being set to missing?",
         "0 means 'unrated', a real state that can differ from 'not recorded' (null). Merging them would hide a difference the ranker and users may respond to."),
    ]),
]

# ---------------------------------------------------------------- 02
NOTEBOOKS["02_benchmark"] = [
    md("""# 02 · The experimental benchmark

**Goal:** compute the effect of a top-3 position on clicks in randomly ranked searches, where
position cannot depend on hotel quality. This is the truth the observational estimates are scored
against.

The benchmark lives in its own module (`benchmark.py`), the only code allowed to read the randomly
ranked arm for Q1. The observational specification was fixed in `config/spec_locked.yaml` first."""),
    md(SETUP_MD), code(SETUP),
    md("""## Did randomisation work?

**What:** standardised mean differences (SMD) of hotel attributes between top-3 and positions 4-10,
inside the randomly ranked arm. SMD = difference in means divided by the pooled standard deviation.

**Why:** this is the causal counterpart of the sample-ratio check in an A/B test. If order is truly
random, hotel attributes cannot differ by position.

**Expect:** near zero for most attributes. Look closely at `prop_location_score2` and
`srch_query_affinity_score`: in this data they are **not** balanced. That is a finding, discussed below."""),
    code('''from causallab import benchmark
res = benchmark.run_untrimmed(cfg, SYNTHETIC, save=False)  # computed here, not written over results/
pd.Series(res["randomisation_balance_smd"]).sort_values(key=abs, ascending=False).round(3)'''),
    md("""## Why are two scores unbalanced in a randomised arm?

**What:** the mean of each score by position in the randomly ranked arm.

**Why:** if a score falls steadily with position even though position was random, two explanations
are possible. (1) *Selection on clicks*: the file keeps only searches with a click, and in a randomly
ordered search a click is more likely when good hotels happen to land on top, so the kept searches have
better hotels on top. (2) *Click-log leakage*: the score may be computed partly from clicks that include
these searches, which would make it partly an outcome. The next cell tests the first explanation.

**Expect:** location score 2 falling from position 1 to position 10; star rating, which cannot come
from click logs, falling slightly too."""),
    code('''from causallab import data, features
exp = features.q1_rows(data.load_arm("experimental", cfg, SYNTHETIC))
exp.groupby("position")[["prop_location_score2", "srch_query_affinity_score", "prop_starrating", "prop_review_score"]].mean().round(3)'''),
    md("""## Testing selection on clicks

**What:** the imbalance between top-3 and positions 4-10, split by whether the search also had a click
below position 10.

**Why:** a search with a click below position 10 would be in the file whatever hotels sat on top, so
selection on clicks cannot create imbalance there. If the imbalance shrinks in that group, selection
explains part of it.

**Expect:** on the real sample the location-score imbalance halves, from about 0.12 overall to 0.06 in
searches with a click below position 10 (0.16 in searches without one). Star rating is nearly balanced
(about 0.02) in every group."""),
    code('''r = data.load_arm("experimental", cfg, SYNTHETIC)
low = r[r.position > 10].groupby("srch_id").click_bool.max()
q = exp.assign(click_below=exp.srch_id.map(low).fillna(0).astype(int))
def smd(d, c):
    a, b = d.loc[d.top3 == 1, c].dropna(), d.loc[d.top3 == 0, c].dropna()
    return (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
groups = [("all searches", q), ("also a click below 10", q[q.click_below == 1]), ("no click below 10", q[q.click_below == 0])]
pd.DataFrame({name: {c: smd(d, c) for c in ["prop_location_score2", "srch_query_affinity_score", "prop_starrating"]}
              for name, d in groups}).round(3)'''),
    md("""## The benchmark with a cluster bootstrap

**What:** difference in click rate, top-3 minus positions 4-10, with a 95% interval from 1,000
bootstrap draws that resample **whole searches**.

**Why:** resampling rows would treat the hotels in a search as independent and make the interval too
narrow. To make 1,000 draws fast, each search is reduced to four numbers (treated rows, treated
clicks, control rows, control clicks) and the draws are vectorised.

**Expect:** about 5 percentage points."""),
    code('''b = res["benchmark"]
print(f"Benchmark: {b['estimate']*100:.2f} pp, 95% interval [{b['ci_low']*100:.2f}, {b['ci_high']*100:.2f}], "
      f"{b['n_searches']:,} searches")'''),
    md("""## Do the two arms describe the same searches?

**What:** search-level covariates compared across arms, and a benchmark reweighted so the randomly
ranked searches look like the algorithm-ranked ones.

**Why:** if the arms differ, the benchmark answers a question about a different population.
Reweighting on covariates fixes covariate differences, but not selection on the outcome (every search
has a click).

**Expect:** longer stays and longer booking windows in the randomly ranked arm; a reweighted
benchmark close to the original."""),
    code('''pd.DataFrame(res["cross_arm_search_covariates"]).T.round(2)'''),
    code('''w = res["benchmark_reweighted_to_observational"]
print(f"Reweighted benchmark: {w['estimate']*100:.2f} pp [{w['ci_low']*100:.2f}, {w['ci_high']*100:.2f}]")'''),
    md("""## Click rate by position, both arms

**What:** the figure saved by the benchmark module.

**Why:** the gap between the two curves at the top is the confounding the observational estimators
must remove: in algorithm-ranked searches, top positions hold better hotels.

**Expect:** the observational curve above the experimental one at positions 1-4 and below it further
down: the ranker moves good hotels up and poor ones down. The dotted lines mark the reserved slots."""),
    code('''from IPython.display import Image
fig = results_dir(cfg) / "figures" / ("click_by_position_synthetic.png" if SYNTHETIC else "click_by_position.png")
if not fig.exists():  # `python -m causallab benchmark` saves it; draw it here only if missing
    benchmark.click_by_position_figure(cfg, SYNTHETIC)
Image(filename=str(fig))'''),
    quiz([
        ("Why must the benchmark bootstrap resample searches, not rows?",
         "Clicks within a search are correlated (same user, same query, one click often replaces another). Row resampling ignores that and understates the uncertainty."),
        ("Why was the observational specification written down before this notebook was run?",
         "So the specification cannot be tuned to match the benchmark. Any change after seeing the benchmark is labelled exploratory."),
        ("What does an unbalanced attribute in a randomised arm tell you?",
         "Either randomisation failed, or the rows were selected after randomisation, or the attribute is not truly pre-treatment. Here selection on clicks explains about half of the imbalance (it halves in searches that also had a click below position 10, which would be in the file regardless); the rest may be click-log leakage or other selection."),
        ("What can reweighting the experimental arm fix, and what can it not fix?",
         "It can fix differences in observed search covariates between arms. It cannot fix selection on the outcome, such as the file only containing searches with a click."),
    ]),
]

# ---------------------------------------------------------------- 03
NOTEBOOKS["03_estimators"] = [
    md("""# 03 · The five observational estimators

**Goal:** estimate the top-3 effect from algorithm-ranked searches only, where the ranker put
better hotels on top, and see how much of the confounding each method removes.

Inverse propensity weighting and AIPW are first written by hand in a few lines of NumPy, so the
mechanics are visible, then run through the library code."""),
    md(SETUP_MD), code(SETUP),
    md("""## The data and the reference specification

**What:** the Q1 rows from the observational arm, and the adjustment set fixed in Phase 1.

**Why:** the adjustment set holds variables that drive both the ranker's placement and clicks:
hotel attributes, price and its rank on the page, promotion, query match and search context.

**Expect:** about 1.2 million rows and 44 adjustment variables."""),
    code('''from causallab import data, estimators as E, features
spec = load_spec(cfg)
df = features.q1_rows(data.load_arm("observational", cfg, SYNTHETIC))
cov = spec["adjustment_set"]
y, t, g = df.click_bool.to_numpy(float), df.top3.to_numpy(int), df.srch_id.to_numpy()
print(f"{len(df):,} rows, {len(cov)} adjustment variables")'''),
    md("""## 1. The naive comparison

**What:** click rate of top-3 hotels minus click rate of positions 4-10.

**Why:** it shows how big the confounding is: it mixes the effect of position with the ranker's
choice of better hotels for the top.

**Expect:** about 10 points, roughly double the benchmark."""),
    code('''naive = E.naive(y, t, g)
print(f"naive: {naive.estimate*100:.2f} pp  [{naive.ci_low*100:.2f}, {naive.ci_high*100:.2f}]")'''),
    md("""## 2. The propensity score, cross-fitted by search

**What:** P(top-3 | covariates), from LightGBM. Cross-fitting means each row's score comes from a
model trained on the *other* folds, and folds are groups of whole searches.

**Why:** a score from a model that saw the row overfits; a fold that splits a search leaks
information between training and test rows.

**Expect:** most scores well inside 0 and 1 (on the real sample the largest is about 0.98), so
trimming at 0.01 and 0.99 removes almost no rows."""),
    code('''folds = E.group_folds(g, cfg["models"]["cv_folds"], cfg["seed"])
e, _ = E.crossfit_propensity(df[cov], t, folds, cfg, cfg["seed"])
print(pd.Series(e).describe().round(3))'''),
    md("""## 3. Inverse propensity weighting by hand

**What:** weight each treated row by 1/e and each control row by 1/(1-e), normalise the weights,
compare weighted means.

**Why:** a top-3 hotel that the ranker was *unlikely* to put on top looks like the controls, so it
gets a large weight. Weighting builds a pseudo-population where placement no longer depends on the
covariates.

**Expect:** a big drop from the naive estimate."""),
    code('''e = np.clip(e, 0.01, 0.99)  # never divide by 0; the library trims rows outside [0.01, 0.99] instead
w1, w0 = t / e, (1 - t) / (1 - e)
ipw_by_hand = (w1 * y).sum() / w1.sum() - (w0 * y).sum() / w0.sum()
print(f"IPW by hand: {ipw_by_hand*100:.2f} pp")'''),
    md("""## 4. AIPW (doubly robust) by hand

**What:** predict each row's click probability under top-3 (m1) and under control (m0), then
correct the prediction errors with inverse weights:

`psi = m1 - m0 + t*(y - m1)/e - (1-t)*(y - m0)/(1-e)`; the estimate is the mean of `psi`.

**Why:** consistent if *either* the outcome model or the propensity model is right. The same `psi`
gives the standard error: sum it within each search, then square (a clustered standard error).

**Expect:** close to the IPW estimate, with a slightly narrower interval."""),
    code('''m1, m0 = E.crossfit_outcome(df[cov], y, t, folds, cfg, cfg["seed"])
psi = m1 - m0 + t * (y - m1) / e - (1 - t) * (y - m0) / (1 - e)
ate = psi.mean()
se = E.cluster_se(psi - ate, g)
print(f"AIPW by hand: {ate*100:.2f} pp, clustered SE {se*100:.2f}")'''),
    md("""## 5. All five estimators through the library

**What:** naive, regression adjustment (linear probability model, statsmodels, clustered errors),
IPW, AIPW and double machine learning (EconML), all on the same trimmed rows.

**Why:** DML regresses the outcome residual on the treatment residual. When the effect varies
across hotels it estimates a *variance-weighted* average, not the average effect, which is why AIPW
is the primary estimate.

**Expect:** naive > regression > IPW ≈ AIPW ≈ DML; a few minutes on a laptop."""),
    code('''out = E.estimate_all(df, "top3", "click_bool", cov, "srch_id", cfg)
table = pd.DataFrame({k: {"estimate_pp": v.estimate*100, "ci_low": v.ci_low*100, "ci_high": v.ci_high*100}
                      for k, v in out["estimates"].items()}).T.round(2)
table'''),
    md("""## How much of the confounding did adjustment remove?

**What:** compare with the benchmark computed in notebook 02.

**Why:** this is the within-study comparison: the gap between the observational estimate and the
experiment is the headline result, whatever it is.

**Expect:** most, but not all, of the naive gap removed."""),
    code('''p = results_dir(cfg) / ("benchmark_synthetic.json" if SYNTHETIC else "benchmark.json")
if not p.exists():  # notebook 02 (or `python -m causallab benchmark`) writes it; build it here if not
    from causallab.benchmark import run_untrimmed
    run_untrimmed(cfg, SYNTHETIC)
bench = json.loads(p.read_text(encoding="utf-8"))["benchmark"]
a, n, b = out["estimates"]["aipw"].estimate, out["estimates"]["naive"].estimate, bench["estimate"]
print(f"benchmark {b*100:.2f} pp | naive {n*100:.2f} | AIPW {a*100:.2f}")
print(f"share of the naive gap removed: {(n - a) / (n - b):.0%}")'''),
    md("""## Sensitivity check: a wider comparison group

**What:** the spec's robustness check: compare the top 3 with every non-reserved position from 4 down,
not just 4-10, with the same specification, trimming and gap test. The benchmark module recomputes the
benchmark for the same window.

**Why:** the 4-10 window is a choice. Widening it changes both sides (lower hotels are worse, and clicks
keep falling with position), so the question is whether the verdict holds, not whether the number stays
the same.

**Expect:** more rows trimmed than in the main run, because hotels far down the page rarely have any
chance of a top-3 slot; the overlap line says whether the comparison is usable."""),
    code('''# `python -m causallab estimate --all-positions` saves this check; reuse the file, or run it if missing
from causallab.sensitivity_window import run_all_positions
p = results_dir(cfg) / ("sensitivity_all_positions_synthetic.json" if SYNTHETIC else "sensitivity_all_positions.json")
sw = json.loads(p.read_text(encoding="utf-8")) if p.exists() else run_all_positions(cfg, SYNTHETIC)
print(f"overlap {'pass' if sw['overlap']['passed'] else 'FAIL'} ({sw['overlap']['trimmed_share']:.1%} trimmed), "
      f"benchmark {sw['benchmark']['estimate'] * 100:.2f} pp")
pd.DataFrame.from_dict({k: {"estimate_pp": v["estimate"] * 100, "gap_pp": sw["gap"][k]["gap"] * 100,
                            "verdict": sw["gap"][k]["verdict"]} for k, v in sw["estimates"].items()}, orient="index").round(2)'''),
    quiz([
        ("Why is the naive comparison biased upward here?",
         "The ranker places hotels it expects to be clicked at the top, so top-3 hotels are better on average. Their extra clicks are partly quality, not position."),
        ("What does 'doubly robust' mean for AIPW?",
         "The estimate is consistent if either the outcome model or the propensity model is correctly specified (not necessarily both)."),
        ("Why are folds made of whole searches?",
         "Rows in a search are correlated. A fold that splits a search lets the model learn about test rows from their search-mates, which defeats cross-fitting."),
        ("Why can DML and AIPW give different numbers on the same data?",
         "AIPW targets the average treatment effect. DML's residual-on-residual regression weights units by the variance of their treatment residual, so with heterogeneous effects it targets a weighted average."),
        ("If all five estimators agree but miss the benchmark, what is the most likely explanation?",
         "A confounder none of them sees, such as information the ranker used that is not in the file, or a covariate that is partly an outcome. Agreement between estimators says the modelling is stable, not that the adjustment set is right."),
    ]),
]

# ---------------------------------------------------------------- 04
NOTEBOOKS["04_refuters"] = [
    md("""# 04 · Trying to break the answer

**Goal:** run the checks of Section 4.6 on the Q1 estimate. Code makes every pass/fail decision with
thresholds from `config.yaml`; nothing here is judged by eye.

To keep this notebook quick, the half-sample check uses 10 subsets instead of 20. The full-strength
version is `python -m causallab refute`."""),
    md(SETUP_MD), code(SETUP + '''
cfg["refuters"]["subset_n"] = 10   # notebook speed only; the CLI uses the config value (20)'''),
    md("""## Estimate first

**What:** the five estimators on the reference specification, as in notebook 03.

**Expect:** a few minutes."""),
    code('''from causallab import data, estimators as E, features, refuters as R
spec = load_spec(cfg); cov = spec["adjustment_set"]
df = features.q1_rows(data.load_arm("observational", cfg, SYNTHETIC))
est = E.estimate_all(df, "top3", "click_bool", cov, "srch_id", cfg)
primary = est["estimates"]["aipw"]
kept = df[est["keep"]].reset_index(drop=True)
t, y, e = kept.top3.to_numpy(), kept.click_bool.to_numpy(float), est["propensity"][est["keep"]]
print(f"AIPW {primary.estimate*100:.2f} pp")'''),
    md("""## Overlap

**What:** the distribution of propensity scores, and the share of rows outside [0.01, 0.99].

**Why:** if some hotels are almost certain to be (or not to be) on top, there is no comparable
control for them, and weights explode.

**Expect:** no rows trimmed: the ranker is far from deterministic given the covariates."""),
    code('''R.overlap(est, cfg)'''),
    md("""## Covariate balance after weighting

**What:** SMDs of every adjustment variable between top-3 and control, after weighting with trimmed,
stabilised inverse propensity weights.

**Why:** if weighting worked, the weighted groups look alike on everything we adjusted for.

**Expect:** all below 0.1 (large before weighting, tiny after)."""),
    code('''bal = R.balance(kept, cov, t, e, cfg)
print("passed:", bal["passed"], "| worst:", bal["worst_covariate"], round(bal["max_abs_smd_after"], 3))
pd.DataFrame({"before": bal["smd_before"], "after": bal["smd_after"]}).sort_values("before", key=abs, ascending=False).head(10).round(3)'''),
    md("""## Placebo treatment

**What:** permute top-3 across the whole sample and re-estimate, five times, and average the five estimates.

**Why:** a permuted treatment is unrelated to anything, so any 'effect' points to a bug. (The spec
shuffled within each search; that leaks the true effect when searches differ in their number of
treated rows, so the permutation is across the whole sample.)

**Rule:** the average must lie within ±1.96 *single-run* standard errors of zero. Averaging removes most of one permutation's chance noise (with one permutation, the check fired by chance on 3 of 53 runs). This over-corrects: the average of five is held to about 4.4 of its own standard errors (1.96 × √5), so chance false alarms almost never happen and only a systematic placebo effect bigger than about two single-run standard errors is caught. It is not a 5% test. Tested against its own standard error, the Q1 placebo still passes easily (z = −0.06); the Q3 clicks placebo does not (z = −2.4, see the README).

**Expect:** an interval that includes zero."""),
    code('''R.placebo(df, "top3", "click_bool", cov, "srch_id", cfg)'''),
    md("""## Random common cause and subset stability

**What:** add a pure-noise covariate and re-estimate; then re-estimate on random half-samples of
searches.

**Why:** a stable estimate should not move when noise is added, and half-samples should mostly land
inside the full-sample interval.

**Expect:** a shift well under half a standard error; at least 80% of half-samples inside (in practice nearly all)."""),
    code('''print(R.random_common_cause(df, "top3", "click_bool", cov, "srch_id", cfg, primary))
sub = R.subset_stability(df, "top3", "click_bool", cov, "srch_id", cfg, primary)
print({k: v for k, v in sub.items() if k != "estimates"})'''),
    md("""## Held-out covariate (reported, not pass or fail)

**What:** leave location score 2 out, together with the two columns computed from it (its missing flag
and its value relative to the search), and estimate the 'effect' of top-3 on it.

**Why:** in the observational arm the ranker uses hotel attributes, so top-3 hotels *should* differ
on them; this measures how much. It is not a pass/fail check.

**Expect:** a clearly nonzero value."""),
    code('''R.held_out_covariate(kept, t, cov, "prop_location_score2", kept.srch_id.to_numpy())'''),
    md("""## Sensitivity to a hidden confounder

**What:** omitted-variable bias bounds (Cinelli & Hazlett, 2020) on the linear regression: if a hidden
confounder were as strong as each observed covariate, how far would the estimate move? Also the
robustness value and the E-value.

**Why:** no check above can detect a confounder that is not in the data, such as the ranker's own
score. This says how strong one would need to be to explain the effect away.

**Expect:** the effect keeps its sign even with a confounder as strong as the strongest covariate."""),
    code('''sens = R.sensitivity(y, t, kept[cov], primary, cfg, groups=kept.srch_id.to_numpy())
print({k: sens[k] for k in ("passed", "strongest_benchmark_covariate", "robustness_value_q1", "e_value")})
sens["bound_at_strongest"]'''),
    md("""## The uncomfortable lesson

Every check can pass while the estimate still misses the benchmark. The checks test what the data can
reveal; a confounder the file does not contain (the ranker's information), or a covariate that is
secretly an outcome, can leave all of them quiet."""),
    quiz([
        ("Why is the placebo check a test of the code rather than of confounding?",
         "A permuted treatment has no relation to the covariates or the outcome, so a nonzero estimate can only come from a bug or a leak in the pipeline. It says nothing about whether the real treatment is confounded."),
        ("What does it mean that balance passes after weighting?",
         "The weighted treated and control groups look alike on every variable we adjusted for. It cannot show balance on variables we did not include."),
        ("Why is the held-out covariate check expected to show an effect here?",
         "The ranker uses hotel attributes to place hotels, so top-3 hotels genuinely differ on them. The check measures how strong that selection is."),
        ("What does a robustness value of about 0.12 mean?",
         "A hidden confounder would need to explain about 12% of the residual variance of both the treatment and the outcome to remove the whole estimate. It refers to the regression estimate (8.04 pp), the model the sensitivity analysis is built on, not to the AIPW estimate."),
    ]),
]

# ---------------------------------------------------------------- 05
NOTEBOOKS["05_agent_walkthrough"] = [
    md("""# 05 · The agent, one node at a time

**Goal:** see what each LangGraph node does to the state, then watch the human-review interrupt pause
a run and resume it from the SQLite checkpoint.

This notebook runs in **offline mode** (`offline=True`): the Claude nodes are replaced by fixed stubs
that use the reference graph, so it needs no API key. The tool nodes are the real code. To keep it
quick the refuters use 10 half-samples here."""),
    md(SETUP_MD), code(SETUP + '''
cfg["refuters"]["subset_n"] = 10
from causallab.agent import nodes as N
from causallab.agent.state import AgentState
rt = N.make_runtime("walkthrough", cfg)
rt.trace_path.unlink(missing_ok=True)  # a re-run starts a fresh trace instead of appending to the last one
state = AgentState(run_id="walkthrough", question="What is the causal effect of a top-3 position on clicks?",
                   offline=True, synthetic=SYNTHETIC, auto_approve=True)

def apply(state, updates):
    """What LangGraph does between nodes: merge the node's updates into the state."""
    return state.model_copy(update=updates)'''),
    md("""## intake

**What:** turns the question into a typed spec: treatment, outcome, unit, population, target arm.

**Why:** everything downstream reads these typed fields. Code then enforces the firewall: a top-3
question can only target the observational arm.

**Expect:** `top3`, `click_bool`, `observational`."""),
    code('''state = apply(state, N.intake(rt, state)); state.spec'''),
    md("""## profile_data

**What:** rows, searches, treatment share, outcome rate, most-missing columns, from the target arm only."""),
    code('''state = apply(state, N.profile_data(rt, state)); state.profile'''),
    md("""## propose_dag

**What:** each variable's role, an explicit list of directed edges, the adjustment set, the excluded
variables with reasons, and unobserved threats. Claude sees variable **descriptions only**, never data.

**Why the explicit edges:** the identification check builds the graph from the edges. Built from role
labels alone, every 'confounder' would be wired to treatment and outcome and the check would always pass.

**Expect:** 44 adjustment variables; `booking_bool` and `position` excluded."""),
    code('''state = apply(state, N.propose_dag(rt, state))
print(len(state.dag.adjustment_set), "adjustment variables;", len(state.dag.edges), "edges")
[(x.name, x.reason) for x in state.dag.excluded]'''),
    md("""## human_review

**What:** in a real run this node pauses and waits for a person (shown at the end of the notebook).
Here `auto_approve=True`, so it approves at once.

**Why:** approval sits on the **assumptions**, before any estimate exists, so nobody approves a graph
because they liked the number it produced."""),
    code('''state = apply(state, N.human_review(rt, state)); state.review'''),
    md("""## identify

**What:** the back-door criterion on the approved graph: (1) nothing in the adjustment set is a
descendant of the treatment, and (2) the set blocks every back-door path (DoWhy's d-separation check).

**Why both:** DoWhy's function covers (2) only; on its own it would accept adjusting for a mediator.

**Expect:** valid."""),
    code('''state = apply(state, N.identify(rt, state)); state.identification'''),
    md("""## route_design, estimate, refute

**What:** check the treatment varies; run the five estimators; run the checks. `estimate` also saves the
propensity model, and the **benchmark module** (not this path) uses it to trim the randomly ranked arm
and compute the trimmed benchmark and the gap.

**Expect:** a few minutes; the AIPW estimate and a gap verdict."""),
    code('''state = apply(state, N.route_design(rt, state))
state = apply(state, N.estimate(rt, state))
print({k: round(v["estimate"] * 100, 2) for k, v in state.estimates.items()})
print("gap:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in state.gap["primary"].items()})'''),
    code('''state = apply(state, N.refute(rt, state))
{k: v["passed"] for k, v in state.refutations.items()}'''),
    md("""## judge

**What:** code applies the thresholds and picks the route: pass, re-specify (balance failed, at most
twice, never dropping an approved confounder) or refuse (overlap, placebo, random common cause, subset
stability or sensitivity failed, or balance still failed after two re-specifications).
Claude only writes the explanation.

**Expect:** `pass` (every check passed)."""),
    code('''state = apply(state, N.judge(rt, state)); state.verdict'''),
    md("""## narrate

**What:** the report. Claude may only write prose with named placeholders such as `{primary_estimate}`;
code fills them from state, and a guard rejects the report if any number in it was not written by code."""),
    code('''state = apply(state, N.narrate(rt, state))
print((rt.run_dir / "report.md").read_text(encoding="utf-8")[:2500])'''),
    md("""## The interrupt and resume, explicitly

**What:** the compiled graph with a SQLite checkpointer. Without `auto_approve`, the run stops at
`human_review` and returns the review request. Calling `resume` with a decision continues from the
checkpoint, even in a new Python process.

**Why:** a person must be able to read the assumptions, take their time, and come back.

This demo uses a small synthetic scenario so it runs in seconds."""),
    code('''from causallab.agent import graph
r = graph.start("Does the featured slot cause more clicks?", dataset="trap:clean_confounding", offline=True, cfg=cfg)
print("paused:", r["paused"])
r["review_request"]'''),
    code('''r = graph.resume(r["run_id"], {"decision": "approve"}, cfg)
print("paused:", r["paused"], "| status:", r["state"]["outcome_status"], "| route:", r["state"]["verdict"]["route"])'''),
    md("""## The graph diagram"""),
    code('''# its own file: `python -m causallab diagram` writes the published results/agent_graph.mmd
print(graph.export_diagram(results_dir(cfg) / "agent_graph_notebook.mmd", cfg).read_text(encoding="utf-8"))'''),
    quiz([
        ("Why does human approval happen before estimation rather than after?",
         "Approving after seeing the number invites choosing the specification that gives a pleasing answer. Approving the assumptions first keeps the analysis honest."),
        ("Why must propose_dag return explicit edges?",
         "The identification check needs a real graph. If edges were inferred from role labels, every labelled confounder would be wired as a common cause and the check would be a tautology."),
        ("What may the re-specification loop change, and what may it not?",
         "It may add covariates or interaction terms and tighten trimming. It may never drop an approved confounder, because removing a variable to improve balance is a causal fallacy."),
        ("How can Claude write a report without writing any numbers?",
         "It writes prose with named placeholders; code fills them from state, and a guard checks every number in the final report against the values code wrote."),
        ("What does the firewall prevent, and how is it enforced?",
         "The Q1 estimation path reading randomly ranked rows. Each arm has its own loader, a top-3 question can only get the observational one, and a test checks the access log."),
    ]),
]

# ---------------------------------------------------------------- 06
NOTEBOOKS["06_heterogeneity_and_promotion"] = [
    md("""# 06 · Who gains most from a top slot (Q2), and does the promotion badge work (Q3)?

**Q2:** fit a causal forest on the observational arm and compare group effects with the same groups in
the randomly ranked arm. **Q3:** estimate the promotion badge's effect inside the randomly ranked arm,
where position is random and so cannot confound it. Q3 has no experimental truth, so its verdict is at
most *suggestive, not experimentally verified*.

The forest here uses 60,000 rows for speed; `python -m causallab q2` uses the config value."""),
    md(SETUP_MD), code(SETUP),
    md("""## Q2: the causal forest

**What:** EconML `CausalForestDML` estimates an effect for every row as a function of the effect
modifiers (within-search price tercile, star band, trip type), adjusting for the other covariates.
Group effects are averages within each group.

**Why:** a 'which listings gain most from a boost' policy needs effects by segment, not one average.

**Expect:** a few minutes. With 60,000 rows the forest is noisy: its intervals are wide, and the
order of groups can differ from the full 1.2-million-row run shown in the next cell."""),
    code('''from causallab import data, features
from causallab.heterogeneity import run_q2
spec = load_spec(cfg)
df = features.q1_rows(data.load_arm("observational", cfg, SYNTHETIC))
q2 = run_q2(df, spec["adjustment_set"], cfg, SYNTHETIC, rows=60000)
rows = []
for gr in q2["groups"]:
    ex = gr["experimental"] or {}
    rows.append({"group": f"{gr['modifier']}={gr['level']}", "forest_pp": gr["estimate"] * 100,
                 "forest_low": gr["ci_low"] * 100, "forest_high": gr["ci_high"] * 100,
                 "experimental_pp": ex.get("estimate", np.nan) * 100 if ex else np.nan})
print("Small forest (60,000 rows) for speed: read the intervals, not just the points.")
pd.DataFrame(rows).round(2)'''),
    md("""## The full run (1.2 million rows)

**What:** the saved result of `python -m causallab q2`, which the README reports.

**Expect:** on the real sample, the forest above the experiment in every group, by about 1.2 pp on average."""),
    code('''p = results_dir(cfg) / ("q2_group_effects_synthetic.json" if SYNTHETIC else "q2_group_effects.json")
full = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"groups": []}
if not full["groups"]:
    print("Not run yet: `python -m causallab q2` writes it.")
pd.DataFrame([{"group": f"{g['modifier']}={g['level']}", "forest_pp": g["estimate"] * 100,
               "forest_low": g["ci_low"] * 100, "forest_high": g["ci_high"] * 100,
               "experimental_pp": (g["experimental"] or {}).get("estimate", np.nan) * 100}
              for g in full["groups"]]).round(2)'''),
    md("""## Reading the Q2 comparison

The experimental column comes from the randomly ranked arm (via the benchmark module). Where the forest
tracks it, the ranking of segments is credible; where it does not, the same confounding that biased Q1
is likely at work within that segment. The forest intervals are not clustered by search."""),
    md("""## Q3: the promotion badge, inside the randomly ranked arm

**What:** the five estimators with `promotion_flag` as treatment, on all positions of the randomly
ranked arm, adjusting for hotel and search covariates, with position as a precision covariate.

**Why:** promotion is not randomised, so it is confounded by hotel type and the hotel's usual price
level (`prop_log_historical_price` stays in). The *displayed* price is different: a promotion lowers it,
so displayed price and the features built from it are mediators, part of how the badge works. They are
left out, as Claude also did in the agent's Q3 run. Claude's final clicks run left out more (the
page-relative quality, affinity and visitor-history columns: 30 variables against 39 here), so the two
numbers can differ slightly. Working inside the random arm removes position as a confounder.

**Expect:** about 1.4 pp on clicks on the real sample, close to the agent's Q3 run (1.35 pp)."""),
    code('''from causallab import estimators as E, refuters as R
exp = data.load_arm("experimental", cfg, SYNTHETIC)
# The badge lowers the displayed price: displayed price and features built from it are mediators
MEDIATORS = ["log_price", "price_rank_pct", "price_rel_median", "comp_expedia_cheaper_n", "comp_expedia_dearer_n"]
cov3 = [c for c in spec["adjustment_set"] if c not in ("promotion_flag", *MEDIATORS)] + ["position"]
q3 = E.estimate_all(exp, "promotion_flag", "click_bool", cov3, "srch_id", cfg)
pd.DataFrame({k: {"pp": v.estimate * 100, "low": v.ci_low * 100, "high": v.ci_high * 100}
              for k, v in q3["estimates"].items()}).T.round(2)'''),
    md("""## What adjusting for a mediator does

**What:** the same doubly robust estimate, with displayed price and the features built from it put back in.

**Why:** adjusting for a mediator blocks part of the effect you are trying to measure: here, the part of
the badge's effect that works through the lower price.

**Expect:** a smaller number, roughly half (about 0.7 pp on the real sample). Treat this as an
illustration of what conditioning on a mediator does, not as a measured share of the effect that runs
through price: displayed price also partly reflects the hotel's own price level, which is a confounder."""),
    code('''q3_med = E.estimate_all(exp, "promotion_flag", "click_bool", cov3 + MEDIATORS, "srch_id", cfg, which=("aipw",))
a, b = q3["estimates"]["aipw"].estimate, q3_med["estimates"]["aipw"].estimate
print(f"displayed price left out: {a * 100:.2f} pp | displayed price adjusted for: {b * 100:.2f} pp")'''),
    md("""## Q3 sensitivity, reported prominently

**What:** the same hidden-confounder bounds as in notebook 04.

**Why:** with no randomised truth, sensitivity is the main evidence about how fragile the estimate is.

**Expect:** the verdict wording: *suggestive, not experimentally verified*."""),
    code('''k = exp[q3["keep"]].reset_index(drop=True)
s3 = R.sensitivity(k.click_bool.to_numpy(float), k.promotion_flag.to_numpy(), k[cov3], q3["estimates"]["aipw"], cfg, groups=k.srch_id.to_numpy())
print("passed:", s3["passed"], "| strongest:", s3["strongest_benchmark_covariate"], "| RV:", round(s3["robustness_value_q1"], 3))
print("Verdict: suggestive, not experimentally verified.")'''),
    quiz([
        ("Why compare forest group effects with group effects from the randomly ranked arm?",
         "The forest is fitted on confounded data. The randomly ranked arm gives an unconfounded estimate for the same groups, so agreement is evidence the segment ranking is real."),
        ("Why does estimating Q3 inside the randomly ranked arm help, and why is it still not an experiment?",
         "Position is random there, so it cannot confound the badge. But the badge itself was not randomised: hotels on sale differ in price and type, so confounding by those (and by unobserved factors) remains."),
        ("Why does adjusting for the displayed price roughly halve the badge's estimated effect on clicks?",
         "A promotion lowers the displayed price, and a lower price attracts clicks, so price lies on the path from badge to click: it is a mediator. Adjusting for it blocks the part of the effect that works through price. The hotel's usual price level is different: it is a confounder and stays in. (The same logic keeps click_bool out when the outcome is booking: a booking requires a click.)"),
    ]),
]

# ---------------------------------------------------------------- 07
NOTEBOOKS["07_trap_suite"] = [
    md("""# 07 · The trap suite

**Goal:** check that the agent behaves well on five synthetic scenarios where the right answer is known:
clean confounding (proceed and recover the effect), no overlap (refuse), a collider offered as a control
(exclude it), a mediator offered as a control (exclude it), and a balance failure (re-specify, then refuse).

This notebook runs **offline**, so the 'agent' proposes the hidden reference answer: the counts test
the plumbing, not judgement. The real test uses Claude: `python -m causallab traps` (needs an API key).
Two runs per scenario here; the CLI uses ten."""),
    md(SETUP_MD), code(SETUP),
    md("""## The scenarios

**What:** each has product-analytics variable names and descriptions (what Claude sees), a hidden
reference answer and a true effect.

**Why the descriptions matter:** with Claude, the agent must infer from 'opened a support ticket after
the session' that the variable is a collider, without being told its role."""),
    code('''from causallab.synthetic import trap_scenarios
for sc in trap_scenarios(n=4000):
    print(f"{sc.name:18s} expected: {sc.expected:22s} true effect {sc.true_effect:.3f}  bad controls: {sc.bad_controls}")'''),
    md("""## Why the traps bite

**What:** AIPW with and without the bad control.

**Why:** adjusting for a collider opens a path between treatment and outcome; adjusting for a mediator
blocks part of the effect. Either way the estimate is wrong.

**Expect:** the collider pulls the estimate towards zero; the mediator roughly halves it."""),
    code('''from causallab import estimators as E
for sc in trap_scenarios(n=12000):
    if not sc.bad_controls:
        continue
    for label, cov in (("reference", sc.reference_adjustment), ("with bad control", sc.reference_adjustment + sc.bad_controls)):
        r = E.estimate_all(sc.df, sc.treatment, sc.outcome, cov, sc.group, cfg, which=("aipw",))["estimates"]["aipw"]
        print(f"{sc.name:9s} {label:17s} AIPW {r.estimate:.3f}  (truth {sc.true_effect:.3f})")'''),
    md("""## Run the suite (offline, two runs per scenario)

**Expect:** a few minutes; every scenario 'as expected' in offline mode, since the stub knows the answer.
`fresh=True` gives these runs new ids, so nothing is reused from the CLI suite."""),
    code('''from causallab import trap_suite
res = trap_suite.run(offline=True, runs=2, cfg=cfg, fresh=True)
pd.DataFrame(res["summary"]).T'''),
    quiz([
        ("Why do offline trap results say nothing about the agent's judgement?",
         "Offline, propose_dag returns the hidden reference answer, so it cannot fall into the traps. Only runs with Claude test whether it recognises a collider or mediator from the descriptions."),
        ("Why is adjusting for a collider harmful?",
         "A collider is caused by both treatment and outcome. Conditioning on it creates an association between them that is not causal, biasing the estimate."),
        ("Why should the no-overlap scenario end in refusal rather than an estimate?",
         "When placement is a hard rule on quality, there are no comparable featured and non-featured listings at the same quality, so any estimate is extrapolation."),
    ]),
]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook()
        nb["cells"] = cells
        nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nbf.write(nb, OUT / f"{name}.ipynb")
        print("wrote", OUT / f"{name}.ipynb")


if __name__ == "__main__":
    main()
