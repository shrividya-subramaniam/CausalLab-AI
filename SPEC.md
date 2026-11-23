**Project specification for Claude Code: an agentic causal inference engine for search-ranking position bias**

Dataset: Expedia hotel searches (ICDM 2013) · Stack: Python, LangGraph, Claude, DoWhy, EconML

# 1. How to use this document

This is the complete brief for building the project. It is written to be handed to Claude Code as-is. Section 10 contains a kickoff prompt to paste alongside it.

## Instructions to Claude Code

- Build the phases in Section 6 in order. Stop at the end of each phase, show the acceptance checks, and wait for the go-ahead before continuing.

- Several dataset facts below are marked **verify in EDA**. Check them against the data and report what you find. If the data disagrees with this document, the data wins: say so and adapt.

- Never invent, round favourably, or hard-code a result. Every number in a report must be read from a saved results file.

- A phase that fails its acceptance checks is a finding to report, not a problem to tune away. In particular, do not adjust the model specification to make the observational estimate match the randomized benchmark (Section 4.4).

- Library APIs change. Before writing LangGraph, langchain-anthropic, DoWhy or EconML code, check the installed version and its current documentation rather than relying on memory.

## What the human must provide first

1.  A Kaggle account that has accepted the competition rules at https://www.kaggle.com/c/expedia-personalized-sort, and train.csv downloaded into data/raw/. The file is over 2 GB. test.csv is not needed.

2.  An Anthropic API key in a .env file as ANTHROPIC_API_KEY. Phases 0 to 4 need no key.

3.  Python 3.11 or later and about 16 GB of RAM. With less memory, lower the sample fraction in the config. The default sample is 10% of searches, about 40,000 searches and 1 million rows.

# 2. The project in brief

On a search results page, listings near the top get more clicks. Part of that is because the ranking algorithm puts better listings on top, and part is because people click what they see first. Separating the two is a causal question, and it matters to any company that ranks things: search results, feeds, ads, marketplace listings, loan offers.

CausalLab AI is a LangGraph multi-agent system, using Claude, that takes a causal question in plain language, proposes the causal assumptions, has a human approve them, picks and runs estimators, tries to break its own answer, and then reports a result or declines to give one.

The project asks three questions of the Expedia data:

| **\#** | **Question**                                                                           | **Why it is in the project**                                                                             |
|--------|----------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| Q1     | What is the causal effect of a top-3 position on the probability a listing is clicked? | Headline result. Estimated from algorithm-ranked searches, then scored against randomly ranked searches. |
| Q2     | Does that effect differ by price tier, star rating or trip type?                       | Heterogeneous effects. Leads to a "which listings gain most from a boost" policy.                        |
| Q3     | Does the promotion badge cause more clicks and bookings?                               | No randomized truth exists. Shows how the agent behaves when it cannot verify itself.                    |

The headline claim is a within-study comparison: an observational estimate next to an experimental benchmark for the same effect. The gap between them, whatever it turns out to be, is the main result.

# 3. The dataset

## 3.1 Source and shape

- **Name:** Personalize Expedia Hotel Searches, ICDM 2013 (Kaggle competition expedia-personalized-sort).

- **Size:** train.csv has about 9.9 million rows and 54 columns. **Verify in EDA.**

- **Unit of a row:** one hotel shown in one search (an impression). A search is identified by srch_id and contains several rows, one per hotel displayed.

- **The key feature:** random_bool is 1 when the results of that search were displayed in random order and 0 when Expedia's ranking algorithm ordered them. Published work using this data reports that roughly one third of searches were randomly ranked. **Verify in EDA.**

## 3.2 Columns and their causal role

| **Columns**                                                                                                                                      | **Meaning**                                               | **Role in this project**                                               |
|--------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|------------------------------------------------------------------------|
| srch_id                                                                                                                                          | Search identifier                                         | Cluster for standard errors and cross-fitting folds                    |
| random_bool                                                                                                                                      | 1 = random order, 0 = algorithm order                     | Splits the data into the experimental and observational arms           |
| position                                                                                                                                         | Rank of the hotel on the page, 1 = top                    | Treatment for Q1 and Q2 (top-3 indicator)                              |
| click_bool                                                                                                                                       | 1 if the user clicked the hotel                           | Primary outcome                                                        |
| booking_bool                                                                                                                                     | 1 if the user booked the hotel                            | Secondary outcome. Downstream of the click                             |
| gross_bookings_usd                                                                                                                               | Value of the booking                                      | Post-outcome. Never a control                                          |
| promotion_flag                                                                                                                                   | 1 if a sale-price promotion was displayed                 | Confounder in Q1. Treatment in Q3                                      |
| price_usd                                                                                                                                        | Displayed price                                           | Confounder. See quirk below                                            |
| prop_starrating, prop_review_score, prop_brand_bool, prop_location_score1, prop_location_score2, prop_log_historical_price                       | Hotel attributes                                          | Confounders: they drive both ranking and clicks                        |
| srch_length_of_stay, srch_booking_window, srch_adults_count, srch_children_count, srch_room_count, srch_saturday_night_bool, srch_destination_id | What the user searched for                                | Context and effect modifiers                                           |
| site_id, visitor_location_country_id, prop_country_id, orig_destination_distance                                                                 | Site and geography                                        | Context                                                                |
| visitor_hist_starrating, visitor_hist_adr_usd                                                                                                    | Visitor purchase history, mostly missing                  | Use a missing-indicator, not imputation alone                          |
| srch_query_affinity_score                                                                                                                        | Log probability the hotel is clicked in internet searches | Strong quality proxy. Often missing                                    |
| comp1_rate to comp8_rate_percent_diff                                                                                                            | Competitor price and availability                         | Optional. Mostly missing. Summarise into two or three features if used |
| prop_id, date_time                                                                                                                               | Hotel identifier, search timestamp                        | Grouping, time features                                                |

## 3.3 Quirks to handle

- **Price is not comparable across searches.** price_usd may be per night or for the whole stay, and with or without taxes, depending on the country. Build within-search features: price rank, and price relative to the search median. Winsorise prices and compute every within-search feature, including the number of results, on all rows of the search before any position filter is applied.

- **Rows within a search are not independent.** Always sample, split into folds, and bootstrap by srch_id, never by row.

- **Sample selection.** Published work reports that the Kaggle file only contains searches with at least one click, and that bookings were sampled differently across the two arms. **Verify in EDA** by checking the share of searches with zero clicks and the booking rate in each arm. This is why click, not booking, is the primary outcome.

- **Reserved positions.** Some positions may be nearly absent in one arm because they were reserved for other listing types. **Verify in EDA** by tabulating row counts by position and arm.

- **Missing values.** Review score, location score 2, affinity score, distance and visitor history have many nulls. LightGBM handles nulls natively. For linear models, impute and add missing-indicators. In prop_starrating, prop_review_score and prop_log_historical_price a value of 0 is a distinct state (unrated, no reviews yet, not sold recently) and is different from a null. Add a zero-indicator column for each before treating the 0 as missing. Verify in EDA.

- **Outliers.** A small number of prices are implausibly large. Winsorise before building relative-price features.

# 4. Causal strategy

## 4.1 Why the naive comparison is wrong

In algorithm-ranked searches, the ranker places hotels it expects to be clicked at the top. Hotel quality therefore causes both position and clicks. Comparing click rates of top-3 hotels with the rest mixes the effect of position with the effect of quality, and overstates the position effect. In randomly ranked searches, quality cannot influence position, so the same comparison is a valid estimate.

## 4.2 Definitions

| **Term**           | **Definition for this project**                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
|--------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Unit               | One impression: a hotel displayed in a search                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Treatment (Q1, Q2) | top3 = 1 if position is 1, 2 or 3; 0 if position is 4 to 10. Rows at position 11 or lower are left out of the primary comparison, so the control group covers a similar mix of positions in both arms. Searches with no row in positions 4 to 10 are excluded. If EDA shows that a position from 1 to 10 is nearly absent in one arm, drop that position from both arms and record it in the specification. Sensitivity check: use all positions 4 or lower as the comparison group |
| Treatment (Q3)     | promotion_flag                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Outcome            | click_bool (primary), booking_bool (secondary)                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Estimand           | Average treatment effect, as a difference in click probability in percentage points, on the population that survives overlap trimming. The trimming rule is fitted on the observational arm only and then applied to the experimental arm, so the benchmark and the estimate describe the same population                                                                                                                                                                           |
| Observational arm  | Rows with random_bool = 0. The agent estimates Q1 from these only                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Experimental arm   | Rows with random_bool = 1. Used to compute the benchmark and never shown to the Q1 estimation path                                                                                                                                                                                                                                                                                                                                                                                  |

## 4.3 Starting causal graph

The agent proposes its own graph, but the expected structure is below and serves as the reference answer in tests.

- **Confounders (adjust for):** hotel attributes, displayed price and its within-search rank, promotion flag, affinity score, search context, number of results in the search.

- **Treatment:** top3. **Outcome:** click_bool.

- **Mediator (do not adjust for):** click_bool when the outcome is booking_bool.

- **Post-outcome (do not adjust for):** gross_bookings_usd, booking_bool when the outcome is click.

- **Unobserved:** the ranker's internal score and anything it knows that is not in the file, such as photos or past performance of the hotel. This is the main threat to Q1 and must be stated in every report.

## 4.4 The experimental benchmark

1.  Confirm randomization worked: in the experimental arm, standardized mean differences of hotel attributes between top3 = 1 and 0 should be near zero. This is the causal counterpart of the sample-ratio check in an A/B test.

2.  Benchmark effect: difference in click rate between top3 = 1 and 0 in the experimental arm, with a 95% confidence interval from a bootstrap that resamples whole searches. Use 1,000 repetitions, drawn over per-search counts of rows and clicks by treatment that are computed once, so the bootstrap is vectorised.

3.  Check the two arms describe similar searches: compare search-level covariates across arms. If they differ, also report a benchmark reweighted to the observational arm's covariate mix. Reweighting cannot correct selection on the outcome: if every search has at least one click, each arm was selected under a different ranking. State this next to the gap.

**Firewall rule.** The benchmark lives in its own module and its code is fixed before any observational estimate exists. It saves two numbers: the untrimmed benchmark, computed once in Phase 2, and the trimmed benchmark, recomputed for each specification the agent runs and saved with that specification. The observational specification (covariates, models, trimming rule) is written down and committed at the end of Phase 1, before the benchmark module is run or any experimental-arm outcome is viewed. Any change made after seeing the benchmark is labelled exploratory in the report.

- **Reference and agent specification.** The Phase 1 commit is the reference specification (config/spec_locked.yaml). In the agent run, propose_dag sees variable descriptions only, never outcome data or benchmark results. The specification approved at human_review is logged as the agent specification. Phase 3 results use the reference, Phase 6 results use the agent specification, and the report shows both and lists every difference between them.

- **Data access isolation.** data.py exposes a separate loader for each arm. The estimation tools receive only the loader for the target arm held in state, and a test fails if a run with treatment top3 reads a row with random_bool = 1.

- **Cross-arm trimming.** The estimators save the fitted propensity model under a path scoped to the run and question (results/models/{run_id}\_propensity.pkl), so one run never overwrites the model of another. For top3 runs, the benchmark module, not the estimation path, loads that model to score and trim the experimental arm. Q3 has no benchmark, so nothing is trimmed across arms.

- **Gap test.** Report the gap with its own 95% interval, computed from the two standard errors, since the arms are independent. The estimate counts as matching the benchmark only if the whole interval lies within a tolerance set in config before the run (default: 20% of the benchmark effect). An interval that includes zero but is wider than the tolerance is reported as inconclusive, not as a match.

## 4.5 Observational estimators for Q1

| **Estimator**                | **What it does**                                                                                                                                                                                     | **Purpose here**                                                                                                             |
|------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| Naive difference in means    | Top-3 click rate minus the rest, observational arm                                                                                                                                                   | Shows the size of the confounding                                                                                            |
| Regression adjustment        | Outcome model with covariates and treatment. Linear probability model fitted by ordinary least squares in statsmodels, which the sensitivity check needs. The coefficient reads in percentage points | Simple baseline                                                                                                              |
| Inverse propensity weighting | LightGBM model for P(top3 \| covariates), weights trimmed                                                                                                                                            | Classic approach. Exposes overlap problems                                                                                   |
| Doubly robust (AIPW)         | Combines outcome and propensity models                                                                                                                                                               | Consistent if either model is right. Primary estimate: it targets the average treatment effect, as the benchmark does        |
| Double machine learning      | Cross-fitted residual-on-residual, via EconML                                                                                                                                                        | Secondary estimate. It targets a variance-weighted effect, which differs from the average treatment effect when effects vary |

All models use cross-fitting with folds grouped by srch_id, and standard errors clustered by srch_id. Compute clustered standard errors from each estimator's per-row scores (its influence function), summed within srch_id. Only where no such formula is available, use a bootstrap that resamples whole searches, with 200 repetitions. In EconML, supply srch_id as the grouping for cross-fitting, because an integer cv on its own splits by row.

## 4.6 Diagnostics and refutation

| **Check**                         | **What it tests**                                                                           | **Default pass rule**                                                                                                                                                                                                                                                                                                                              |
|-----------------------------------|---------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Overlap                           | Propensity score distributions of treated and control                                       | Trim outside 0.01 to 0.99 and report the share of rows dropped                                                                                                                                                                                                                                                                                     |
| Covariate balance                 | Standardized mean differences after weighting                                               | All below 0.1, computed with the trimmed, stabilised weights over the approved adjustment set                                                                                                                                                                                                                                                      |
| Placebo treatment                 | Shuffle top3 within each search and re-estimate                                             | Interval includes zero. Code check only: this cannot detect confounding                                                                                                                                                                                                                                                                            |
| Held-out covariate                | Leave one hotel attribute out of the adjustment set and estimate the "effect" of top3 on it | Reported, not pass or fail. A nonzero result is expected, because the ranker uses these attributes. Run it on one attribute, named in the reference specification, with the regression adjustment estimator                                                                                                                                        |
| Random common cause               | Add a random noise covariate (DoWhy refuter)                                                | Estimate moves by less than half its standard error                                                                                                                                                                                                                                                                                                |
| Subset stability                  | Re-estimate on random subsets of searches, primary estimator only                           | On 20 half-sample subsets, at least 90% of estimates fall inside the full-sample 95% interval                                                                                                                                                                                                                                                      |
| Sensitivity to hidden confounding | E-value, or the DoWhy or sensemakr sensitivity tools                                        | Computed on the regression adjustment model, using the robustness value with covariate benchmarks (the PySensemakr package, or the closed-form formulas of Cinelli and Hazlett, 2020). Fail if a hidden confounder as strong as the strongest observed covariate would reduce the effect to zero. Also report the E-value for the primary estimate |

Thresholds live in a config file. The pass or fail decision is made by code. Claude explains the decision but does not make it.

## 4.7 Q2 and Q3

- **Q2:** fit a causal forest (EconML CausalForestDML) on a subsample of the observational arm (200,000 rows by default). Report group average effects by within-search price tercile, star rating and trip type. Define trip type in config from the srch\_ columns, for example length of stay, Saturday night and children count. Validate against the same group effects computed directly in the experimental arm.

- **Q3:** promotion is not randomized in either arm. Estimating inside the experimental arm removes position as a confounder, because position is random there. Run the full estimator set in Section 4.5, with AIPW as the primary estimate, run all refuters, and report the sensitivity analysis prominently. The verdict wording must be "suggestive, not experimentally verified". Click is the primary outcome. Report booking only as a secondary result, with the sampling caveat from Section 3.3. Q3 keeps all positions and includes position as a covariate. The pipeline runs once for click and once for booking.

## 4.8 Limitations every report must state

- The ranker's internal score is unobserved, so some confounding may remain in Q1.

- Hotels in one search compete for the same slots, so one hotel's position affects another's. The estimates ignore this interference.

- If the file only contains searches with a click, all rates are conditional on that selection. The selection also differs between the arms, and reweighting cannot correct that.

- The data is from 2012 to 2013 and one company. The method transfers. The numbers do not.

# 5. Agent architecture

## 5.1 Graph nodes

| **Node**      | **Type**      | **Responsibility**                                                                                                                                                                                                                                                                                                                                                                        |
|---------------|---------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| intake        | Claude        | Turn the plain-language question into a typed spec: treatment, outcome, unit, population, and target arm (observational or experimental). The experimental arm is allowed only when the treatment is not top3                                                                                                                                                                             |
| profile_data  | Tool          | Schema, missingness, row counts per arm, treatment prevalence                                                                                                                                                                                                                                                                                                                             |
| propose_dag   | Claude        | Structured output: each variable's role with a one-line reason, the adjustment set, and the variables excluded and why. Also an explicit list of directed edges, from which the graph is built. Sees variable descriptions only, never outcome data                                                                                                                                       |
| human_review  | Interrupt     | Pause. Show the graph and assumptions. Resume on approve, edit or reject                                                                                                                                                                                                                                                                                                                  |
| identify      | Tool          | Check with DoWhy that the adjustment set satisfies the backdoor criterion on the approved graph, built from the explicit edge list and not from the role labels                                                                                                                                                                                                                           |
| route_design  | Router        | Conditional edge: proceed to estimation, or go to refusal if treatment has no variation. Overlap is checked after estimation, by judge                                                                                                                                                                                                                                                    |
| estimate      | Tools         | Run the estimators in Section 4.5 and write results to state                                                                                                                                                                                                                                                                                                                              |
| refute        | Tools         | Run the checks in Section 4.6 in parallel                                                                                                                                                                                                                                                                                                                                                 |
| judge         | Code + Claude | Code applies thresholds. A balance failure loops back to propose_dag and passes through human_review again. Re-specification may add covariates or interaction terms, or tighten the trimming range. It may never drop an approved confounder. An overlap or sensitivity failure goes to refuse. At most two loops, and every attempt is logged and reported. Claude writes the rationale |
| heterogeneity | Tool          | Causal forest and group effects, only if judge passes                                                                                                                                                                                                                                                                                                                                     |
| narrate       | Claude        | Write the prose with named placeholders. Code fills the placeholders with numbers from state                                                                                                                                                                                                                                                                                              |
| refuse        | Claude        | Explain why the effect cannot be credibly estimated and what data would fix it                                                                                                                                                                                                                                                                                                            |

## 5.2 Design rules

- **Claude never computes or copies a number.** Estimates travel in typed graph state (Pydantic models). Claude receives them read-only, and the report template inserts them. A test checks every number in the narrative against state.

- **Human approval sits on assumptions, not results.** Use LangGraph's interrupt mechanism with a SQLite checkpointer so a paused run can be resumed.

- **The loop is bounded.** Two re-specification attempts, then refusal. Every attempt and its results appear in the report, so the loop cannot hide a specification search.

- **Refusal is a first-class outcome** with its own node and its own tests.

- **Offline mode.** A --no-llm flag swaps Claude nodes for fixed stubs using the reference graph in Section 4.3, so the whole pipeline runs without an API key.

- **Models.** Read model IDs from config. Suggested: claude-sonnet-5-5 for propose_dag and judge, claude-haiku-5-5 for intake and narrate. Confirm current IDs at https://docs.claude.com. Temperature 0.

- **Logging.** Log every node entry, exit and routing decision to a JSONL trace file per run.

# 6. Build phases

| **Phase**      | **Build**                                                                                                                                                                                                                                                                                  | **Acceptance checks**                                                                                                                                                                                                                                                                                                                                                           |
|----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 0\. Setup      | Repo skeleton, pyproject.toml or requirements.txt with pinned versions, config file, .env.example, seed handling                                                                                                                                                                           | pytest runs. Imports succeed for every library. A smoke test makes one minimal call to each library feature this document depends on: LangGraph interrupt and resume from the SQLite checkpointer, DoWhy backdoor identification, and EconML cross-fitting with grouped folds, with an assertion that no srch_id appears in both the training and test part of any fold         |
| 1\. Data       | Convert the CSV to Parquet without loading it all in memory (DuckDB or chunked pandas). Sample by srch_id hash. EDA notebook covering every verify in EDA item. Commit the reference specification (covariates, models, trimming rule) to config/spec_locked.yaml at the end of this phase | Row and search counts per arm. Random share. Click and booking rates per arm. Rows by position and arm. A short written list of where the data differs from this document, and which sections each difference affects                                                                                                                                                           |
| 2\. Benchmark  | Feature builder. Randomization balance check. Benchmark effect with clustered bootstrap. Cross-arm covariate comparison. Click-rate-by-position figure for both arms                                                                                                                       | Balance table in the experimental arm. Benchmark estimate (untrimmed) and interval saved to results/benchmark.json                                                                                                                                                                                                                                                              |
| 3\. Estimators | The five estimators in Section 4.5 as plain functions with a common return type. Uses the reference specification committed in Phase 1, unchanged                                                                                                                                          | Each adjusted estimator recovers a known effect on a small confounded synthetic dataset, within a fixed tolerance and with a fixed seed. The naive estimator recovers it only on an unconfounded dataset. Results saved to JSON                                                                                                                                                 |
| 4\. Refuters   | The checks in Section 4.6 and the threshold config                                                                                                                                                                                                                                         | Each pass or fail check fires on a synthetic dataset built to trip it and stays quiet on a clean one. The placebo and random common cause checks are tested for staying quiet only                                                                                                                                                                                              |
| 5\. Agent      | LangGraph state, nodes, router, interrupt, checkpointer, loop, refusal path, offline mode, trace logging                                                                                                                                                                                   | Offline end-to-end run passes. A paused run resumes from the checkpoint. Graph diagram exported                                                                                                                                                                                                                                                                                 |
| 6\. Q1 run     | Full agent run on Q1 with Claude and a human approval step. Comparison table against the benchmark, using the trimmed benchmark for the same specification                                                                                                                                 | Results table in Section 8 filled from files. Gap to benchmark reported as found                                                                                                                                                                                                                                                                                                |
| 7\. Q2 and Q3  | Causal forest and group effects. Promotion analysis with sensitivity                                                                                                                                                                                                                       | Group effects compared with the experimental arm. Q3 report carries the "not verified" wording                                                                                                                                                                                                                                                                                  |
| 8\. Trap suite | Five small synthetic scenarios with described variables: clean confounding, no overlap, a collider offered as a control, a mediator offered as a control, and a failing balance check                                                                                                      | For each scenario, record whether the agent proceeded, excluded the bad control, or refused. Report counts, with the number of runs (at least 10 per scenario). Expected outcomes: clean confounding, proceed and recover the effect; no overlap, refuse; collider, exclude it and proceed; mediator, exclude it and proceed; failing balance check, re-specify and then refuse |
| 9\. Write-up   | README with the question, method, results, limitations and how to run. Final teaching notebooks (Section 9)                                                                                                                                                                                | A fresh clone runs the offline pipeline from the README alone on synthetic data (--synthetic), without the Kaggle file                                                                                                                                                                                                                                                          |

# 7. Repository layout

```text
causallab/
  config.yaml              thresholds, sample fraction, model IDs, seeds, bootstrap repetitions, forest subsample size
  data/raw/                train.csv goes here (git-ignored)
  data/processed/          Parquet files (git-ignored)
  src/causallab/
    data.py                loading, sampling by srch_id, arm split
    features.py            within-search features, missing-indicators
    benchmark.py           experimental-arm estimate (firewalled)
    estimators.py          naive, regression, IPW, AIPW, DML
    refuters.py            overlap, balance, placebo, sensitivity
    heterogeneity.py       causal forest, group effects
    synthetic.py           data generators for tests and the trap suite, and an Expedia-schema dataset for --synthetic runs
    agent/state.py         Pydantic state models
    agent/nodes.py         node functions
    agent/graph.py         graph assembly, router, interrupt
    agent/prompts.py       prompts and report template
    cli.py                 run, resume, --no-llm, --synthetic
  notebooks/               01 to 07, see Section 9
  tests/
  results/                 JSON outputs, figures, traces
  README.md
```

# 8. Results to report

Fill this table from the saved result files. Leave nothing estimated by eye.

| **Estimate of top-3 effect on click** | **Effect (pp)** | **95% interval** | **Gap to benchmark** |
|---------------------------------------|-----------------|------------------|----------------------|
| Experimental benchmark                |                 |                  | n/a                  |
| Naive, observational arm              |                 |                  |                      |
| Regression adjustment                 |                 |                  |                      |
| Inverse propensity weighting          |                 |                  |                      |
| Doubly robust (AIPW)                  |                 |                  |                      |
| Double machine learning               |                 |                  |                      |

Also report:

- How much of the naive gap the adjustment removed.

- The refutation scorecard, check by check.

- Group effects for Q2 next to their experimental counterparts.

- The Q3 estimate with its sensitivity result.

- Trap suite outcomes as counts.

If the adjusted estimates still miss the benchmark, that is a legitimate and interesting result. Say so plainly and discuss the unobserved ranker score as the likely reason.

# 9. Requirements for learning from the code

The owner of this project will rerun the code by hand to learn the concepts. Build with that reader in mind.

- Produce seven notebooks that mirror the phases: 01_eda, 02_benchmark, 03_estimators, 04_refuters, 05_agent_walkthrough, 06_heterogeneity_and_promotion, 07_trap_suite.

- Each notebook runs top to bottom on its own, importing from src/causallab, on the sampled Parquet file. Notebooks 05 and 07 default to offline mode, so they run without an API key.

- Before each code cell that introduces a concept, add a short markdown cell: what the concept is, why it is needed here, and what output to expect.

- In 03_estimators, implement inverse propensity weighting and AIPW once by hand in a few lines of NumPy before using a library, so the mechanics are visible.

- In 05_agent_walkthrough, build the graph one node at a time, print the state after each node, and show the interrupt and resume step explicitly.

- Prefer plain functions and short modules over class hierarchies. Add type hints and docstrings. Comment the why, not the what.

- Fix random seeds. Keep each notebook under about ten minutes on the default sample. If the defaults in the config break this limit, lower them and report the values used.

- End each notebook with three to five "check your understanding" questions and their answers.

# 10. Kickoff prompt for Claude Code

Paste this with the document attached or saved in the repository (exported to Markdown) as SPEC.md.

```text
Read SPEC.md in full before doing anything. It is the specification for this project.

Build the project phase by phase as listed in Section 6. Start with Phase 0 and Phase 1 only. At the end of each phase, run the acceptance checks, show me the output, list anything in the data that disagrees with the spec, and wait for me to say continue.

train.csv is in data/raw/. Do not load it into memory in one go.

Rules: never hard-code or invent a result; read every reported number from a file in results/; do not change the observational specification after seeing the benchmark without labelling it exploratory; check current library documentation before using LangGraph, langchain-anthropic, DoWhy or EconML APIs.

I will rerun this code by hand to learn from it, so follow Section 9.
```

# References

- Kaggle. Personalize Expedia Hotel Searches, ICDM 2013. https://www.kaggle.com/c/expedia-personalized-sort

- Ursu, R. (2018). The Power of Rankings: Quantifying the Effect of Rankings on Online Consumer Search and Purchase Decisions. Marketing Science. Uses the same data and its random-ranking arm to identify position effects.

- Chernozhukov et al. (2018). Double/debiased machine learning for treatment and structural parameters. The Econometrics Journal.

- LaLonde, R. (1986). Evaluating the Econometric Evaluations of Training Programs with Experimental Data. American Economic Review. The original within-study comparison design this project follows.
