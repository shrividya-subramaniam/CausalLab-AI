# CausalLab report: What is the causal effect of a top-3 position on the probability a listing is clicked?

Run `run-ff2df23c` · status: **reported** · dataset: expedia

## Question

- Treatment: `top3` · Outcome: `click_bool` · Target arm: observational
- Unit: one hotel displayed in one search results page
- Population: all hotel-in-search rows in the observational arm (algorithm-ranked searches)

## Summary

The study estimated the causal effect of placing a listing in the top three positions on the probability that it is clicked. The observational estimate uses searches where Expedia's ranking algorithm chose the positions, and it is compared with randomly ranked searches where position was assigned at random. The primary estimate was 6.61 percentage points, with a ninety-five percent interval from 6.49 to 6.73. A naive comparison that ignores the ranker's choices gave 9.77 percentage points, which is why adjustment was needed.

## Estimates (percentage points, 95% interval, clustered by `srch_id`)

| Estimator | Effect (pp) | 95% interval | Gap to benchmark (pp) |
|---|---|---|---|
| Experimental benchmark (trimmed) | 5.25 | [5.09, 5.42] | n/a |
| Naive, before trimming | 9.77 | [9.64, 9.89] | 4.52 [4.31, 4.72], mismatch |
| Naive difference in means | 9.77 | [9.64, 9.89] | 4.52 [4.31, 4.72], mismatch |
| Regression adjustment | 8.04 | [7.92, 8.17] | 2.80 [2.59, 3.00], mismatch |
| Inverse propensity weighting | 6.96 | [6.83, 7.09] | 1.71 [1.50, 1.92], mismatch |
| Doubly robust (AIPW), primary | 6.61 | [6.49, 6.73] | 1.36 [1.16, 1.57], mismatch |
| Double machine learning | 6.53 | [6.40, 6.65] | 1.28 [1.07, 1.49], mismatch |

## Comparison with the experimental benchmark

The benchmark is the difference in click rates in randomly ranked searches, on the rows that survive the same trimming rule (0% of rows trimmed). Primary gap: 1.36 pp, 95% interval [1.16, 1.57], tolerance ±1.05 pp: **mismatch**.

### Post-hoc: comparing like with like

Added after the pre-registered comparison: the ranker fills the top ten with different hotels than random ranking does, and the effect differs by hotel, so the gap above mixes leftover confounding with a difference in population.

- **Ranker's hotels:** benchmark reweighted to resemble them 5.89 pp [5.68, 6.12]; gap to the primary estimate 0.72 pp [0.46, 0.98], match. The weakest of the comparisons: the reweighting under-corrects (it did on synthetic data, where the truth is known), and its interval ignores the uncertainty in the weights.
- **Randomly ranked hotels:** observational effect carried over to them 6.49 pp against the benchmark 5.25 pp; gap 1.24 pp [1.07, 1.39], mismatch (interval covers the randomly ranked sample only, not the models' own uncertainty).

## Refutation scorecard

| Check | Result | Measured | Rule |
|---|---|---|---|
| overlap | pass | trimmed share 0% | share of rows outside [0.01, 0.99] <= 0.3 |
| balance | pass | max \|SMD\| 0.024 (srch_query_affinity_score) | all \|SMD\| after weighting < 0.1 |
| placebo | pass | 0.00 pp [-0.11, 0.10] | average of 5 permutation estimates within +/-1.96 single-run standard errors of zero |
| random common cause | pass | shift 0.02 SE | estimate moves by < 0.5 standard errors |
| subset stability | pass | 95% inside | at least 80% of 20 half-sample estimates inside the full-sample 95% interval |
| sensitivity | pass | regression estimate 8.04 pp would fall to 7.69 pp (cluster-robust 95% interval 7.57 to 7.81) with a hidden confounder as strong as n_results; robustness value 0.121; E-value of the primary estimate 3.39 | effect keeps its sign with a hidden confounder kd=1.0, ky=1.0 times the strongest covariate |
| held out covariate | reported | prop_location_score2: 0.255 SD | reported, not pass or fail |

## Interpretation

The estimate should be trusted with caution. The main threat is Expedia's internal hotel quality or relevance score, which the ranker used and which likely affects both position and click probability, but which is not observed. The overlap, balance, placebo, random common cause, subset stability, and sensitivity checks all passed, and the held-out covariate check was run and reported. Against the randomized benchmark of 5.25 percentage points, the plain gap of 1.36 percentage points is a mismatch. That gap mixes leftover confounding with a difference between the hotels the ranker places in the top ten and the hotels that random ranking places there, because the effect differs by hotel. Two post-hoc comparisons, added after the pre-registered comparison, put both estimates on the same hotels. Reweighting the random-ranking benchmark to the ranker's hotels gave 5.89 percentage points, a reweighted gap of 0.72 percentage points, with verdict match. Carrying the observational effect over to the random-ranking hotels gave 6.49 percentage points, a transported gap of 1.24 percentage points, with verdict mismatch. Reweighting cannot correct for information that only the ranker had, and the interval for the carried-over estimate ignores the uncertainty in the outcome models, so both comparisons should be read as provisional.

## Assumptions and limitations

- Adjustment set approved at human review (44 variables): `promotion_flag`, `prop_starrating`, `star_zero`, `prop_review_score`, `review_zero`, `review_missing`, `prop_brand_bool`, `prop_location_score1`, `prop_location_score2`, `loc2_missing`, `prop_log_historical_price`, `hist_price_zero`, `log_price`, `price_rank_pct`, `price_rel_median`, `star_rel`, `review_rel`, `loc1_rel`, `loc2_rel`, `srch_query_affinity_score`, `affinity_missing`, `log_distance`, `distance_missing`, `srch_length_of_stay`, `srch_booking_window`, `srch_adults_count`, `srch_children_count`, `srch_room_count`, `srch_saturday_night_bool`, `n_results`, `visitor_same_country`, `site_5`, `site_14`, `site_15`, `site_24`, `site_32`, `search_month`, `search_dow`, `search_hour`, `visitor_hist_starrating`, `visitor_hist_adr_usd`, `visitor_hist_missing`, `comp_expedia_cheaper_n`, `comp_expedia_dearer_n`
- Unobserved threats: Expedia's internal hotel quality or relevance score used by the ranking algorithm; Hotel historical click and booking rates, and the commission or bid paid to Expedia; Personalised user features such as search history, device and user ID; Room availability and inventory that the ranker uses; Other undisclosed search context used for ranking
- The ranker's internal score is unobserved, so some confounding may remain.
- Hotels in one search compete for the same slots; the estimates ignore this interference.
- The file only contains searches with at least one click, so all rates are conditional on that selection, and the selection differs between the arms.
- The data is from 2012 to 2013 and one company. The method transfers; the numbers do not.

## Specification history

- Attempt 0: 44 adjustment variables, failures: none, route: pass

Difference from the reference specification (config/spec_locked.yaml): added none; removed none.
