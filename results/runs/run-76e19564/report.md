# CausalLab report: Does the promotion badge cause more clicks?

Run `run-76e19564` · status: **reported** · dataset: expedia

## Verdict

**Suggestive, not experimentally verified.** The treatment was not randomised in either arm; see the sensitivity row of the scorecard.

## Question

- Treatment: `promotion_flag` · Outcome: `click_bool` · Target arm: experimental
- Unit: one hotel displayed in one search results page
- Population: hotel-in-search rows from randomly ranked searches (experimental arm), where position is assigned at random

## Summary

The question was whether the promotion badge causes more clicks on hotel listings. The effect was estimated inside randomly ranked searches, so a listing's position cannot confound the comparison. The primary estimate is 1.35 percentage points clicks, with a ninety-five percent interval from 1.25 to 1.45. The naive comparison of badged and unbadged listings, which does not account for the design, gives 1.67 percentage points and is reported only for contrast.

## Estimates (percentage points, 95% interval, clustered by `srch_id`)

| Estimator | Effect (pp) | 95% interval | Gap to benchmark (pp) |
|---|---|---|---|
| Naive, before trimming | 1.67 | [1.58, 1.76] | n/a |
| Naive difference in means | 1.67 | [1.58, 1.76] | n/a |
| Regression adjustment | 1.46 | [1.37, 1.55] | n/a |
| Inverse propensity weighting | 1.39 | [1.28, 1.50] | n/a |
| Doubly robust (AIPW), primary | 1.35 | [1.25, 1.45] | n/a |
| Double machine learning | 1.35 | [1.25, 1.45] | n/a |



## Refutation scorecard

| Check | Result | Measured | Rule |
|---|---|---|---|
| overlap | pass | trimmed share 1% | share of rows outside [0.01, 0.99] <= 0.3 |
| balance | pass | max \|SMD\| 0.054 (star_zero) | all \|SMD\| after weighting < 0.1 |
| placebo | pass | -0.05 pp [-0.13, 0.04] | average of 5 permutation estimates within +/-1.96 single-run standard errors of zero |
| random common cause | pass | shift 0.11 SE | estimate moves by < 0.5 standard errors |
| subset stability | pass | 95% inside | at least 80% of 20 half-sample estimates inside the full-sample 95% interval |
| sensitivity | pass | regression estimate 1.46 pp would fall to 1.33 pp (cluster-robust 95% interval 1.23 to 1.42) with a hidden confounder as strong as n_results; robustness value 0.027; E-value of the primary estimate 1.94 | effect keeps its sign with a hidden confounder kd=1.0, ky=1.0 times the strongest covariate |
| held out covariate | reported | prop_location_score2: 0.039 SD | reported, not pass or fail |

## Interpretation

The result should be treated as suggestive, not experimentally verified. Hotels and Expedia decide which listings get the badge, so the treatment was not randomised and no experimental benchmark exists to check the estimate against. The main threat is Expedia's internal hotel information used for promotions, such as commission, margin and inventory, which could influence both whether a hotel is promoted and how many people click on it, and none of it is in the file. Listing quality, visitor intent and whether promotions were targeted at particular search contexts are also unobserved. The overlap, balance, placebo, random common cause, subset stability and sensitivity checks all passed. A held-out covariate check was run and reported, but it is not a pass or fail test.

## Assumptions and limitations

- Adjustment set approved at human review (30 variables): `prop_starrating`, `star_zero`, `prop_review_score`, `review_zero`, `review_missing`, `prop_brand_bool`, `prop_location_score1`, `prop_location_score2`, `loc2_missing`, `prop_log_historical_price`, `hist_price_zero`, `log_distance`, `distance_missing`, `srch_length_of_stay`, `srch_booking_window`, `srch_saturday_night_bool`, `visitor_same_country`, `site_5`, `site_14`, `site_15`, `site_24`, `site_32`, `search_month`, `search_dow`, `search_hour`, `position`, `srch_adults_count`, `srch_children_count`, `srch_room_count`, `n_results`
- Unobserved threats: Expedia's internal hotel information used for promotions (commission, margin, inventory); Hotel photos, descriptions and other listing quality not in the file; Visitor intent and unobserved user characteristics; Whether the hotel's promotion was targeted at particular search contexts
- Hotels in one search compete for the same slots; the estimates ignore this interference.
- The file only contains searches with at least one click, so all rates are conditional on that selection, and the selection differs between the arms.
- The data is from 2012 to 2013 and one company. The method transfers; the numbers do not.

## Specification history

- Attempt 0: 30 adjustment variables, failures: none, route: pass
