# CausalLab report: Does the promotion badge cause more bookings?

Run `run-87f4e454` · status: **reported** · dataset: expedia

## Verdict

**Suggestive, not experimentally verified.** The treatment was not randomised in either arm; see the sensitivity row of the scorecard.

## Question

- Treatment: `promotion_flag` · Outcome: `booking_bool` · Target arm: experimental
- Unit: one hotel displayed in one search results page
- Population: hotel rows from randomly ranked searches (experimental arm), where position is assigned at random

## Summary

We estimated the effect of the promotion badge on bookings using searches whose results were randomly ranked, so the badge's position on the results page cannot confound the comparison. The primary estimate is 0.25 percentage points, with a ninety-five percent interval from 0.21 to 0.29.

## Estimates (percentage points, 95% interval, clustered by `srch_id`)

| Estimator | Effect (pp) | 95% interval | Gap to benchmark (pp) |
|---|---|---|---|
| Naive, before trimming | 0.18 | [0.15, 0.21] | n/a |
| Naive difference in means | 0.18 | [0.15, 0.21] | n/a |
| Regression adjustment | 0.23 | [0.20, 0.27] | n/a |
| Inverse propensity weighting | 0.26 | [0.22, 0.30] | n/a |
| Doubly robust (AIPW), primary | 0.25 | [0.21, 0.29] | n/a |
| Double machine learning | 0.25 | [0.22, 0.29] | n/a |



## Refutation scorecard

| Check | Result | Measured | Rule |
|---|---|---|---|
| overlap | pass | trimmed share 1% | share of rows outside [0.01, 0.99] <= 0.3 |
| balance | pass | max \|SMD\| 0.057 (star_zero) | all \|SMD\| after weighting < 0.1 |
| placebo | pass | -0.01 pp [-0.04, 0.02] | average of 5 permutation estimates within +/-1.96 single-run standard errors of zero |
| random common cause | pass | shift 0.20 SE | estimate moves by < 0.5 standard errors |
| subset stability | pass | 85% inside | at least 80% of 20 half-sample estimates inside the full-sample 95% interval |
| sensitivity | pass | regression estimate 0.23 pp would fall to 0.21 pp (cluster-robust 95% interval 0.17 to 0.24) with a hidden confounder as strong as srch_length_of_stay; robustness value 0.012; E-value of the primary estimate 2.39 | effect keeps its sign with a hidden confounder kd=1.0, ky=1.0 times the strongest covariate |
| held out covariate | reported | prop_location_score2: 0.038 SD | reported, not pass or fail |

## Interpretation

The badge itself was not randomised, because hotels and Expedia decide which properties receive it, so the verdict is capped at suggestive rather than experimentally verified. The main threat is hotel attributes that Expedia knows but that are absent from the file, such as popularity or commission, which could drive both promotion and bookings. The overlap, balance, placebo, random common cause, subset stability, and sensitivity checks all passed. A held-out covariate check was run and is reported, though it is not a pass or fail test. No experimental benchmark exists, so the estimate cannot be compared against one and no gap verdict is given.

## Assumptions and limitations

- Adjustment set approved at human review (35 variables): `prop_starrating`, `star_zero`, `prop_review_score`, `review_zero`, `review_missing`, `prop_brand_bool`, `prop_location_score1`, `prop_location_score2`, `loc2_missing`, `prop_log_historical_price`, `hist_price_zero`, `srch_query_affinity_score`, `affinity_missing`, `log_distance`, `distance_missing`, `visitor_same_country`, `srch_length_of_stay`, `srch_booking_window`, `srch_saturday_night_bool`, `site_5`, `site_14`, `site_15`, `site_24`, `site_32`, `search_month`, `search_dow`, `search_hour`, `position`, `srch_adults_count`, `srch_children_count`, `srch_room_count`, `n_results`, `visitor_hist_starrating`, `visitor_hist_adr_usd`, `visitor_hist_missing`
- Unobserved threats: Hotel attributes known to Expedia but not in the file, such as hotel popularity or commission, which affect promotion and bookings; Timing and targeting of promotion campaigns; Visitor intent and unobserved visitor characteristics; Hotel availability and room inventory
- Hotels in one search compete for the same slots; the estimates ignore this interference.
- The file only contains searches with at least one click, so all rates are conditional on that selection, and the selection differs between the arms.
- The data is from 2012 to 2013 and one company. The method transfers; the numbers do not.
- Booking is a secondary outcome: bookings were sampled differently in the two arms of the file (see the EDA), so booking rates are not comparable across arms and this result ranks below the click result.

## Specification history

- Attempt 0: 35 adjustment variables, failures: none, route: pass
