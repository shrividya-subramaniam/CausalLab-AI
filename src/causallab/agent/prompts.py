"""Prompts and the report template.

Rule for every prompt that produces prose: Claude writes no numbers. Where a number belongs
it writes a named placeholder such as {primary_estimate}; code fills it from state.
"""
from __future__ import annotations

INTAKE = """You turn a causal question about a dataset into a typed specification.
Pick the treatment and outcome from the listed variable names exactly as written.
target_arm: use "observational" unless the question is about a treatment other than the page
position AND the dataset has a randomly ranked arm where position is random; then use
"experimental", which removes position as a confounder. Questions about position (top3) always
use "observational"."""

PROPOSE_DAG = """You are a careful applied econometrician. Propose a causal graph for the question.
You see variable names and descriptions only, never data.

Return:
- a role for every variable (treatment, outcome, confounder, precision, mediator, collider,
  post_outcome, instrument, design, irrelevant), each with a one-line reason;
- an explicit list of directed edges (cause -> effect) including treatment -> outcome. The graph
  is built from these edges, so draw every edge you believe in, including edges into and out of
  variables you exclude;
- the adjustment set: variables that block every back-door path from treatment to outcome.
  Never include a mediator, a collider, anything measured after the treatment or the outcome, or
  a variable that defines the treatment;
- the excluded variables with the reason each is excluded;
- unobserved threats: important common causes that are not in the data.
Judge timing from the descriptions: something recorded after the session or caused by the
treatment cannot be a confounder.
A variable that defines the treatment (for example the position a treatment indicator is computed
from) has role "design": exclude it from the adjustment set. Code merges it into the treatment
node, so its edges to the outcome count as part of the treatment's effect."""

RESPECIFY = """The previous specification failed the covariate balance check after weighting.
You may ONLY: add covariates from the variable list, add interaction terms between adjustment
variables, or tighten the trimming range (for example to [0.05, 0.95]). You may NOT drop any
approved confounder: removing a confounder to pass a balance check is a causal fallacy.
Return the full revised proposal."""

JUDGE = """You explain a decision that code has already made. Do not change it.
Write two to four plain sentences explaining why the run was {route}, naming the checks that
failed or passed. Write NO digits and no numbers in words; refer to checks by name."""

NARRATE = """Write two short paragraphs for a causal-analysis report.
Paragraph one: what was estimated and the headline result. Paragraph two: how much to trust it,
the main threat (choose it from the unobserved threats in the facts), and how the result relates
to the benchmark if one exists. Describe the design exactly as the facts state it.
You must not write any digit, not even for the interval level or a percent sign: write "ninety-five percent
interval", never "95%". Where a number belongs, write one of these placeholders exactly,
in curly braces: {placeholders}. Use only placeholders from that list.
Do not say that any matching was done."""

NARRATE_POST_HOC = """
The facts include post-hoc population checks. In paragraph two, do not attribute the whole gap to
ranker information: say the gap mixes leftover confounding with a difference between the hotels the
ranker shows and those random ranking shows, state both post-hoc comparisons with their placeholders
(reweighted_benchmark, reweighted_gap, reweighted_verdict; transported_estimate, transported_gap,
transported_verdict, each written in curly braces), say they were added after the pre-registered
comparison, and give the caveat."""

REFUSE = """The analysis was refused. Explain in two to four plain sentences why the effect cannot
be credibly estimated here, and what data or design would fix it (for example a randomized
experiment, better overlap, or measurement of the missing confounder).
Write NO digits and no numbers in words."""


REPORT_TEMPLATE = """# CausalLab report: {question}

Run `{run_id}` · status: **{status}** · dataset: {dataset}{exploratory_label}

## Question

- Treatment: `{treatment}` · Outcome: `{outcome}` · Target arm: {target_arm}
- Unit: {unit}
- Population: {population}

## Summary

{summary_prose}

## Estimates (percentage points, 95% interval, clustered by `{group}`)

{estimates_table}

{benchmark_section}

## Refutation scorecard

{scorecard_table}

## Interpretation

{interpretation_prose}

## Assumptions and limitations

- Adjustment set approved at human review ({n_adjust} variables): {adjustment_list}
- Unobserved threats: {threats}
{limitations}

## Specification history

{history}
"""

BOOKING_CAVEAT = ("Booking is a secondary outcome: bookings were sampled differently in the two arms of the "
                  "file (see the EDA), so booking rates are not comparable across arms and this result ranks "
                  "below the click result.")

RANKER_LIMITATION = "The ranker's internal score is unobserved, so some confounding may remain."

EXPEDIA_LIMITATIONS = [
    "Hotels in one search compete for the same slots; the estimates ignore this interference.",
    "The file only contains searches with at least one click, so all rates are conditional on that "
    "selection, and the selection differs between the arms.",
    "The data is from 2012 to 2013 and one company. The method transfers; the numbers do not.",
]
