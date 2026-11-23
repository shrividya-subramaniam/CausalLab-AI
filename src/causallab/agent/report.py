"""Report building with a number guard.

Every number in a report is inserted by `Numbers.fmt`, which records the exact string it
wrote. `check_numbers` then confirms that every number in the finished report is one of
those strings, so no number can come from the LLM or from anywhere but state.
"""
from __future__ import annotations

import re
from typing import Any

NUMBER = re.compile(r"(?<![A-Za-z_\d.`])-?\d+(?:\.\d+)?(?![A-Za-z_\d`])")
PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")
ALLOWED_LITERALS = {"95"}  # "95% interval" in headings


class Numbers:
    def __init__(self) -> None:
        self.written: set[str] = set()

    def fmt(self, x: float | int | None, kind: str = "pp", digits: int = 2) -> str:
        if x is None:
            return "n/a"
        if kind == "pp":
            s = f"{x * 100:.{digits}f}"
        elif kind == "pct":
            s = f"{x * 100:.0f}%"
        elif kind == "int":
            s = f"{int(x):,}".replace(",", "")
        else:
            s = f"{x:.{digits}f}"
        if s.startswith("-") and not re.search(r"[1-9]", s):
            s = s[1:]  # a value that rounds to zero prints as 0.00, never -0.00
        self.written.add(s.rstrip("%"))
        return s

    def text(self, s: str) -> str:
        """Register every number in a code-written string (e.g. a rule built from config)."""
        for m in NUMBER.finditer(s):
            self.written.add(m.group(0))
        return s


def check_numbers(report: str, numbers: Numbers) -> list[str]:
    """Return numbers in the report that were not written by code. Empty list = clean."""
    found = [m.group(0) for m in NUMBER.finditer(report)]
    return [n for n in found if n not in numbers.written and n.lstrip("-") not in numbers.written
            and n not in ALLOWED_LITERALS]


def prose_is_clean(prose: str, allowed: set[str]) -> bool:
    """LLM prose may hold placeholders from the allowed set, and no digit anywhere else.
    A placeholder name written without its braces (e.g. benchmark_estimate) is also rejected,
    because code would leave it unfilled in the report."""
    if any(p not in allowed for p in PLACEHOLDER.findall(prose)):
        return False
    rest = PLACEHOLDER.sub("", prose)
    if any(re.search(rf"\b{re.escape(name)}\b", rest) for name in allowed if "_" in name):
        return False
    return not re.search(r"\d", rest)


def repair_placeholders(prose: str, allowed: set[str]) -> str:
    """Put braces around an allowed placeholder name that Claude wrote without them
    (primary_estimate -> {primary_estimate}). Only names with an underscore are repaired, since
    plain words such as "gap" are ordinary English. Nothing else in the prose changes, and the
    digit check still runs afterwards."""
    for name in sorted((n for n in allowed if "_" in n), key=len, reverse=True):
        prose = re.sub(rf"(?<![{{\w]){re.escape(name)}(?![}}\w])", "{" + name + "}", prose)
    return prose


def fill(prose: str, values: dict[str, str]) -> str:
    return PLACEHOLDER.sub(lambda m: values.get(m.group(1), m.group(0)), prose)


def estimates_table(estimates: dict[str, dict[str, Any]], nums: Numbers, bench: dict | None,
                    gap_by_name: dict[str, dict] | None) -> str:
    rows = ["| Estimator | Effect (pp) | 95% interval | Gap to benchmark (pp) |", "|---|---|---|---|"]
    if bench:
        rows.append(f"| Experimental benchmark (trimmed) | {nums.fmt(bench['estimate'])} | "
                    f"[{nums.fmt(bench['ci_low'])}, {nums.fmt(bench['ci_high'])}] | n/a |")
    order = ["naive_untrimmed", "naive", "regression", "ipw", "aipw", "dml"]
    label = {"naive_untrimmed": "Naive, before trimming", "naive": "Naive difference in means",
             "regression": "Regression adjustment", "ipw": "Inverse propensity weighting",
             "aipw": "Doubly robust (AIPW), primary", "dml": "Double machine learning"}
    for k in order:
        if k not in estimates:
            continue
        r = estimates[k]
        gap = "n/a"
        if gap_by_name and k in gap_by_name:
            g = gap_by_name[k]
            gap = f"{nums.fmt(g['gap'])} [{nums.fmt(g['ci_low'])}, {nums.fmt(g['ci_high'])}], {g['verdict']}"
        rows.append(f"| {label[k]} | {nums.fmt(r['estimate'])} | [{nums.fmt(r['ci_low'])}, {nums.fmt(r['ci_high'])}] | {gap} |")
    return "\n".join(rows)


def scorecard_table(refutations: dict[str, dict[str, Any]], nums: Numbers) -> str:
    rows = ["| Check | Result | Measured | Rule |", "|---|---|---|---|"]
    for name, r in refutations.items():
        res = {True: "pass", False: "FAIL", None: "reported"}[r.get("passed")]
        if name == "overlap":
            m = f"trimmed share {nums.fmt(r['trimmed_share'], 'pct')}"
        elif name == "balance":
            m = f"max |SMD| {nums.fmt(r['max_abs_smd_after'], 'x', 3)} ({r['worst_covariate']})"
        elif r.get("error"):
            m = "re-estimate lost overlap"
        elif name == "placebo":
            m = f"{nums.fmt(r['estimate'])} pp [{nums.fmt(r['ci'][0])}, {nums.fmt(r['ci'][1])}]"
        elif name == "random_common_cause":
            m = f"shift {nums.fmt(r['shift_in_se'], 'x', 2)} SE"
        elif name == "subset_stability":
            m = f"{nums.fmt(r['share_inside'], 'pct')} inside"
        elif name == "held_out_covariate":
            m = f"{r['covariate']}: {nums.fmt(r['estimate_in_sd'], 'x', 3)} SD"
        elif name == "sensitivity":
            b = r["bound_at_strongest"]
            ci = r.get("adjusted_ci_cluster")
            ci_txt = f" (cluster-robust 95% interval {nums.fmt(ci[0])} to {nums.fmt(ci[1])})" if ci else ""
            m = (f"regression estimate {nums.fmt(r['ols_estimate'])} pp would fall to {nums.fmt(b['adjusted_estimate'])} pp"
                 f"{ci_txt} with a hidden confounder as strong as {r['strongest_benchmark_covariate']}; robustness value "
                 f"{nums.fmt(r['robustness_value_q1'], 'x', 3)}; E-value of the primary estimate {nums.fmt(r['e_value'], 'x', 2)}")
        else:
            m = ""
        cell = lambda s: s.replace("|", "\\|")  # a bare | inside a cell (as in |SMD|) would split the table row
        rows.append(f"| {name.replace('_', ' ')} | {res} | {cell(m)} | {cell(nums.text(r['rule']))} |")
    return "\n".join(rows)
