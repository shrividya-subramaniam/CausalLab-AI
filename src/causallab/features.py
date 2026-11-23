"""Feature builder.

Order of operations matters and follows Section 3.3 of the spec:
  1. Winsorise price, then compute every within-search feature (price rank, price relative
     to the search median, number of results, quality relative to the search) on ALL rows
     of a search. A user sees the whole page, and the ranker scored every hotel on it.
  2. Only then flag the rows used by each analysis (positions 1-10 minus reserved slots).
  3. Zero-coded values (no star rating, no reviews yet, not sold recently) get their own
     indicator before the 0 is treated as missing, because 0 and NULL are different states.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .config import load_config

TOP_SITES = [5, 14, 15, 24, 32]  # 87% of rows (Phase 1 EDA)

# Covariates available to the adjustment set. The reference specification picks from these.
HOTEL = [
    "prop_starrating", "prop_review_score", "prop_brand_bool", "prop_location_score1",
    "prop_location_score2", "prop_log_historical_price",
    "star_zero", "review_zero", "review_missing", "hist_price_zero", "loc2_missing",
]
PRICE = ["log_price", "price_rank_pct", "price_rel_median", "promotion_flag"]
RELATIVE = ["star_rel", "review_rel", "loc1_rel", "loc2_rel"]
QUERY = ["srch_query_affinity_score", "affinity_missing", "log_distance", "distance_missing"]
SEARCH = [
    "srch_length_of_stay", "srch_booking_window", "srch_adults_count", "srch_children_count",
    "srch_room_count", "srch_saturday_night_bool", "n_results", "visitor_same_country",
    *[f"site_{s}" for s in TOP_SITES], "search_month", "search_dow", "search_hour",
]
VISITOR = ["visitor_hist_starrating", "visitor_hist_adr_usd", "visitor_hist_missing"]
COMPETITOR = ["comp_expedia_cheaper_n", "comp_expedia_dearer_n"]
ALL_COVARIATES = HOTEL + PRICE + RELATIVE + QUERY + SEARCH + VISITOR + COMPETITOR
# Columns computed from another column. Holding a covariate out must hold these out too,
# otherwise the "held-out" value is still in the model through its own derivatives.
DERIVED_FROM = {
    "prop_starrating": ["star_zero", "star_rel"],
    "prop_review_score": ["review_zero", "review_missing", "review_rel"],
    "prop_location_score1": ["loc1_rel"],
    "prop_location_score2": ["loc2_missing", "loc2_rel"],
    "prop_log_historical_price": ["hist_price_zero"],
    "srch_query_affinity_score": ["affinity_missing"],
}

ID_COLS = ["srch_id", "prop_id", "random_bool", "position"]
OUTCOMES = ["click_bool", "booking_bool", "gross_bookings_usd"]


def _sql(src: str, lo: float, hi: float) -> str:
    comp_cheaper = " + ".join(f"coalesce((comp{i}_rate = 1)::INT, 0)" for i in range(1, 9))
    comp_dearer = " + ".join(f"coalesce((comp{i}_rate = -1)::INT, 0)" for i in range(1, 9))
    sites = ", ".join(f"(site_id = {s})::INT AS site_{s}" for s in TOP_SITES)
    return f"""
    WITH base AS (
      SELECT *, LEAST(GREATEST(price_usd, {lo}), {hi}) AS price_w FROM '{src}'
    )
    SELECT
      srch_id, prop_id, random_bool, position, click_bool, booking_bool, gross_bookings_usd,
      prop_brand_bool, prop_location_score1, promotion_flag,
      -- zero-coded states get their own flag, then the 0 becomes missing
      coalesce(prop_starrating = 0, false)::INT AS star_zero,
      coalesce(prop_review_score = 0, false)::INT AS review_zero,
      (prop_review_score IS NULL)::INT AS review_missing,
      coalesce(prop_log_historical_price = 0, false)::INT AS hist_price_zero,
      (prop_location_score2 IS NULL)::INT AS loc2_missing,
      NULLIF(prop_starrating, 0) AS prop_starrating,
      NULLIF(prop_review_score, 0) AS prop_review_score,
      NULLIF(prop_log_historical_price, 0) AS prop_log_historical_price,
      prop_location_score2,
      -- price: within-search features on the full page
      ln(1 + price_w) AS log_price,
      percent_rank() OVER (PARTITION BY srch_id ORDER BY price_w) AS price_rank_pct,
      price_w / NULLIF(median(price_w) OVER (PARTITION BY srch_id), 0) AS price_rel_median,
      count(*) OVER (PARTITION BY srch_id) AS n_results,
      -- quality relative to the rest of the page
      NULLIF(prop_starrating, 0) - avg(NULLIF(prop_starrating, 0)) OVER (PARTITION BY srch_id) AS star_rel,
      NULLIF(prop_review_score, 0) - avg(NULLIF(prop_review_score, 0)) OVER (PARTITION BY srch_id) AS review_rel,
      prop_location_score1 - avg(prop_location_score1) OVER (PARTITION BY srch_id) AS loc1_rel,
      prop_location_score2 - avg(prop_location_score2) OVER (PARTITION BY srch_id) AS loc2_rel,
      srch_query_affinity_score,
      (srch_query_affinity_score IS NULL)::INT AS affinity_missing,
      ln(1 + orig_destination_distance) AS log_distance,
      (orig_destination_distance IS NULL)::INT AS distance_missing,
      srch_length_of_stay, srch_booking_window, srch_adults_count, srch_children_count,
      srch_room_count, srch_saturday_night_bool,
      coalesce(visitor_location_country_id = prop_country_id, false)::INT AS visitor_same_country,
      {sites},
      month(date_time::TIMESTAMP) AS search_month,
      dayofweek(date_time::TIMESTAMP) AS search_dow,
      hour(date_time::TIMESTAMP) AS search_hour,
      visitor_hist_starrating, visitor_hist_adr_usd,
      (visitor_hist_starrating IS NULL AND visitor_hist_adr_usd IS NULL)::INT AS visitor_hist_missing,
      ({comp_cheaper}) AS comp_expedia_cheaper_n,
      ({comp_dearer}) AS comp_expedia_dearer_n
    FROM base
    """


def build_features(sample: Path, out: Path, cfg: dict | None = None) -> Path:
    """Build the feature table for every row of the sample, both arms."""
    cfg = cfg or load_config()
    con = duckdb.connect()
    try:  # cosmetic; inside Jupyter DuckDB refuses to change it unless ipywidgets is installed
        con.execute("SET enable_progress_bar = false")
    except duckdb.Error:
        pass
    q_lo, q_hi = cfg["data"]["price_winsor_quantiles"]
    lo, hi = con.execute(
        f"SELECT quantile_cont(price_usd, {q_lo}), quantile_cont(price_usd, {q_hi}) FROM '{sample.as_posix()}'"
    ).fetchall()[0]
    df = con.execute(_sql(sample.as_posix(), lo, hi)).df()
    # Nullable integer columns (e.g. star rating after NULLIF) become plain floats with NaN, so
    # every downstream library sees ordinary NumPy arrays.
    nullable = [c for c, t in df.dtypes.items() if isinstance(t, pd.api.extensions.ExtensionDtype)
                and t.kind in "iuf" and c not in ("srch_id",)]
    df[nullable] = df[nullable].astype("float64")
    df = add_flags(df, cfg)
    # DuckDB sums window averages in parallel, so the last bit of a float can differ between two
    # builds of the same sample (seen in loc1_rel and loc2_rel). Rounding makes every build identical,
    # which keeps the downstream estimates reproducible to the last printed digit.
    floats = [c for c, t in df.dtypes.items() if t == "float64"]
    df[floats] = df[floats].round(9)
    df = df.sort_values(["srch_id", "position"]).reset_index(drop=True)
    df.to_parquet(out, index=False)
    return out


def add_flags(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Treatment, analysis windows, Q2 groupings. Applied after within-search features."""
    k = cfg["treatment"]["top_k"]
    reserved = set(cfg["data"]["reserved_positions"])
    pos = df["position"]
    df["top3"] = (pos <= k).astype(np.int8)
    df["reserved_position"] = pos.isin(reserved).astype(np.int8)
    # Primary Q1 window: positions 1..10 without reserved slots, so both arms compare
    # the same mix of positions.
    df["in_q1_window"] = ((pos <= cfg["treatment"]["control_max_position"]) & ~pos.isin(reserved)).astype(np.int8)
    # Sensitivity window: every non-reserved position.
    df["in_all_positions"] = (~pos.isin(reserved)).astype(np.int8)
    df["price_tercile"] = np.clip(np.ceil(df["price_rank_pct"].fillna(0) * 3), 1, 3).astype(np.int8)
    df["trip_type"] = trip_type(df, cfg)
    return df


def trip_type(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """First matching rule in config wins."""
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    for name, rule in cfg["q2"]["trip_types"].items():
        mask = out.isna() & (df.eval(rule) if rule != "True" else True)
        out[mask] = name
    return out.astype("category")


def _window_rows(df: pd.DataFrame, flag: str) -> pd.DataFrame:
    """Rows inside a window, keeping only searches with at least one control row in it, so
    every kept search contributes to the comparison."""
    w = df[df[flag] == 1]
    has_control = w.groupby("srch_id")["top3"].transform("min") == 0
    return w[has_control].reset_index(drop=True)


def q1_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows used for Q1: positions 1-3 (treated) and 4-10 minus reserved (control)."""
    return _window_rows(df, "in_q1_window")


def all_positions_rows(df: pd.DataFrame) -> pd.DataFrame:
    """The spec's Q1 sensitivity window: positions 1-3 (treated) against every non-reserved
    position from 4 down (control)."""
    return _window_rows(df, "in_all_positions")
