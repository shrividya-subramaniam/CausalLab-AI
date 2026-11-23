"""Synthetic data with a known answer.

Two generators:
  * expedia_like(): a fake file with the same columns as the Kaggle train.csv, so the whole
    pipeline runs from a fresh clone without the Kaggle file (--synthetic). The true top-3
    effect is known. It also has an algorithm arm where a ranker puts better hotels on top.
  * trap scenarios: five small datasets for the trap suite (Phase 8). Each has plausible
    product-analytics variable names and descriptions, a hidden reference answer, a true
    effect, and the behaviour we expect from a good analyst.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

RAW_COLUMNS = [
    "srch_id", "date_time", "site_id", "visitor_location_country_id", "visitor_hist_starrating",
    "visitor_hist_adr_usd", "prop_country_id", "prop_id", "prop_starrating", "prop_review_score",
    "prop_brand_bool", "prop_location_score1", "prop_location_score2", "prop_log_historical_price",
    "position", "price_usd", "promotion_flag", "srch_destination_id", "srch_length_of_stay",
    "srch_booking_window", "srch_adults_count", "srch_children_count", "srch_room_count",
    "srch_saturday_night_bool", "srch_query_affinity_score", "orig_destination_distance", "random_bool",
    *[f"comp{i}_{k}" for i in range(1, 9) for k in ("rate", "inv", "rate_percent_diff")],
    "click_bool", "gross_bookings_usd", "booking_bool",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def expedia_like(n_searches: int = 6000, hotels_per_search: int = 25, seed: int = 0,
                 position_effect: float = 0.9) -> tuple[pd.DataFrame, float]:
    """Return (raw-schema DataFrame, true top-3 ATE on the click probability).

    Click model: logit P(click) = -3.4 + quality + position_effect * top3 - 0.08 * max(position - 3, 0).
    In the algorithm arm the ranker sorts by quality plus noise, so better hotels sit on top and
    the naive comparison is biased upward."""
    rng = np.random.default_rng(seed)
    n = n_searches * hotels_per_search
    sid = np.repeat(np.arange(1, n_searches + 1), hotels_per_search)
    random_bool = np.repeat(rng.random(n_searches) < 0.3, hotels_per_search).astype(int)
    star = rng.integers(1, 6, n).astype(float)
    review = np.clip(np.round(rng.normal(3.8, 0.6, n) * 2) / 2, 0, 5)
    loc1 = rng.uniform(0, 7, n)
    loc2 = np.clip(rng.beta(2, 10, n), 0, 1)
    price = np.exp(rng.normal(4.8, 0.5, n) + 0.15 * star)
    promo = (rng.random(n) < 0.22).astype(int)
    quality = 0.35 * (star - 3) + 0.5 * (review - 3.8) + 3.0 * (loc2 - 0.17) + 0.08 * (loc1 - 3.5) \
        - 0.3 * (np.log(price) - 5.2) + 0.3 * promo
    score = quality + rng.normal(0, 0.6, n)  # the ranker's view; noise is its private information
    df = pd.DataFrame({"srch_id": sid, "random_bool": random_bool, "quality": quality, "score": score})
    df["rand"] = rng.random(n)
    key = np.where(df["random_bool"] == 1, df["rand"], -df["score"])
    df["position"] = pd.Series(key).groupby(df["srch_id"]).rank(method="first").astype(int).to_numpy()
    pos = df["position"].to_numpy()
    top3 = (pos <= 3).astype(float)
    decay = -0.08 * np.clip(pos - 3, 0, None)
    lin = -3.4 + quality + decay
    click = (rng.random(n) < _sigmoid(lin + position_effect * top3)).astype(int)
    book = ((rng.random(n) < 0.3) & (click == 1)).astype(int)
    window = np.repeat(rng.integers(0, 120, n_searches), hotels_per_search)
    out = pd.DataFrame({
        "srch_id": sid,
        "date_time": pd.Timestamp("2013-01-01") + pd.to_timedelta(np.repeat(rng.integers(0, 180 * 24, n_searches), hotels_per_search), unit="h"),
        "site_id": np.repeat(rng.choice([5, 14, 15, 24, 32, 7], n_searches), hotels_per_search),
        "visitor_location_country_id": np.repeat(rng.choice([219, 100, 55], n_searches), hotels_per_search),
        "visitor_hist_starrating": np.where(rng.random(n) < 0.95, np.nan, rng.uniform(2, 5, n)),
        "visitor_hist_adr_usd": np.where(rng.random(n) < 0.95, np.nan, rng.uniform(50, 300, n)),
        "prop_country_id": np.repeat(rng.choice([219, 100, 55], n_searches), hotels_per_search),
        "prop_id": rng.integers(1, 140000, n),
        "prop_starrating": star, "prop_review_score": review,
        "prop_brand_bool": (rng.random(n) < 0.6).astype(int),
        "prop_location_score1": loc1, "prop_location_score2": np.where(rng.random(n) < 0.2, np.nan, loc2),
        "prop_log_historical_price": np.where(rng.random(n) < 0.14, 0.0, np.log(price) + rng.normal(0, 0.1, n)),
        "position": pos, "price_usd": price, "promotion_flag": promo,
        "srch_destination_id": np.repeat(rng.integers(1, 30000, n_searches), hotels_per_search),
        "srch_length_of_stay": np.repeat(rng.integers(1, 6, n_searches), hotels_per_search),
        "srch_booking_window": window,
        "srch_adults_count": np.repeat(rng.integers(1, 4, n_searches), hotels_per_search),
        "srch_children_count": np.repeat(rng.integers(0, 3, n_searches), hotels_per_search),
        "srch_room_count": np.repeat(rng.integers(1, 3, n_searches), hotels_per_search),
        "srch_saturday_night_bool": np.repeat(rng.integers(0, 2, n_searches), hotels_per_search),
        "srch_query_affinity_score": np.where(rng.random(n) < 0.93, np.nan, rng.normal(-24, 10, n)),
        "orig_destination_distance": np.where(rng.random(n) < 0.3, np.nan, rng.exponential(1000, n)),
        "random_bool": random_bool,
    })
    for i in range(1, 9):
        out[f"comp{i}_rate"] = np.where(rng.random(n) < 0.6, np.nan, rng.choice([-1, 0, 1], n))
        out[f"comp{i}_inv"] = np.where(rng.random(n) < 0.6, np.nan, rng.choice([0, 1], n))
        out[f"comp{i}_rate_percent_diff"] = np.where(rng.random(n) < 0.8, np.nan, rng.uniform(1, 50, n))
    out["click_bool"] = click
    out["gross_bookings_usd"] = np.where(book == 1, price * out["srch_length_of_stay"], np.nan)
    out["booking_bool"] = book
    # Unlike the real file, searches without a click are kept, so the true effect below is
    # exactly what the randomly ranked arm estimates: top-3 versus control positions 4, 6-10.
    q = quality
    p_top = _sigmoid(-3.4 + q + position_effect)
    p_ctrl = np.mean([_sigmoid(-3.4 + q - 0.08 * (k - 3)) for k in (4, 6, 7, 8, 9, 10)], axis=0)
    true_ate = float(np.mean(p_top - p_ctrl))
    # The ranker fills the top ten with better hotels, and better hotels gain more from a top slot,
    # so the true average effect differs by population. Q1 rows = positions 1-10 without slot 5.
    row_effect = p_top - p_ctrl
    win = (pos <= 10) & (pos != 5)
    truths = {"all_hotels": true_ate,
              "observational_q1_rows": float(row_effect[win & (random_bool == 0)].mean()),
              "experimental_q1_rows": float(row_effect[win & (random_bool == 1)].mean())}
    res = out[RAW_COLUMNS].copy()
    res.attrs["truths"] = truths
    return res, true_ate


# ---------------------------------------------------------------- trap suite

@dataclass
class Scenario:
    name: str
    df: pd.DataFrame
    dictionary: dict[str, str]           # shown to the agent: names and descriptions only
    treatment: str
    outcome: str
    group: str
    true_effect: float
    reference_adjustment: list[str]      # hidden from the agent
    bad_controls: list[str] = field(default_factory=list)
    expected: str = "proceed"            # proceed | refuse | respecify_then_refuse
    question: str = ""

    def reference_edges(self) -> list[tuple[str, str]]:
        e = [(c, self.treatment) for c in self.reference_adjustment] + \
            [(c, self.outcome) for c in self.reference_adjustment] + [(self.treatment, self.outcome)]
        for b in self.bad_controls:
            if b == "detail_page_seconds":
                e += [(self.treatment, b), (b, self.outcome)]
            if b == "support_ticket_opened":
                e += [(self.treatment, b), (self.outcome, b)]
        return e


BASE_DICT = {
    "session_id": "Identifier of the browsing session. Several listings are shown per session.",
    "featured_slot": "1 if the listing was placed in the featured slot at the top of the page for this session.",
    "clicked": "1 if the user clicked the listing in this session.",
    "listing_quality": "Quality score of the listing computed from reviews collected before the session.",
    "listing_price": "Nightly price shown for the listing in this session.",
    "review_count": "Number of reviews the listing had before the session (standardised).",
    "photo_count": "Number of photos on the listing before the session (standardised).",
    "host_response_rate": "Share of messages the host answered in the month before the session.",
    "user_tenure_days": "Days since the user created an account, measured at the start of the session.",
}
CONFOUNDERS = ["listing_quality", "listing_price", "review_count", "photo_count", "host_response_rate"]
# Several moderate confounders rather than one dominant one: together they bias the naive
# comparison, but no single one is strong enough to fail the sensitivity rule.
T_WEIGHTS = np.array([0.35, -0.30, 0.30, 0.25, 0.25])
Y_WEIGHTS = np.array([0.30, -0.20, 0.25, 0.20, 0.20])


def _base(n: int, rng: np.random.Generator, per_group: int = 5) -> pd.DataFrame:
    d = pd.DataFrame({"session_id": np.repeat(np.arange(n // per_group + 1), per_group)[:n]})
    for c in CONFOUNDERS + ["user_tenure_days"]:
        d[c] = rng.normal(0, 1, n)
    return d


def _assign(d: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    return (rng.random(len(d)) < _sigmoid(-0.6 + d[CONFOUNDERS].to_numpy() @ T_WEIGHTS)).astype(int)


def _lin(d: pd.DataFrame) -> np.ndarray:
    return -1.6 + d[CONFOUNDERS].to_numpy() @ Y_WEIGHTS + 0.3 * d["user_tenure_days"].to_numpy()


def _outcome(df: pd.DataFrame, t: np.ndarray, rng: np.random.Generator, effect: float, extra: np.ndarray | float = 0.0):
    lin = _lin(df) + extra
    p1, p0 = _sigmoid(lin + effect), _sigmoid(lin)
    y = (rng.random(len(df)) < np.where(t == 1, p1, p0)).astype(int)
    return y, float(np.mean(p1 - p0))


def trap_scenarios(n: int = 20000, seed: int = 0) -> list[Scenario]:
    rng = np.random.default_rng(seed)
    out = []
    conf = CONFOUNDERS + ["user_tenure_days"]
    q = "Does placing a listing in the featured slot cause more clicks?"

    # 1. clean confounding
    d = _base(n, rng)
    t = _assign(d, rng)
    d["featured_slot"] = t
    d["clicked"], te = _outcome(d, t, rng, 0.6)
    out.append(Scenario("clean_confounding", d, dict(BASE_DICT), "featured_slot", "clicked", "session_id", te, conf,
                        expected="proceed", question=q))

    # 2. no overlap: the slot is decided by a hard quality cut-off
    d = _base(n, rng)
    t = (d["listing_quality"].to_numpy() > 0.3).astype(int)  # a hard placement rule
    d["featured_slot"] = t
    d["clicked"], te = _outcome(d, t, rng, 0.6)
    dd = dict(BASE_DICT)
    dd["featured_slot"] = "1 if the listing was placed in the featured slot. The merchandising team features listings by a quality rule."
    out.append(Scenario("no_overlap", d, dd, "featured_slot", "clicked", "session_id", te, conf, expected="refuse", question=q))

    # 3. collider offered as a control
    d = _base(n, rng)
    t = _assign(d, rng)
    d["featured_slot"] = t
    d["clicked"], te = _outcome(d, t, rng, 0.6)
    d["support_ticket_opened"] = (rng.random(n) < _sigmoid(-2.5 + 1.5 * t + 2.0 * d["clicked"])).astype(int)
    dd = dict(BASE_DICT)
    dd["support_ticket_opened"] = "1 if the user opened a customer-support ticket about this listing after the session ended."
    out.append(Scenario("collider", d, dd, "featured_slot", "clicked", "session_id", te, conf,
                        bad_controls=["support_ticket_opened"], expected="proceed", question=q))

    # 4. mediator offered as a control
    d = _base(n, rng)
    t = _assign(d, rng)
    d["featured_slot"] = t
    secs = np.exp(1.5 + 0.8 * t + 0.2 * d["listing_quality"] + rng.normal(0, 0.5, n))
    d["detail_page_seconds"] = secs
    d["clicked"], te = _outcome(d, t, rng, 0.3, extra=0.6 * (np.log(secs) - 1.5))
    # total effect: through the slot directly and through visible seconds. The seconds carry noise
    # (sd 0.5 on the log scale), so the truth averages over it (Gauss-Hermite quadrature).
    lin = _lin(d)
    quality = d["listing_quality"].to_numpy()
    nodes, weights = np.polynomial.hermite_e.hermegauss(40)
    weights = weights / weights.sum()
    te = 0.0
    for z, w in zip(nodes, weights):
        noise = 0.5 * z
        s1 = 0.8 + 0.2 * quality + noise   # log(seconds) - 1.5 when featured
        s0 = 0.2 * quality + noise         # log(seconds) - 1.5 when not featured
        te += w * float(np.mean(_sigmoid(lin + 0.3 + 0.6 * s1) - _sigmoid(lin + 0.6 * s0)))
    dd = dict(BASE_DICT)
    dd["detail_page_seconds"] = "Seconds the listing stayed visible on the user's screen during the session."
    out.append(Scenario("mediator", d, dd, "featured_slot", "clicked", "session_id", te, conf,
                        bad_controls=["detail_page_seconds"], expected="proceed", question=q))

    # 5. balance cannot be reached: few sessions, many confounders acting together
    m = 3000
    k = 30
    d = pd.DataFrame({"session_id": np.repeat(np.arange(m // 5), 5)[:m]})
    signals = rng.normal(0, 1, (m, k))
    names = [f"signal_{i:02d}" for i in range(1, k + 1)]
    for i, nm in enumerate(names):
        d[nm] = signals[:, i]
    index = signals @ np.full(k, 0.35)
    t = (rng.random(m) < _sigmoid(index)).astype(int)
    d["featured_slot"] = t
    p0 = _sigmoid(-1.5 + 0.15 * index)
    p1 = _sigmoid(-1.5 + 0.15 * index + 0.6)
    d["clicked"] = (rng.random(m) < np.where(t == 1, p1, p0)).astype(int)
    dd = {"session_id": BASE_DICT["session_id"], "featured_slot": BASE_DICT["featured_slot"], "clicked": BASE_DICT["clicked"]}
    dd.update({nm: f"Pre-session engagement signal number {i} for this user and listing, measured before the session."
               for i, nm in enumerate(names, 1)})
    out.append(Scenario("failing_balance", d, dd, "featured_slot", "clicked", "session_id", float(np.mean(p1 - p0)), names,
                        expected="respecify_then_refuse", question=q))
    return out
