"""What the agent knows about each dataset: descriptions, how to load an arm, which rows a
question uses, and the reference answer used by the offline (--no-llm) stubs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .. import data, features
from ..config import load_config, load_dictionary, load_spec
from ..synthetic import Scenario, trap_scenarios
from .state import DagProposal, Edge, Excluded, QuestionSpec, VariableRole


@dataclass
class DatasetContext:
    name: str
    description: str
    dictionary: dict[str, str]
    group: str
    arms: tuple[str, ...]
    load: Callable[[str], pd.DataFrame]          # arm -> data; wrapped by the firewall in nodes
    prepare: Callable[[pd.DataFrame, QuestionSpec], pd.DataFrame]
    offline_intake: Callable[[str], QuestionSpec]
    reference_dag: Callable[[QuestionSpec], DagProposal]
    has_benchmark: Callable[[QuestionSpec], bool]
    held_out: str | None = None
    true_effect: float | None = None
    scenario: Scenario | None = None


# ---------------------------------------------------------------- Expedia

def _expedia_intake(question: str) -> QuestionSpec:
    q = question.lower()
    if "promotion" in q or "badge" in q:
        return QuestionSpec(treatment="promotion_flag", outcome="booking_bool" if "book" in q and "click" not in q else "click_bool",
                            unit="one hotel shown in one search", population="randomly ranked searches, all positions",
                            target_arm="experimental")
    return QuestionSpec(treatment="top3", outcome="booking_bool" if "book" in q and "click" not in q else "click_bool",
                        unit="one hotel shown in one search",
                        population="algorithm-ranked searches, positions 1-10 without reserved slots",
                        target_arm="observational")


def _expedia_reference(spec: QuestionSpec) -> DagProposal:
    """The reference graph of Section 4.3, built from config/spec_locked.yaml."""
    ref = load_spec()
    conf = [c for c in ref["adjustment_set"] if c != spec.treatment]
    roles = [VariableRole(name=c, role="confounder", reason="drives both placement and clicks") for c in conf]
    edges = [Edge(source=c, target=spec.treatment) for c in conf] + [Edge(source=c, target=spec.outcome) for c in conf]
    edges.append(Edge(source=spec.treatment, target=spec.outcome))
    adjust = list(conf)
    excluded = [Excluded(name="gross_bookings_usd", reason="post-outcome")]
    if spec.treatment == "top3":
        edges += [Edge(source="position", target="top3"), Edge(source="click_bool", target="booking_bool"),
                  Edge(source="booking_bool", target="gross_bookings_usd")]
        excluded += [Excluded(name="position", reason="defines the treatment"),
                     Excluded(name="booking_bool", reason="downstream of the click")]
        roles += [VariableRole(name="position", role="design", reason="defines top3")]
    else:
        # Q3: position is random in this arm, so it is a precision covariate, not a confounder
        adjust.append("position")
        edges += [Edge(source="position", target=spec.outcome)]
        roles += [VariableRole(name="position", role="precision", reason="randomised in this arm")]
        if spec.outcome == "booking_bool":
            edges += [Edge(source=spec.treatment, target="click_bool"), Edge(source="click_bool", target="booking_bool")]
            excluded.append(Excluded(name="click_bool", reason="mediator between promotion and booking"))
        else:
            edges += [Edge(source="click_bool", target="booking_bool")]
            excluded.append(Excluded(name="booking_bool", reason="downstream of the click"))
        edges.append(Edge(source="booking_bool", target="gross_bookings_usd"))
    roles += [VariableRole(name=spec.treatment, role="treatment", reason="the question"),
              VariableRole(name=spec.outcome, role="outcome", reason="the question")]
    threats = (["ranker_score: Expedia's internal ranking score"] if spec.treatment == "top3" else
               ["hotel_strategy: why a hotel runs a promotion (occupancy, pricing plans), not in the file"])
    return DagProposal(roles=roles, edges=edges, adjustment_set=adjust, excluded=excluded, unobserved_threats=threats)


def expedia(synthetic: bool = False, cfg: dict | None = None) -> DatasetContext:
    cfg = cfg or load_config()
    d = load_dictionary(cfg)

    def prepare(df: pd.DataFrame, spec: QuestionSpec) -> pd.DataFrame:
        if spec.treatment == "top3":
            return features.q1_rows(df)
        return df.reset_index(drop=True)  # Q3 keeps all positions

    return DatasetContext(
        name="expedia_synthetic" if synthetic else "expedia",
        description=d["dataset"] + " " + d["context"],
        dictionary=d["variables"], group="srch_id", arms=("observational", "experimental"),
        load=lambda arm: data.loader_for(arm)(cfg, synthetic),
        prepare=prepare, offline_intake=_expedia_intake, reference_dag=_expedia_reference,
        has_benchmark=lambda s: s.treatment == "top3",
        held_out=load_spec(cfg).get("held_out_covariate"),
    )


# ---------------------------------------------------------------- trap scenarios

def trap(name: str, seed: int = 0, cfg: dict | None = None) -> DatasetContext:
    cfg = cfg or load_config()
    sc = next(s for s in trap_scenarios(n=cfg["trap_suite"]["rows"], seed=seed) if s.name == name)

    def load(arm: str) -> pd.DataFrame:
        if arm != "observational":
            raise PermissionError("Trap scenarios have no experimental arm.")
        return sc.df.copy()

    def reference(spec: QuestionSpec) -> DagProposal:
        """Offline stub for traps = the hidden reference answer. Offline trap runs therefore
        test the plumbing only; the real test of judgement needs Claude."""
        roles = [VariableRole(name=c, role="confounder", reason="reference") for c in sc.reference_adjustment]
        roles += [VariableRole(name=b, role="mediator" if b == "detail_page_seconds" else "collider", reason="reference")
                  for b in sc.bad_controls]
        return DagProposal(roles=roles, edges=[Edge(source=a, target=b) for a, b in sc.reference_edges()],
                           adjustment_set=list(sc.reference_adjustment),
                           excluded=[Excluded(name=b, reason="bad control") for b in sc.bad_controls])

    return DatasetContext(
        name=f"trap:{name}", description="Synthetic product-analytics data about a featured listing slot.",
        dictionary=sc.dictionary, group=sc.group, arms=("observational",), load=load,
        prepare=lambda df, spec: df,
        offline_intake=lambda q: QuestionSpec(treatment=sc.treatment, outcome=sc.outcome, unit="one listing in one session",
                                              population="all sessions", target_arm="observational"),
        reference_dag=reference, has_benchmark=lambda s: False, true_effect=sc.true_effect, scenario=sc,
    )


def get(dataset: str, synthetic: bool = False, seed: int = 0, cfg: dict | None = None) -> DatasetContext:
    if dataset.startswith("trap:"):
        return trap(dataset.split(":", 1)[1], seed=seed, cfg=cfg)
    return expedia(synthetic=synthetic, cfg=cfg)
