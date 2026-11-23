"""Typed graph state (Pydantic).

Numbers travel here, in typed fields filled by code. Claude receives them read-only and
never writes a number into state or into the report.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["treatment", "outcome", "confounder", "precision", "mediator", "collider",
               "post_outcome", "instrument", "design", "irrelevant"]


class QuestionSpec(BaseModel):
    """What intake extracts from the plain-language question."""
    treatment: str = Field(description="Name of the treatment column")
    outcome: str = Field(description="Name of the outcome column")
    unit: str = Field(description="The unit of analysis, e.g. one hotel shown in one search")
    population: str = Field(description="Which rows the question is about")
    target_arm: Literal["observational", "experimental"] = Field(
        description="observational = algorithm-ranked searches; experimental = randomly ranked searches")


class VariableRole(BaseModel):
    name: str
    role: Role
    reason: str = Field(description="One line")


class Edge(BaseModel):
    source: str
    target: str


class Excluded(BaseModel):
    name: str
    reason: str


class DagProposal(BaseModel):
    """propose_dag's structured output. The graph is built from `edges`, never from roles."""
    roles: list[VariableRole]
    edges: list[Edge] = Field(description="Explicit directed edges, cause -> effect")
    adjustment_set: list[str]
    excluded: list[Excluded]
    unobserved_threats: list[str] = Field(default_factory=list)
    interactions: list[list[str]] = Field(default_factory=list,
                                          description="Pairs of adjustment variables to multiply, only on re-specification")
    trim: list[float] | None = Field(default=None, description="Overlap trimming range, only tightened on re-specification")


class Review(BaseModel):
    decision: Literal["approve", "reject", "edit"]
    adjustment_set: list[str] | None = None
    notes: str = ""


class AgentState(BaseModel):
    run_id: str
    question: str
    dataset: str = "expedia"            # expedia | trap:<scenario name>
    offline: bool = False
    synthetic: bool = False
    auto_approve: bool = False
    run_heterogeneity: bool = False
    exploratory: bool = False           # labelled in the report; set for any run made after seeing the benchmark
    adjustment_override: list[str] | None = None  # exploratory runs only

    spec: QuestionSpec | None = None
    profile: dict[str, Any] | None = None
    dag: DagProposal | None = None
    review: Review | None = None
    approved_confounders: list[str] = Field(default_factory=list)
    identification: dict[str, Any] | None = None
    design: dict[str, Any] | None = None

    estimates: dict[str, dict[str, Any]] = Field(default_factory=dict)
    estimation_info: dict[str, Any] = Field(default_factory=dict)
    benchmark: dict[str, Any] | None = None
    gap: dict[str, Any] | None = None
    population_checks: dict[str, Any] | None = None   # post-hoc like-for-like comparisons
    refutations: dict[str, dict[str, Any]] = Field(default_factory=dict)
    verdict: dict[str, Any] | None = None
    heterogeneity: dict[str, Any] | None = None

    attempts: int = 0
    history: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    report_path: str | None = None
    outcome_status: str | None = None   # reported | refused
