"""Graph assembly: nodes, routers, the human-review interrupt and the SQLite checkpointer.

    intake -> profile_data -> propose_dag -> human_review -> identify -> route_design
           -> estimate -> refute -> judge -> (heterogeneity) -> narrate
    judge -> propose_dag (balance failed, at most twice, back through human_review)
    any failed gate -> refuse
"""
from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from ..config import load_config, results_dir
from . import nodes as N
from .state import AgentState


def _route_after_review(state: AgentState) -> str:
    return "refuse" if state.review and state.review.decision == "reject" else "identify"


def _route_after_identify(state: AgentState) -> str:
    return "route_design" if state.identification and state.identification["valid"] else "refuse"


def _route_after_design(state: AgentState) -> str:
    return "estimate" if state.design and state.design["proceed"] else "refuse"


def _route_after_judge(state: AgentState) -> str:
    route = state.verdict["route"]
    if route == "pass":
        return "heterogeneity" if state.run_heterogeneity else "narrate"
    return "propose_dag" if route == "respecify" else "refuse"


def build_graph(rt: N.Runtime, checkpointer=None):
    g = StateGraph(AgentState)
    for name in ("intake", "profile_data", "propose_dag", "human_review", "identify", "route_design",
                 "estimate", "refute", "judge", "heterogeneity", "narrate", "refuse"):
        g.add_node(name, N.traced(rt, name, getattr(N, name)))

    def logged(name, fn):
        def router(state: AgentState) -> str:
            dest = fn(state)
            rt.log(state.run_id, name, "route", dest)
            return dest
        return router

    g.add_edge(START, "intake")
    g.add_edge("intake", "profile_data")
    g.add_edge("profile_data", "propose_dag")
    g.add_edge("propose_dag", "human_review")
    g.add_conditional_edges("human_review", logged("after_review", _route_after_review), ["identify", "refuse"])
    g.add_conditional_edges("identify", logged("after_identify", _route_after_identify), ["route_design", "refuse"])
    g.add_conditional_edges("route_design", logged("after_design", _route_after_design), ["estimate", "refuse"])
    g.add_edge("estimate", "refute")
    g.add_edge("refute", "judge")
    g.add_conditional_edges("judge", logged("after_judge", _route_after_judge),
                            ["heterogeneity", "narrate", "propose_dag", "refuse"])
    g.add_edge("heterogeneity", "narrate")
    g.add_edge("narrate", END)
    g.add_edge("refuse", END)
    return g.compile(checkpointer=checkpointer)


def checkpointer(cfg: dict | None = None) -> SqliteSaver:
    """SQLite file, so a paused run survives closing the terminal and can be resumed."""
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    path: Path = results_dir(cfg) / "checkpoints.sqlite"
    allowed = [("causallab.agent.state", n) for n in
               ("AgentState", "QuestionSpec", "VariableRole", "Edge", "Excluded", "DagProposal", "Review")]
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False),
                       serde=JsonPlusSerializer(allowed_msgpack_modules=allowed))


@contextmanager
def _open_checkpointer(cfg: dict | None):
    """The checkpointer for one start or resume, its database connection closed afterwards
    (an open handle would keep results/checkpoints.sqlite locked on Windows)."""
    saver = checkpointer(cfg)
    try:
        yield saver
    finally:
        saver.conn.close()


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def start(question: str, *, dataset: str = "expedia", offline: bool = False, synthetic: bool = False,
          auto_approve: bool = False, run_heterogeneity: bool = False, run_id: str | None = None,
          exploratory: bool = False, adjustment_override: list[str] | None = None,
          cfg: dict | None = None) -> dict[str, Any]:
    """Start a run. Returns the final state, or the interrupt payload if it paused for review."""
    cfg = cfg or load_config()
    run_id = run_id or new_run_id()
    rt = N.make_runtime(run_id, cfg)
    init = AgentState(run_id=run_id, question=question, dataset=dataset, offline=offline, synthetic=synthetic,
                      auto_approve=auto_approve, run_heterogeneity=run_heterogeneity, exploratory=exploratory,
                      adjustment_override=adjustment_override)
    with _open_checkpointer(cfg) as saver:
        graph = build_graph(rt, saver)
        out = graph.invoke(init, {"configurable": {"thread_id": run_id}})
        return _result(graph, run_id, out)


def resume(run_id: str, decision: dict[str, Any] | None, cfg: dict | None = None) -> dict[str, Any]:
    """Resume a paused run with the reviewer's decision (approve, edit or reject).
    If the run is not paused but stopped on an error, this retries from the last checkpoint."""
    cfg = cfg or load_config()
    rt = N.make_runtime(run_id, cfg)
    config = {"configurable": {"thread_id": run_id}}
    with _open_checkpointer(cfg) as saver:
        graph = build_graph(rt, saver)
        snap = graph.get_state(config)
        paused = any(t.interrupts for t in snap.tasks)
        out = graph.invoke(Command(resume=decision) if paused else None, config)
        return _result(graph, run_id, out)


def _result(graph, run_id: str, out: dict[str, Any]) -> dict[str, Any]:
    snap = graph.get_state({"configurable": {"thread_id": run_id}})
    pending = [i.value for t in snap.tasks for i in (t.interrupts or [])]
    return {"run_id": run_id, "paused": bool(pending), "review_request": pending[0] if pending else None,
            "state": out}


def export_diagram(path: Path, cfg: dict | None = None) -> Path:
    """Mermaid source of the graph (Phase 5 acceptance check)."""
    rt = N.make_runtime("diagram", cfg)
    path.write_text(build_graph(rt).get_graph().draw_mermaid(), encoding="utf-8")
    return path
