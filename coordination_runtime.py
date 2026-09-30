"""Agent-led delegation with controller-owned permissions and bounded execution.

The supervisor chooses work. These functions, not its text, determine which
actions can run. New data scope after retrieval requires a new approved question.
"""
from __future__ import annotations

from dataclasses import replace
import json
from typing import Any

from agents.coordinator import add_clarification
from agents.supervisor import choose_next
from artifacts import ArtifactError
from llm import FakeModelClient, reply_text
from retrieval_review import verify_reviewed_tables

MAX_PLAN_REVISIONS = 2
MAX_ANALYSIS_PASSES = 2
SUPERVISOR_PROMPT = "Answer the coordinator's question before delegation (empty = cancel): "


def _state(c: Any, stage: str) -> dict[str, Any]:
    """Only bounded evidence summaries; never authority supplied by model text."""
    retrieval = c.retrieval
    analysis = c.analysis
    return {
        "stage": stage,
        "plan": c.plan.model_dump(mode="json") if c.plan else None,
        "retrieval": None if retrieval is None else {
            "status": retrieval.status,
            "tables": [{"id": a.artifact_id, "rows": a.row_count, "complete": a.complete}
                       for a in retrieval.artifacts[:20]],
            "tables_omitted": max(0, len(retrieval.artifacts) - 20),
        },
        "data_review": c.data_review["status"],
        "analysis": None if analysis is None else {
            "status": analysis.status,
            "claims": [claim.model_dump(mode="json") for claim in analysis.claims[:20]],
            "claims_omitted": max(0, len(analysis.claims) - 20),
            "caveats": [v[:400] for v in analysis.caveats[:10]],
        },
        "history": c.supervision["decisions"],
        "plan_revisions": c.supervision["plan_revisions"],
        "analysis_passes": c.supervision["analysis_passes"],
    }


def _allowed(c: Any, stage: str) -> list[str]:
    # The phase is controller-owned, never parsed from model-provided state.
    if stage == "planned" and c.plan is not None and c.plan.status == "ready" and c.retrieval is None:
        allowed = ["retrieve", "stop"]
        if c.supervision["plan_revisions"] < MAX_PLAN_REVISIONS:
            allowed.append("revise_plan")
            if c.supervisor_answers < c.config.max_clarification_rounds:
                allowed.append("clarify")
        return allowed
    reviewed = (c.retrieval is not None and c.retrieval.status == "completed"
                and c.data_review["status"] == "approved" and c.approval is not None)
    if stage == "data_reviewed" and reviewed:
        return ["analyze", "stop"]
    if stage == "analyzed" and reviewed and c.analysis is not None and c.analysis.status == "completed":
        return (["finish", "analyze", "stop"] if c.supervision["analysis_passes"] < MAX_ANALYSIS_PASSES
                else ["finish", "stop"])
    return ["stop"]


async def _choose(c: Any, stage: str):
    if len(c.supervision["decisions"]) >= c.config.max_supervisor_turns:
        c.execution_status = "limit_reached"
        c.notes.append("coordinator delegation decision limit reached")
        return None
    allowed = _allowed(c, stage)
    c._enter("supervise")
    if c.models.supervisor is not None:
        model = c.models.supervisor
    elif c.config.mock_model:
        action = {"planned": "retrieve", "data_reviewed": "analyze", "analyzed": "finish"}[stage]
        model = FakeModelClient([reply_text(json.dumps({"status": "completed", "payload": {
            "action": action, "reason": "Synthetic offline supervisor script."}}))])
    else:
        model = await c._real_model()
    choice = await choose_next(model=model, question=c.config.question, state=_state(c, stage),
                               allowed_actions=allowed, budget=c.budget, run_id=c.run_id,
                               log_dir=c.run_dir.parent, model_requested=c.models.requested,
                               turn=len(c.supervision["decisions"]) + 1, clock=c._supervisor_clock)
    c.agents.append(choice.agent)
    c.notes.extend("supervisor: " + note for note in choice.notes)
    decision = choice.decision
    c.supervision["decisions"].append({"stage": stage, "allowed_actions": allowed,
        "status": choice.status, "decision": decision.model_dump(mode="json") if decision else None})
    if choice.status != "completed" or decision is None:
        c.execution_status = choice.status if choice.status in ("failed", "blocked", "limit_reached", "needs_clarification") else "failed"
        return None
    # Recheck at the execution boundary, independently of decision parsing.
    if decision.action not in _allowed(c, stage):
        c.execution_status = "blocked"
        c.notes.append("supervisor requested an action not permitted by the current execution state")
        return None
    if decision.action == "stop":
        c.execution_status = "blocked"
        c.notes.append("coordinator stopped the run: " + decision.reason)
        return None
    return decision


async def run_agent_led(c: Any) -> bool:
    """A bounded delegation loop; Python owns every approval and data gate."""
    while True:
        c._enter("plan")
        await c.plan_phase()
        if c.plan is None or c.plan.status != "ready":
            c._enter("clarify")
            if not await c.clarify_phase():
                return False
        c._enter("resolve")
        c.manifest = c.resolve_manifest()
        choice = await _choose(c, "planned")
        if choice is None:
            return False
        if choice.action in ("revise_plan", "clarify"):
            c.supervision["plan_revisions"] += 1
            c.supervision["plan_history"].append(c.plan.model_dump(mode="json"))
            c.approval = None
            c.manifest = None
            if choice.action == "revise_plan":
                c._supervisor_revision = choice.instruction
            else:
                c.supervisor_questions = [choice.question]
                if c.config.auto:
                    c.execution_status = "needs_clarification"
                    return False
                print("\nTHE COORDINATOR ASKS: " + choice.question)
                answer = c._ask(SUPERVISOR_PROMPT)
                if not answer or answer.lower() == "cancel":
                    c.execution_status = "canceled"
                    c.notes.append("coordinator clarification canceled; no retrieval was performed")
                    return False
                c.supervisor_answers += 1
                c.planner_questions.append(choice.question)
                c.config = replace(c.config, question=add_clarification(c.config.question, answer))
                c.supervision["clarifications"].append({"question": choice.question, "answer": answer,
                                                       "simulated": c.answers_simulated})
                c._supervisor_revision = None
            c.plan = None
            continue
        # 'retrieve' is a delegation request, never an approval.
        c._enter("approve")
        if not c.approve_phase():
            if c.pending_approval_revision is None:
                return False
            revision = c.pending_approval_revision
            c.pending_approval_revision = None
            c.config = replace(c.config, question=add_clarification(c.config.question, revision))
            c._supervisor_revision = None
            continue
        break

    c._enter("retrieve")
    await c.retrieve_phase()
    if c.retrieval is None or c.retrieval.status != "completed":
        c.execution_status = "canceled" if c.retrieval_canceled else (c.retrieval.status if c.retrieval else "failed")
        c.notes.append("retrieval did not complete; analysis was not run")
        return False
    c._enter("review_data")
    if not c.review_data_phase():
        return False
    stage = "data_reviewed"
    while True:
        choice = await _choose(c, stage)
        if choice is None:
            return False
        if choice.action == "finish":
            try:
                verify_reviewed_tables(c.ctx, c.retrieval, c.data_review["artifacts"])
            except ArtifactError as exc:
                c.execution_status = "blocked"
                c.notes.append("reviewed data changed before final rendering: " + str(exc))
                return False
            c._enter("render")
            c.render_phase()
            c.execution_status = "completed"
            c._enter("accept")
            c.accept_phase()
            return True
        # A supervisor can ask for another calculation pass on the SAME reviewed
        # evidence, but cannot fetch new data or supply a new artifact allowlist.
        try:
            verify_reviewed_tables(c.ctx, c.retrieval, c.data_review["artifacts"])
        except ArtifactError as exc:
            c.execution_status = "blocked"
            c.notes.append("reviewed data changed before analysis: " + str(exc))
            return False
        c._supervisor_analysis_instruction = choice.instruction
        c.supervision["analysis_passes"] += 1
        c._enter("analyze")
        await c.analyze_phase()
        if c.analysis is not None:
            c.supervision["analysis_history"].append(c.analysis.model_dump(mode="json"))
        if c.analysis is None or c.analysis.status != "completed":
            c.execution_status = "canceled" if c.analysis_canceled else (c.analysis.status if c.analysis else "failed")
            return False
        stage = "analyzed"
