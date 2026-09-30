"""Deterministic, source-backed study/variable metadata answers; no model or HTTP calls."""
from __future__ import annotations

from datetime import datetime
import html
import re
from typing import Any

from agents.coordinator import person_words
from brapi_mcp_server import materialize_metadata_result
from contracts import AnalysisReport, MetadataFact, Plan, RetrievalReport

MAX_METADATA_ENTITIES = 20
IDENTITY = {"studies": ("study", "studyDbId", "studyName"),
            "variables": ("variable", "observationVariableDbId", "observationVariableName")}
STUDY_FIELDS = {
    "studyDbId": "Database ID", "studyName": "Study name", "studyDescription": "Description",
    "studyType": "Study type", "commonCropName": "Crop", "locationDbId": "Location ID",
    "locationName": "Location", "seasons": "Season(s)", "startDate": "Study start date",
    "endDate": "Study end date", "plantingDate": "Planting date", "harvestDate": "Harvest date",
    "experimentalDesignDescription": "Experimental design", "experimentalDesignPUI": "Design identifier",
    "programDbId": "Program ID", "programName": "Program name", "trialDbId": "Trial ID", "trialName": "Trial name",
    "additionalInfoProgramDbId": "Program ID (additionalInfo)", "additionalInfoProgramName": "Program name (additionalInfo)",
}
VARIABLE_FIELDS = {
    "observationVariableDbId": "Database ID", "observationVariableName": "Variable name",
    "traitName": "Trait", "traitDescription": "Trait description", "methodName": "Method name",
    "methodDescription": "Method description", "units": "Units (structured field)", "scaleName": "Scale name",
    "dataType": "Scale data type", "scaleValidValues": "Scale valid values", "timepoint": "Timepoint",
}


def metadata_only_plan(plan: Plan | None) -> bool:
    """Do not redirect numeric, counting, observation, or unsupported operations into this path."""
    if plan is None or plan.statistic is not None:
        return False
    retrieval = [s for s in plan.steps if s.agent == "retriever"]
    if not retrieval:
        return False
    for step in plan.steps:
        if step.agent == "analyst":
            if step.action not in {"filter_rows", "table_info"}:
                return False
        elif step.agent == "retriever":
            if step.action not in {"get_study", "search_studies", "list_variables", "export_metadata"}:
                return False
            if step.action == "export_metadata" and step.inputs.get("entity") not in IDENTITY:
                return False
        else:
            return False
    return True


def materialize_metadata(plan: Plan, tools: Any, ctx: Any) -> None:
    """After any MCP registry reload, before build_report; touches only this run's metadata evidence."""
    if not metadata_only_plan(plan):
        return
    for record in tools.records:
        if getattr(record, "blocked_here", False) or getattr(record, "refused_here", False):
            continue
        record.result = materialize_metadata_result(ctx, record.name, record.args, record.result)


def _clarify(message: str) -> AnalysisReport:
    return AnalysisReport(status="needs_clarification", methods=["metadata: read exact source rows without statistical analysis"],
                          caveats=["analyst asks: " + message])


def _mentioned(question: str, text: str) -> bool:
    return bool(text and re.search(r"(?<!\w)" + re.escape(text) + r"(?!\w)", question, re.IGNORECASE))


def _dates(request_ids: list[str], requests: list[dict[str, Any]]) -> list[datetime]:
    dates: set[datetime] = set()
    for request in requests:
        raw = request.get("fetched_at_utc")
        if request.get("request_id") not in request_ids or not isinstance(raw, str):
            continue
        try:
            date = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if date.tzinfo is not None:
                dates.add(date)
        except ValueError:
            continue
    return sorted(dates)


def validate_metadata_facts(facts: list[MetadataFact], registry: Any) -> None:
    """Prove values against loaded, hash-checked CSV rows; schema validation alone cannot do that."""
    loaded: dict[str, Any] = {}
    for fact in facts:
        if fact.source_artifact_id not in loaded:
            loaded[fact.source_artifact_id] = registry.load(fact.source_artifact_id)
        manifest, rows = loaded[fact.source_artifact_id]
        wanted_kind = "studies" if fact.entity_kind == "study" else "variables"
        if manifest.kind != wanted_kind or not manifest.complete or manifest.sha256 != fact.source_artifact_hash:
            raise ValueError("metadata source hash, completeness or entity kind does not match")
        if fact.field not in manifest.columns or fact.source_request_ids != manifest.source_request_ids:
            raise ValueError("metadata source field or request IDs do not match")
        _, id_column, name_column = IDENTITY[wanted_kind]
        matches = [r for r in rows if r.get(id_column) == fact.entity_id]
        if len(matches) != 1:
            raise ValueError("metadata entity must identify exactly one source row")
        row = matches[0]
        if fact.value != (row.get(fact.field) or None) or fact.entity_label != (row.get(name_column) or None):
            raise ValueError("metadata value or label differs from its exact source row")


def analyze_metadata(plan: Plan, retrieval: RetrievalReport, registry: Any, *,
                     requests: list[dict[str, Any]] | None = None) -> AnalysisReport:
    if not metadata_only_plan(plan):
        return _clarify("This plan includes analysis beyond supported study or variable metadata lookup.")
    if not retrieval.complete or retrieval.status != "completed":
        return AnalysisReport(status="incomplete", caveats=["Metadata retrieval is incomplete; no complete metadata answer was produced."])
    artifacts = [a for a in retrieval.artifacts if a.kind in IDENTITY]
    if not artifacts:
        return _clarify("No source-backed study or variable record was retrieved. Please specify the exact ID or name; this does not prove the record is absent.")
    expected = {"studies": set(plan.scope.study_ids), "variables": set(plan.scope.variable_ids)}
    for step in plan.steps:
        if step.agent == "retriever" and step.action == "get_study" and isinstance(step.inputs.get("study_db_id"), str):
            expected["studies"].add(step.inputs["study_db_id"])
    filters: list[tuple[str, str]] = []
    for step in plan.steps:
        if step.agent != "analyst" or step.action != "filter_rows":
            continue
        column, operator, value = (step.inputs.get(k) for k in ("column", "op", "value"))
        # This bounded path handles exact entity filters, not arbitrary catalog analysis.
        if (operator != "eq" or column not in {"studyDbId", "studyName", "observationVariableDbId", "observationVariableName"}
                or not isinstance(value, str) or not value):
            return _clarify("Please specify an exact study or variable ID/name for this metadata lookup; the planned filter is not supported here.")
        filters.append((column, value))
    question_words = person_words(plan.question)
    selected: dict[tuple[str, str], tuple[Any, dict[str, str]]] = {}
    all_ids = {"studies": set(), "variables": set()}
    for artifact in artifacts:
        manifest, rows = registry.load(artifact.artifact_id)
        if manifest.sha256 != artifact.sha256 or manifest.kind != artifact.kind:
            return AnalysisReport(status="incomplete", caveats=["Retrieved metadata provenance does not match the registered source table."])
        if not manifest.complete:
            return AnalysisReport(status="incomplete", caveats=["Metadata came from an incomplete or paged result. Narrow the lookup or retrieve the complete matching records."])
        entity_kind, id_column, name_column = IDENTITY[manifest.kind]
        ids = [r.get(id_column, "") for r in rows]
        if any(not value for value in ids) or len(set(ids)) != len(ids):
            return _clarify("The source contains missing or duplicate entity IDs; one authoritative metadata row cannot be selected.")
        all_ids[manifest.kind].update(ids)
        applicable = [(column, value) for column, value in filters if column in {id_column, name_column}]
        for row in rows:
            identifier = row[id_column]
            if expected[manifest.kind]:
                keep = identifier in expected[manifest.kind]
            else:
                keep = _mentioned(question_words, identifier) or _mentioned(question_words, row.get(name_column, ""))
                if applicable:
                    keep = True  # exact code-checked row predicates below resolve the entity
            if not keep or any(row.get(column) != value for column, value in applicable):
                continue
            key = (manifest.kind, identifier)
            if key in selected:
                previous_manifest, previous_row = selected[key]
                common = set(previous_manifest.columns) | set(manifest.columns)
                if any(previous_row.get(column, "") != row.get(column, "") for column in common):
                    return _clarify(f"Retrieved sources disagree about metadata for {entity_kind} {identifier}; please verify the source record.")
                continue
            selected[key] = (manifest, row)
    for kind, ids in expected.items():
        missing = ids - {identifier for (selected_kind, identifier) in selected if selected_kind == kind}
        if missing:
            return _clarify(f"Requested {kind} IDs were not uniquely established by the retrieved records: {', '.join(sorted(missing))}. This is not evidence that the records do not exist.")
    if not selected:
        return _clarify("No exact requested study or variable was identified in the retrieved records. Please give its exact ID or full recorded name; no absence claim was made.")
    # A name may identify multiple distinct IDs. Never silently choose one for a singular lookup.
    for kind in IDENTITY:
        named = [row for (k, _), (_, row) in selected.items() if k == kind]
        if not expected[kind] and len(named) > 1:
            return _clarify(f"More than one {kind} record matches the supplied name. Please choose an exact database ID.")
    if len(selected) > MAX_METADATA_ENTITIES:
        return _clarify(f"This metadata lookup selected more than {MAX_METADATA_ENTITIES} records. Please narrow the exact study or variable IDs.")
    facts: list[MetadataFact] = []
    omitted_fields: set[str] = set()
    for (kind, identifier), (manifest, row) in sorted(selected.items()):
        entity_kind, _, name_column = IDENTITY[kind]
        fields = STUDY_FIELDS if kind == "studies" else VARIABLE_FIELDS
        for field in fields:
            if field not in manifest.columns:
                omitted_fields.add(field)
                continue
            value = row.get(field) or None
            facts.append(MetadataFact(entity_kind=entity_kind, entity_id=identifier, entity_label=row.get(name_column) or None,
                                      field=field, value=value,
                                      missing_reason="not stated in the retrieved record" if value is None else None,
                                      source_artifact_id=manifest.artifact_id, source_artifact_hash=manifest.sha256,
                                      source_request_ids=list(manifest.source_request_ids),
                                      source_fetched_at_utc=_dates(manifest.source_request_ids, requests or [])))
    validate_metadata_facts(facts, registry)
    caveats = ["Missing fields describe these retrieved records; they do not establish absence from the entire database."]
    if any(kind == "studies" for kind, _ in selected):
        caveats.append("Study start/end dates retain their source labels; planting/harvest dates are shown only from separately recorded fields.")
    if any(kind == "variables" for kind, _ in selected):
        caveats.append("Trait descriptions and scale names are source text. They do not fill an empty structured units field or supply an unrecorded measurement protocol.")
    if omitted_fields:
        caveats.append("These fields are absent from the supplied table schema and could not be checked: " + ", ".join(sorted(omitted_fields)))
    return AnalysisReport(status="completed", metadata_facts=facts,
                          methods=["metadata: selected exact entity rows from hash-checked registered tables; no numeric calculation or model inference"],
                          caveats=caveats)


def _safe_text(value: str) -> str:
    text = " ".join(value.split())
    text = html.escape(text, quote=False)
    return re.sub(r"([\\`*_{}\[\]()#+!|])", r"\\\1", text)


def render_metadata_facts(facts: list[MetadataFact]) -> list[str]:
    """Readable values only; IDs and provenance remain in the saved typed facts."""
    lines: list[str] = []
    previous = None
    for fact in facts:
        key = (fact.entity_kind, fact.entity_id)
        if key != previous:
            if lines:
                lines.append("")
            heading = f"{fact.entity_kind.title()} {fact.entity_id}"
            if fact.entity_label:
                heading += f" — {fact.entity_label}"
            lines.append("**" + _safe_text(heading) + "**")
            if fact.source_fetched_at_utc:
                lines.append("Source fetched: " + ", ".join(d.isoformat() for d in fact.source_fetched_at_utc))
            previous = key
        labels = STUDY_FIELDS if fact.entity_kind == "study" else VARIABLE_FIELDS
        value = fact.value if fact.value is not None else fact.missing_reason or "not stated in the retrieved record"
        lines.append(f"- {labels.get(fact.field, fact.field)}: {_safe_text(value)}")
    return lines
