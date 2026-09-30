"""Execute a bounded, approved catalog count plan with fixed tools and no Analyst model."""
from __future__ import annotations

import json
import re
from time import monotonic
from typing import Any, Callable, Generator

import analyst_tools as at
from artifacts import ArtifactError
from agents.coordinator import person_words
from agents.retriever import catalog_export_key
from contracts import AnalysisReport, Claim, Plan, RetrievalReport, ToolResult
from llm import Budget, ToolInterface

ENTITY_IDS = {"studies": "studyDbId", "variables": "observationVariableDbId", "locations": "locationDbId",
              "programs": "programDbId", "seasons": "seasonDbId"}
SCOPE_FIELDS = {"location_id": ("locationDbId", "eq"), "location_name": ("locationName", "eq"),
                "study_type": ("studyType", "eq"), "program_id": ("programDbId", "eq"),
                "name_contains": ("studyName", "contains")}


def catalog_count_plan(plan: Plan | None) -> bool:
    """Structural gate only. Invalid arguments/lineage within this shape fail closed below."""
    if plan is None or plan.statistic is not None and plan.statistic.kind != "count":
        return False
    steps = plan.steps
    return (len(steps) >= 2 and steps[0].agent == "retriever" and steps[0].action == "export_metadata"
            and steps[0].inputs.get("entity") in ENTITY_IDS
            and steps[-1].agent == "analyst" and steps[-1].action == "count_records"
            and all(s.agent == "analyst" and s.action == "filter_rows" for s in steps[1:-1]))


def capture_catalog_sources(plan: Plan | None, records: list[Any]) -> dict[str, list[str]]:
    """Capture exact export-argument bindings from real successful ToolRecords, before discarding them."""
    if not catalog_count_plan(plan):
        return {}
    step = plan.steps[0]
    try:
        expected = catalog_export_key(step.inputs)
    except ValueError:
        return {}
    handles = []
    for record in records:
        if record.name != "export_metadata" or getattr(record, "blocked_here", False) or getattr(record, "refused_here", False):
            continue
        if not record.result.ok or record.result.complete is not True:
            continue
        try:
            matches = catalog_export_key(record.args) == expected
        except ValueError:
            continue
        if matches:
            handles.extend(h for h in record.result.artifact_ids if h not in handles)
    return {step.step_id: handles}


def _ids(rows: list[dict[str, str]], column: str) -> set[str]:
    values = [row.get(column, "") for row in rows]
    if any(not value.strip() for value in values) or len(set(values)) != len(values):
        raise ValueError(f"{column} must identify each source row uniquely; missing or duplicate IDs require review")
    return set(values)


def _exact_season_ids(rows: list[dict[str, str]], value: str) -> set[str]:
    selected = set()
    for row in rows:
        try:
            seasons = json.loads(row.get("seasons", ""))
        except (ValueError, TypeError) as exc:
            raise ValueError("seasons is missing or malformed JSON; exact season membership is unknown") from exc
        if not isinstance(seasons, list) or not seasons or any(not isinstance(s, str) or not s for s in seasons):
            raise ValueError("seasons must be a nonempty JSON list of recorded string IDs; membership is unknown otherwise")
        if value in seasons:
            selected.add(row["studyDbId"])
    return selected


def _season_filter(plan: Plan, inputs: dict[str, Any], manifest: Any, rows: list[dict[str, str]]) -> tuple[dict[str, str], set[str]]:
    value = inputs.get("value")
    if (manifest.kind != "studies" or inputs.get("column") != "seasonDbId" or inputs.get("op") != "eq"
            or "seasonDbId" in manifest.columns or "seasons" not in manifest.columns or not isinstance(value, str) or not value):
        raise ValueError("the requested filter column is not in the catalog; no arbitrary column alias is permitted")
    if plan.scope.filters.season_id != value or plan.scope.season_ids != [value]:
        raise ValueError("season filter does not agree with the approved season scope")
    question = person_words(plan.question)
    if not re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", question):
        raise ValueError("the season value is not stated in the person's question")
    expected = _exact_season_ids(rows, value)
    if expected != set(plan.scope.study_ids) or plan.scope.matching_study_count != len(expected):
        raise ValueError("exact JSON season membership differs from the code-approved study IDs or count")
    # The fixed text-filter tool retains lineage. Its output is separately checked against
    # exact decoded JSON membership, so text matching alone can never authorize a count.
    return {"column": "seasons", "op": "contains", "value": json.dumps(value, ensure_ascii=True)}, expected


def _catalog_plan_steps(plan: Plan, retrieval: RetrievalReport, registry: Any, *, export_sources: dict[str, list[str]],
                        budget: Budget | None = None, clock: Callable[[], float] = monotonic
                        ) -> Generator[tuple[str, dict[str, Any]], ToolResult, AnalysisReport | None]:
    """One proof/lineage/budget algorithm, independent of direct or MCP tool transport.

    None means unsupported shape. Supported but unproven plans always refuse;
    neither driver may fall back to a model or another tool transport.
    """
    if not catalog_count_plan(plan):
        return None
    methods: list[str] = []
    derived: list[str] = []
    caveats: list[str] = []

    def stop(message: str, status: str = "needs_clarification") -> AnalysisReport:
        return AnalysisReport(status=status, methods=methods, result_artifact_ids=derived,
                              caveats=[*caveats, "analyst asks: " + message])

    def call(name: str, args: dict[str, Any]):
        if budget is not None:
            budget.start(clock)
            if budget.tool_calls >= budget.max_tool_calls or budget.elapsed(clock) >= budget.max_elapsed_seconds:
                raise TimeoutError("the shared tool or time budget is exhausted")
            budget.tool_calls += 1
        result = yield name, args
        methods.append(f"{name}({json.dumps(args, sort_keys=True)}) -> {'ok' if result.ok else 'refused'}")
        if not result.ok:
            raise ValueError(result.error.message if result.error else "fixed tool failed")
        if result.complete is not True:
            raise ValueError("a planned catalog operation returned incomplete evidence")
        caveats.extend(w for w in result.warnings if w not in caveats)
        return result

    try:
        if not retrieval.complete or retrieval.status != "completed":
            return stop("The required catalog retrieval is incomplete.", "incomplete")
        export = plan.steps[0]
        export_args = json.loads(catalog_export_key(export.inputs))  # Validate approved export arguments too.
        for previous, step in zip(plan.steps, plan.steps[1:]):
            if step.depends_on != [previous.step_id]:
                return stop("Catalog operations need one unambiguous export/filter/count lineage.")
        if export.depends_on:
            return stop("The catalog export has an unsupported dependency.")
        supplied = {a.artifact_id: a for a in retrieval.artifacts}
        candidates = []
        for handle in export_sources.get(export.step_id, []):
            evidence = supplied.get(handle)
            if evidence is None:
                return stop("A planned export handle was not supplied by this retrieval.")
            manifest, rows = registry.load(handle)
            if (not manifest.complete or manifest.sha256 != evidence.sha256 or manifest.kind != export.inputs["entity"]
                    or manifest.kind != evidence.kind):
                return stop("The approved export does not match a complete registered catalog and hash.", "incomplete")
            candidates.append((manifest, rows))
        if not candidates:
            return stop("No complete artifact is bound to the exact approved export arguments.", "incomplete")
        if len({(m.sha256, m.kind, tuple(m.columns)) for m, _ in candidates}) != 1:
            return stop("Different complete tables were returned for the same export; the source is ambiguous.")
        manifest, rows = sorted(candidates, key=lambda item: item[0].artifact_id)[0]
        entity_kind = manifest.kind
        entity_column = ENTITY_IDS[entity_kind]
        _ids(rows, entity_column)
        handle = manifest.artifact_id
        # Every equivalent source supplied to the MCP session must be inspected.
        # Duplicate exports therefore consume an inspection call each, even though
        # only one canonical table is selected for the subsequent count.
        for candidate, _ in sorted(candidates, key=lambda item: item[0].artifact_id):
            yield from call("table_info", {"artifact_id": candidate.artifact_id})
        used_scope_filters = set()
        applied_scope_values = {}
        for name, value in export_args.get("filters", {}).items():
            if entity_kind != "studies":
                return stop("Filters on this exported entity are unsupported.")
            declared = getattr(plan.scope.filters, name)
            if declared is not None and declared != value:
                return stop("The exported filter differs from the approved scope filter.")
            if name == "season_id":
                if _exact_season_ids(rows, value) != _ids(rows, entity_column):
                    return stop("The filtered export contains rows outside its exact JSON season constraint.")
            else:
                column, _ = SCOPE_FIELDS[name]
                if column not in manifest.columns or any(not row.get(column) or row[column] != value for row in rows):
                    return stop("The filtered export contains missing or contradictory filter metadata.")
            used_scope_filters.add(name)
            applied_scope_values[name] = value
        for step in plan.steps[1:-1]:
            args = dict(step.inputs)
            if set(args) - {"column", "op", "value"} or set(args) < {"column", "op", "value"}:
                return stop("A catalog filter needs only explicit column, op, and value arguments.")
            if args["op"] not in {"eq", "ne", "contains"} or not isinstance(args["value"], str) or not args["value"]:
                return stop("Only explicit nonempty text filters are supported by this bounded catalog path.")
            exact_ids = None
            original_args = dict(args)
            if args["column"] not in manifest.columns:
                args, exact_ids = _season_filter(plan, args, manifest, rows)
                used_scope_filters.add("season_id")
                applied_scope_values["season_id"] = original_args["value"]
                methods.append("validated seasonDbId alias using exact JSON seasons membership and the approved study IDs/count")
            for name, (column, operator) in SCOPE_FIELDS.items():
                scoped = getattr(plan.scope.filters, name)
                if (original_args["column"], original_args["op"]) == (column, operator):
                    applied_scope_values[name] = original_args["value"]
                    if scoped is None or original_args["value"] == scoped:
                        used_scope_filters.add(name)
            result = yield from call("filter_rows", {"artifact_id": handle, **args})
            if result.data.get("rows_with_unknown_filter_metadata", 0):
                return stop("Some catalog rows lack the requested filter metadata; a complete matching count is unknown.", "incomplete")
            handle = result.data["artifact_id"]
            derived.append(handle)
            manifest, rows = registry.load(handle)
            selected_ids = _ids(rows, entity_column)
            if exact_ids is not None and selected_ids != exact_ids:
                return stop("The fixed filter result differs from exact JSON season membership; no count was produced.")
        scope_filters = {k for k, v in plan.scope.filters.model_dump().items() if k != "contract_version" and v is not None}
        if scope_filters - used_scope_filters:
            return stop("The bounded plan did not apply every approved scope filter: " + ", ".join(sorted(scope_filters - used_scope_filters)))
        ids = _ids(rows, entity_column)
        scope_lists = {"studies": plan.scope.study_ids, "variables": plan.scope.variable_ids,
                       "locations": plan.scope.location_ids, "seasons": plan.scope.season_ids}
        for kind, requested_ids in scope_lists.items():
            if not requested_ids or kind == entity_kind:
                continue
            if entity_kind == "studies" and kind in {"locations", "seasons"}:
                field = "location_id" if kind == "locations" else "season_id"
                if len(requested_ids) == 1 and applied_scope_values.get(field) == requested_ids[0]:
                    continue
            return stop("An approved scope from another entity was not represented by the catalog operations.")
        scoped_ids = scope_lists.get(entity_kind, [])
        if scoped_ids and ids != set(scoped_ids):
            return stop("The filtered catalog rows differ from the exact approved entity IDs.")
        if entity_kind == "studies" and plan.scope.matching_study_count is not None:
            if len(ids) != plan.scope.matching_study_count:
                return stop("The filtered study count differs from the code-approved scope count.")
        count = dict(plan.steps[-1].inputs)
        if set(count) - {"group_by", "distinct_by"}:
            return stop("The count step contains unsupported arguments; artifact handles come only from the validated lineage.")
        for key, value in count.items():
            if not isinstance(value, str) or value not in manifest.columns:
                return stop(f"The {key} column is absent from the selected complete catalog.")
        result = yield from call("count_records", {"artifact_id": handle, **count})
        claims = [Claim.model_validate(c) for c in result.data.get("claims", [])]
        if not claims or any(c.source_artifact_hashes != [manifest.sha256] for c in claims):
            return stop("The fixed count did not provide claims linked to the selected table hash.", "incomplete")
        if budget is not None and budget.elapsed(clock) >= budget.max_elapsed_seconds:
            return stop("The shared time budget was exhausted.", "limit_reached")
        return AnalysisReport(status="completed", claims=claims, methods=methods, result_artifact_ids=derived,
                              caveats=[*caveats, "Counts describe the retrieved catalog; its records may change after retrieval."])
    except TimeoutError as exc:
        return stop(str(exc), "limit_reached")
    except ArtifactError as exc:
        return stop("The registered source could not be verified: " + str(exc), "incomplete")
    except (ValueError, KeyError, TypeError) as exc:
        return stop(str(exc))


def run_catalog_plan(plan: Plan, retrieval: RetrievalReport, registry: Any, *, export_sources: dict[str, list[str]],
                     budget: Budget | None = None, clock: Callable[[], float] = monotonic) -> AnalysisReport | None:
    """Execute the shared bounded catalog plan using the existing direct tools."""
    steps = _catalog_plan_steps(plan, retrieval, registry, export_sources=export_sources, budget=budget, clock=clock)
    try:
        name, args = next(steps)
        while True:
            try:
                result = at.dispatch_analyst(registry, name, args)
            except Exception as exc:
                name, args = steps.throw(exc)
            else:
                name, args = steps.send(result)
    except StopIteration as done:
        return done.value
    finally:
        steps.close()


async def run_catalog_plan_async(plan: Plan, retrieval: RetrievalReport, registry: Any, *, tools: ToolInterface,
                                 export_sources: dict[str, list[str]], budget: Budget | None = None,
                                 clock: Callable[[], float] = monotonic) -> AnalysisReport | None:
    """Await the same fixed tools through the supplied interface, without direct fallback."""
    steps = _catalog_plan_steps(plan, retrieval, registry, export_sources=export_sources, budget=budget, clock=clock)
    try:
        name, args = next(steps)
        while True:
            try:
                result = await tools.call(name, args)
            except Exception as exc:
                name, args = steps.throw(exc)
            else:
                name, args = steps.send(result)
    except StopIteration as done:
        return done.value
    finally:
        steps.close()
