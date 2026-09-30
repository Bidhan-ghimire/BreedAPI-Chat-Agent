"""
brapi_mcp_server.py — twelve read-only BrAPI tools, callable two ways:

  DIRECT  : dispatch(ctx, "get_study", {"study_db_id": "S1"})   (plain Python, same process)
  MCP     : python -m brapi_mcp_server --offline --fixture synthetic   (stdio server, child process)

Both routes use ONE registry (TOOLS): the same argument models make the JSON
schemas the client sees and validate the arguments that arrive. There is no
second schema definition to drift out of sync.

Rules the tools follow:
* Every tool returns the ToolResult contract: ok + data, or ok=False + a coded error.
* Tool arguments can never carry approval, change the server address or name an
  endpoint. Unknown fields are rejected (extra="forbid").
* Large tables never travel inside a tool message. get_observations,
  get_observation_units and export_metadata save a managed artifact and return
  its handle, a small preview and the true row count.
* Completeness is always stated; an incomplete collection is never presented as
  the whole thing.
* In normal server mode nothing is printed to stdout — stdout is the protocol
  channel. Diagnostics go to stderr.
* `--selftest --offline` exercises all twelve tools on the SYNTHETIC fixtures
  and prints twelve labelled checks without starting MCP.

Everyday example: a library front desk with twelve request forms. Each form has
fixed boxes (the schema); the clerk (dispatch) fills your request from the same
form whether you hand it over in person (direct) or send it by mail (MCP).
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import re
import sys
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from artifacts import ArtifactError, ArtifactRegistry, cell_text
from brapi_client import PART2_DIR, BrapiClient, BrapiClientError, Settings, load_settings
from cache_store import CacheStore
from contracts import (
    CollectionResult,
    FetchApproval,
    IdStr,
    MeasurementMeta,
    RequestRecord,
    StudyFilters,
    ToolError,
    ToolResult,
    dump_json,
)
from explore_cassavabase import SYNTHETIC_BASE, seed_synthetic

__all__ = [
    "SERVER_NAME", "ToolContext", "ToolSpec", "TOOLS", "TOOL_NAMES", "build_context", "dispatch",
    "tool_schemas", "build_server", "serve_stdio", "run_selftest", "main",
]

SERVER_NAME = "cassava"
SERVER_VERSION = "0.2"
log = logging.getLogger("brapi_mcp_server")


# --------------------------------------------------------------------------
# Context: what the tools are allowed to reach
# --------------------------------------------------------------------------

@dataclass
class ToolContext:
    client: BrapiClient
    registry: ArtifactRegistry
    approval: FetchApproval | None = None     # set ONLY by controller code, never from a tool argument
    search_max_pages: int = 5                 # bounded paging for metadata searches
    export_max_pages: int = 20                # bounded paging for full metadata exports
    preview_rows: int = 5
    # Controller-owned metadata, keyed by the exact validated variable ID. Never a tool argument.
    variable_measurements: dict[str, MeasurementMeta] = field(default_factory=dict)


def build_context(
    *,
    offline: bool = True,
    fixture: str | None = None,
    cache_dir: Path | None = None,
    out_dir: Path | None = None,
    run_id: str | None = None,
    approval: FetchApproval | None = None,
    settings: Settings | None = None,
) -> ToolContext:
    """Wire client + registry. fixture='synthetic' uses the invented server and seeds its cache."""
    run_id = run_id or f"mcp_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:6]}"
    if fixture is not None:
        if fixture != "synthetic":
            raise ValueError(f"unknown fixture set {fixture!r}; only 'synthetic' exists")
        cache_dir = Path(cache_dir) if cache_dir else PART2_DIR / "cache" / "synthetic"
        settings = Settings(base_url=SYNTHETIC_BASE, mode="offline", cache_dir=cache_dir)
        cache = CacheStore(cache_dir)
        seed_synthetic(cache, SYNTHETIC_BASE)      # includes the GET /studies/S1 reply (synthetic_study_S1.json)
    else:
        settings = settings or load_settings()
        if offline and settings.mode != "offline":
            settings = dataclasses.replace(settings, mode="offline")
        cache_dir = Path(cache_dir) if cache_dir else settings.cache_dir
        settings = dataclasses.replace(settings, cache_dir=cache_dir)
        cache = CacheStore(cache_dir)
    client = BrapiClient(settings, cache=cache, run_id=run_id)
    registry = ArtifactRegistry(Path(out_dir) if out_dir else PART2_DIR / "out", run_id, base_url=settings.base_url)
    return ToolContext(client=client, registry=registry, approval=approval)


# --------------------------------------------------------------------------
# Argument models = the schemas. strict + extra forbidden.
# --------------------------------------------------------------------------

class ToolArgs(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


Offset = Field(default=0, ge=0, description="how many candidates to skip (paging)")
Limit = Field(default=25, ge=1, le=100, description="how many candidates to return, 1-100")


class NoArgs(ToolArgs):
    """This tool takes no arguments."""


class SearchStudiesArgs(ToolArgs):
    location_id: IdStr | None = Field(default=None, description="exact resolved locationDbId (use list_locations first)")
    season_id: IdStr | None = Field(default=None, description="exact resolved seasonDbId (use list_seasons first)")
    study_type: str | None = Field(default=None, description="exact studyType text as the server reports it")
    program_id: IdStr | None = Field(default=None, description="exact resolved programDbId")
    name_contains: str | None = Field(default=None, description="DISCOVERY only: literal case-insensitive substring of studyName")
    offset: int = Offset
    limit: int = Limit


class GetStudyArgs(ToolArgs):
    study_db_id: IdStr = Field(description="exact studyDbId")


class ListVariablesArgs(ToolArgs):
    name_contains: str = Field(default="", description="literal case-insensitive substring of the variable name; '' = all")
    offset: int = Offset
    limit: int = Limit


class GetObservationsArgs(ToolArgs):
    study_db_id: IdStr = Field(description="exact studyDbId")
    variable_db_id: IdStr = Field(description="exact observationVariableDbId; numeric analysis needs one exact variable")


class ListLocationsArgs(ToolArgs):
    name_contains: str = Field(default="", description="literal case-insensitive substring of locationName; '' = all")
    offset: int = Offset
    limit: int = Limit


class ListSeasonsArgs(ToolArgs):
    season_id: IdStr | None = Field(default=None, description="exact seasonDbId, if already known")
    year: str | None = Field(default=None, description="exact year as text, e.g. '2026'")
    name_contains: str = Field(default="", description="literal case-insensitive substring of seasonName")
    offset: int = Offset
    limit: int = Limit


class ListProgramsArgs(ToolArgs):
    name_contains: str = Field(default="", description="literal case-insensitive substring of programName")
    offset: int = Offset
    limit: int = Limit


class RequestLogArgs(ToolArgs):
    run_id: IdStr = Field(description="the run whose request evidence you want (this process serves one run)")


class GetObservationUnitsArgs(ToolArgs):
    study_db_id: IdStr = Field(description="exact studyDbId")


class ExportFilters(ToolArgs):
    location_id: IdStr | None = None
    season_id: IdStr | None = None
    study_type: str | None = None
    program_id: IdStr | None = None


class ExportMetadataArgs(ToolArgs):
    entity: Literal["studies", "variables", "locations", "programs", "seasons"]
    filters: ExportFilters | None = Field(default=None, description="exact filters; only meaningful for entity='studies'")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

_PAGE_ERROR = re.compile(r"page \d+: ([a-z_]+): ")


def _collection_error(result: CollectionResult) -> ToolError:
    """Turn a failed CollectionResult into a coded ToolError using the client's recorded reason."""
    codes = [m.group(1) for w in result.warnings for m in [_PAGE_ERROR.search(w)] if m]
    code = codes[-1] if codes else "upstream_error"
    mapping = {"offline_cache_miss": "unsupported", "not_authorized": "not_authorized", "timeout": "timeout",
               "invalid_argument": "invalid_argument", "not_found": "not_found"}
    return ToolError(code=mapping.get(code, "upstream_error"), message="; ".join(result.warnings)[:500] or "request failed")


def _map_exception(exc: Exception) -> ToolError:
    if isinstance(exc, BrapiClientError):
        mapping = {"not_authorized": "not_authorized", "offline_cache_miss": "unsupported", "timeout": "timeout",
                   "invalid_argument": "invalid_argument", "not_found": "not_found", "invalid_json": "upstream_error",
                   "not_brapi": "upstream_error", "http_error": "upstream_error", "too_large": "upstream_error"}
        return ToolError(code=mapping.get(exc.code, "upstream_error"), message=exc.message)
    if isinstance(exc, ArtifactError):
        mapping = {"invalid_handle": "invalid_argument", "invalid_argument": "invalid_argument",
                   "hash_mismatch": "internal_error", "escape": "internal_error", "exists": "internal_error"}
        return ToolError(code=mapping.get(exc.code, "internal_error"), message=exc.message)
    if isinstance(exc, ValidationError):
        first = exc.errors()[0]
        return ToolError(code="internal_error", message=f"contract violation at {list(first.get('loc', []))}: {first.get('msg')}")
    log.exception("tool failure")
    return ToolError(code="internal_error", message=f"{type(exc).__name__} (details in the server log, not in this message)")


def _paged(items: list[dict[str, Any]], offset: int, limit: int) -> tuple[list[dict[str, Any]], int | None]:
    page = items[offset:offset + limit]
    next_offset = offset + limit if offset + limit < len(items) else None
    return page, next_offset


def _meta(result: CollectionResult) -> dict[str, Any]:
    return {"status": result.status, "complete": result.complete, "returned": result.returned_count,
            "reported_total": result.reported_total}


def _text(value: Any) -> str | None:
    return cell_text(value)[0]


def _study_row(r: dict[str, Any]) -> dict[str, Any]:
    design = r.get("experimentalDesign") if isinstance(r.get("experimentalDesign"), dict) else {}
    extra = r.get("additionalInfo") if isinstance(r.get("additionalInfo"), dict) else {}
    return {"studyDbId": _text(r.get("studyDbId")), "studyName": _text(r.get("studyName")),
            "studyType": _text(r.get("studyType")), "locationDbId": _text(r.get("locationDbId")),
            "locationName": _text(r.get("locationName")), "seasons": _text(r.get("seasons")),
            "programDbId": _text(r.get("programDbId")), "programName": _text(r.get("programName")),
            "active": _text(r.get("active")), "commonCropName": _text(r.get("commonCropName")),
            "studyDescription": _text(r.get("studyDescription")), "startDate": _text(r.get("startDate")),
            "endDate": _text(r.get("endDate")), "plantingDate": _text(r.get("plantingDate")),
            "harvestDate": _text(r.get("harvestDate")),
            "experimentalDesignDescription": _text(design.get("description")),
            "experimentalDesignPUI": _text(design.get("PUI")),
            "trialDbId": _text(r.get("trialDbId")), "trialName": _text(r.get("trialName")),
            "additionalInfoProgramDbId": _text(extra.get("programDbId")),
            "additionalInfoProgramName": _text(extra.get("programName"))}


def _variable_row(r: dict[str, Any]) -> dict[str, Any]:
    trait = r.get("trait") if isinstance(r.get("trait"), dict) else {}
    method = r.get("method") if isinstance(r.get("method"), dict) else {}
    scale = r.get("scale") if isinstance(r.get("scale"), dict) else {}
    return {"observationVariableDbId": _text(r.get("observationVariableDbId")),
            "observationVariableName": _text(r.get("observationVariableName")),
            "traitName": _text(trait.get("traitName")), "traitDescription": _text(trait.get("traitDescription")),
            "methodName": _text(method.get("methodName")), "methodDescription": _text(method.get("description")),
            "scaleName": _text(scale.get("scaleName")), "units": _text(scale.get("units")),
            "scaleValidValues": _text(scale.get("validValues")),
            "dataType": _text(scale.get("dataType")), "timepoint": _text(r.get("timepoint")),
            "status": _text(r.get("status"))}


def _location_row(r: dict[str, Any]) -> dict[str, Any]:
    return {"locationDbId": _text(r.get("locationDbId")), "locationName": _text(r.get("locationName")),
            "locationType": _text(r.get("locationType")), "countryName": _text(r.get("countryName"))}


def _program_row(r: dict[str, Any]) -> dict[str, Any]:
    return {"programDbId": _text(r.get("programDbId")), "programName": _text(r.get("programName")),
            "commonCropName": _text(r.get("commonCropName"))}


def _season_row(r: dict[str, Any]) -> dict[str, Any]:
    return {"seasonDbId": _text(r.get("seasonDbId")), "seasonName": _text(r.get("seasonName")), "year": _text(r.get("year"))}


def _unit_row(r: dict[str, Any]) -> dict[str, Any]:
    position = r.get("observationUnitPosition") if isinstance(r.get("observationUnitPosition"), dict) else {}
    level = position.get("observationLevel") if isinstance(position.get("observationLevel"), dict) else {}
    return {"observationUnitDbId": _text(r.get("observationUnitDbId")), "observationUnitName": _text(r.get("observationUnitName")),
            "germplasmDbId": _text(r.get("germplasmDbId")), "germplasmName": _text(r.get("germplasmName")),
            "studyDbId": _text(r.get("studyDbId")), "levelName": _text(level.get("levelName")),
            "levelCode": _text(level.get("levelCode")), "observationUnitPosition": _text(position or None)}


OBSERVATION_COLUMNS = {
    "observationDbId": "observation ID (text)", "observationUnitDbId": "plot/unit ID (text)",
    "observationUnitName": "unit label", "germplasmDbId": "clone ID (text)", "germplasmName": "clone label",
    "studyDbId": "study ID (text)", "observationVariableDbId": "variable ID (text)",
    "observationVariableName": "variable label", "observationTimeStamp": "when measured (text)",
    "value": "RAW value token, unconverted ('' = missing)",
}
UNIT_COLUMNS = {
    "observationUnitDbId": "unit ID (text)", "observationUnitName": "unit label", "germplasmDbId": "clone ID",
    "germplasmName": "clone label", "studyDbId": "study ID", "levelName": "unit level (plot, plant, ...)",
    "levelCode": "level code", "observationUnitPosition": "full position object as JSON text",
}
EXPORT_SHAPES: dict[str, tuple[Callable[[dict[str, Any]], dict[str, Any]], dict[str, str], str]] = {
    "studies": (_study_row, {k: "study field (text)" for k in _study_row({})}, "studies"),
    "variables": (_variable_row, {k: "variable field (text)" for k in _variable_row({})}, "variables"),
    "locations": (_location_row, {k: "location field (text)" for k in _location_row({})}, "locations"),
    "programs": (_program_row, {k: "program field (text)" for k in _program_row({})}, "programs"),
    "seasons": (_season_row, {k: "season field (text)" for k in _season_row({})}, "seasons"),
}


def _record_dump(record: RequestRecord) -> dict[str, Any]:
    return json.loads(record.model_dump_json())


# --------------------------------------------------------------------------
# The twelve tools
# --------------------------------------------------------------------------

def tool_server_info(ctx: ToolContext, _: NoArgs) -> ToolResult:
    """Who is the server? Concise identity and advertised calls, with the request ID that proved it.
    Limits: one GET /serverinfo (or its cached copy). Completeness does not apply.
    Do not use it to decide that a feature exists unless it is in advertised_calls."""
    envelope, record = ctx.client.get("/serverinfo", approval=ctx.approval)
    info = envelope.get("result") or {}
    calls = info.get("calls") if isinstance(info, dict) and isinstance(info.get("calls"), list) else []
    data = {
        "base_url": ctx.client.settings.base_url, "mode": ctx.client.settings.mode, "run_id": ctx.client.run_id,
        "server_name": _text(info.get("serverName")), "organization_name": _text(info.get("organizationName")),
        "advertised_calls": sorted({str(c.get("service")) for c in calls if isinstance(c, dict) and c.get("service")}),
        "versions": sorted({str(v) for c in calls if isinstance(c, dict) for v in (c.get("versions") or [])}),
    }
    return ToolResult(ok=True, data=data, request_ids=[record.request_id], complete=None)


def tool_search_studies(ctx: ToolContext, a: SearchStudiesArgs) -> ToolResult:
    """Find CANDIDATE studies. Exact filters (location_id, season_id, study_type, program_id) go to the
    server; name_contains is discovery, applied here. Returns candidates with total_matches, returned,
    complete and next_offset. Limits: at most search_max_pages pages; if complete is False the
    candidate list is not the whole set. Resolve human names to IDs with list_locations/list_seasons
    first; do not treat a name match as an exact ID."""
    filters = StudyFilters(location_id=a.location_id, season_id=a.season_id, study_type=a.study_type,
                           program_id=a.program_id, name_contains=a.name_contains)
    result = ctx.client.studies(filters, approval=ctx.approval, max_pages=ctx.search_max_pages)
    if result.status == "failed":
        return ToolResult(ok=False, error=_collection_error(result), request_ids=result.request_ids, complete=False)
    candidates = [_study_row(r) for r in result.records]
    page, next_offset = _paged(candidates, a.offset, a.limit)
    data = {**_meta(result), "candidates": page, "total_matches": result.returned_count, "returned": len(page),
            "offset": a.offset, "next_offset": next_offset,
            "note": "name_contains is discovery; use exact IDs for observations"}
    return ToolResult(ok=True, data=data, request_ids=result.request_ids, complete=result.complete, warnings=result.warnings)


def tool_study_types(ctx: ToolContext, _: NoArgs) -> ToolResult:
    """Count studies by studyType — only from a COMPLETE study collection. If the collection is not
    complete the tool refuses (incomplete_data) rather than counting a partial list.
    Limits: search_max_pages pages."""
    result = ctx.client.studies(approval=ctx.approval, max_pages=ctx.search_max_pages)
    if result.status == "failed":
        return ToolResult(ok=False, error=_collection_error(result), request_ids=result.request_ids, complete=False)
    if not result.complete:
        msg = f"study collection is {result.status}: {result.returned_count} of {result.reported_total} records; type counts need a complete collection"
        return ToolResult(ok=False, error=ToolError(code="incomplete_data", message=msg), request_ids=result.request_ids,
                          complete=False, warnings=result.warnings)
    counts = Counter(str(r.get("studyType") or "(missing)") for r in result.records)
    data = {**_meta(result), "counts": dict(sorted(counts.items())), "total_studies": result.returned_count}
    return ToolResult(ok=True, data=data, request_ids=result.request_ids, complete=True, warnings=result.warnings)


def tool_get_study(ctx: ToolContext, a: GetStudyArgs) -> ToolResult:
    """One study by exact studyDbId, with provenance (request ID, origin, fetch time, response hash).
    Not for searching: use search_studies to find the ID first."""
    record, receipt = ctx.client.study(a.study_db_id, approval=ctx.approval)
    provenance = {"request_id": receipt.request_id, "origin": receipt.origin,
                  "fetched_at_utc": receipt.fetched_at_utc.isoformat() if receipt.fetched_at_utc else None,
                  "response_sha256": receipt.response_sha256, "base_url": receipt.base_url}
    return ToolResult(ok=True, data={"study": record, "provenance": provenance}, request_ids=[receipt.request_id], complete=None)


def tool_list_variables(ctx: ToolContext, a: ListVariablesArgs) -> ToolResult:
    """Observation variables: ID and name plus trait, method, scale, units and timepoint when the
    server gives them. name_contains is a literal case-insensitive substring (discovery). Returns
    complete and next_offset; when complete is False the list is partial."""
    if a.name_contains.strip():
        result = ctx.client.find_variables(a.name_contains, approval=ctx.approval, max_pages=ctx.search_max_pages)
    else:
        result = ctx.client.variables(approval=ctx.approval, max_pages=ctx.search_max_pages)
    if result.status == "failed":
        return ToolResult(ok=False, error=_collection_error(result), request_ids=result.request_ids, complete=False)
    items = [_variable_row(r) for r in result.records]
    page, next_offset = _paged(items, a.offset, a.limit)
    data = {**_meta(result), "variables": page, "total_matches": result.returned_count, "returned": len(page),
            "offset": a.offset, "next_offset": next_offset}
    return ToolResult(ok=True, data=data, request_ids=result.request_ids, complete=result.complete, warnings=result.warnings)


def _save_and_preview(ctx: ToolContext, result: CollectionResult, rows: list[dict[str, Any]], *, kind: str,
                      columns: dict[str, str], study_ids: list[str], variable_ids: list[str],
                      measurement: MeasurementMeta | None = None) -> tuple[Any, Any]:
    manifest = ctx.registry.save_table(rows, kind=kind, columns=columns, collection=result, study_ids=study_ids,
                                       variable_ids=variable_ids, measurement=measurement,
                                       quality_notes=list(result.warnings))
    preview = ctx.registry.preview(manifest.artifact_id, ctx.preview_rows)
    return manifest, preview


def tool_get_observations(ctx: ToolContext, a: GetObservationsArgs) -> ToolResult:
    """Fetch observations for ONE exact study and ONE exact variable, save them as a managed artifact
    and return its handle, a small preview, the true row count, the completeness status and the
    study-status ledger. The full table is never returned in the message; the Analyst reads the
    artifact by handle. An incomplete fetch stays incomplete (complete=False, status). Live access
    requires a FetchApproval held by the controller; tool arguments cannot supply one."""
    result = ctx.client.observations(a.study_db_id, a.variable_db_id, approval=ctx.approval)
    rows = [{k: _text(r.get(k)) for k in OBSERVATION_COLUMNS} for r in result.records]
    variable_names = {str(r.get("observationVariableName")) for r in result.records if r.get("observationVariableName")}
    known = ctx.variable_measurements.get(a.variable_db_id)
    observed_trait = next(iter(variable_names)) if len(variable_names) == 1 else None
    measurement = (known.model_copy(update={"trait": known.trait or observed_trait}) if known is not None
                   else MeasurementMeta(trait=observed_trait))
    manifest, preview = _save_and_preview(ctx, result, rows, kind="observations", columns=OBSERVATION_COLUMNS,
                                          study_ids=[a.study_db_id], variable_ids=[a.variable_db_id], measurement=measurement)
    reason = None if result.status in ("complete", "empty") else ("; ".join(result.warnings) or result.status)[:300]
    ctx.registry.record_study(a.study_db_id, result.status, artifact_id=manifest.artifact_id, reason=reason,
                              request_ids=result.request_ids)
    ledger = [json.loads(e.model_dump_json()) for e in ctx.registry.ledger()]
    data = {**_meta(result), "artifact_id": manifest.artifact_id, "row_count": manifest.row_count,
            "columns": preview.columns, "preview_rows": preview.rows, "displayed_rows": preview.displayed_rows,
            "variable_names_seen": sorted(variable_names), "ledger": ledger}
    if result.status == "failed":
        return ToolResult(ok=False, error=_collection_error(result), data=data, request_ids=result.request_ids,
                          artifact_ids=[manifest.artifact_id], complete=False, warnings=result.warnings)
    return ToolResult(ok=True, data=data, request_ids=result.request_ids, artifact_ids=[manifest.artifact_id],
                      complete=result.complete, warnings=result.warnings)


def tool_list_locations(ctx: ToolContext, a: ListLocationsArgs) -> ToolResult:
    """Location CANDIDATES with continuation metadata (total_matches, next_offset, complete).
    name_contains is a literal case-insensitive substring. If complete is False, the candidate set is
    not the whole server list; do not claim it is."""
    result = ctx.client.locations(approval=ctx.approval, max_pages=ctx.search_max_pages)
    if result.status == "failed":
        return ToolResult(ok=False, error=_collection_error(result), request_ids=result.request_ids, complete=False)
    needle = a.name_contains.lower()
    items = [_location_row(r) for r in result.records if needle in str(r.get("locationName", "")).lower()]
    page, next_offset = _paged(items, a.offset, a.limit)
    data = {**_meta(result), "candidates": page, "total_matches": len(items), "returned": len(page),
            "offset": a.offset, "next_offset": next_offset}
    return ToolResult(ok=True, data=data, request_ids=result.request_ids, complete=result.complete, warnings=result.warnings)


def _simple_list(result: CollectionResult, key: str, rows: list[dict[str, Any]],
                 *, offset: int, limit: int) -> ToolResult:
    if result.status == "failed":
        return ToolResult(ok=False, error=_collection_error(result), request_ids=result.request_ids, complete=False)
    page, next_offset = _paged(rows, offset, limit)
    data = {**_meta(result), key: page, "total_matches": len(rows), "returned": len(page),
            "offset": offset, "next_offset": next_offset}
    if not result.complete:
        data["next"] = "collection is not complete; use export_metadata for a full, bounded export"
    return ToolResult(ok=True, data=data, request_ids=result.request_ids, complete=result.complete, warnings=result.warnings)


def tool_list_programs(ctx: ToolContext, a: ListProgramsArgs) -> ToolResult:
    """Program candidates, optionally filtered by literal name substring. Follow next_offset with the
    same filters until it is null. complete describes the source collection, not the displayed page.
    Limits: search_max_pages pages; use export_metadata for a full table."""
    result = ctx.client.programs(approval=ctx.approval, max_pages=ctx.search_max_pages)
    rows = [_program_row(r) for r in result.records]
    rows = [r for r in rows if a.name_contains.casefold() in (r["programName"] or "").casefold()]
    return _simple_list(result, "programs", rows, offset=a.offset, limit=a.limit)


def tool_list_seasons(ctx: ToolContext, a: ListSeasonsArgs) -> ToolResult:
    """Season candidates. Look up a known season_id or exact year directly; name_contains is a literal
    case-insensitive substring. Filters combine with AND. Follow next_offset with the same filters
    until null; a missing ID in one page is not absent from the catalog. complete describes the source
    collection, not the displayed page. Limits: search_max_pages pages."""
    result = ctx.client.seasons(approval=ctx.approval, max_pages=ctx.search_max_pages)
    rows = [_season_row(r) for r in result.records]
    rows = [r for r in rows if (a.season_id is None or r["seasonDbId"] == a.season_id)
            and (a.year is None or r["year"] == a.year)
            and a.name_contains.casefold() in (r["seasonName"] or "").casefold()]
    return _simple_list(result, "seasons", rows, offset=a.offset, limit=a.limit)


def tool_request_log(ctx: ToolContext, a: RequestLogArgs) -> ToolResult:
    """Structured, redacted request evidence for ONE run: every attempt and cache use, in order, with
    request IDs, endpoints, parameters, origin, HTTP status, timing, response hash and coded errors.
    This process serves one run; another run_id is not_found. URLs are already redacted."""
    if a.run_id != ctx.client.run_id:
        return ToolResult(ok=False, error=ToolError(code="not_found", message=f"this server process holds only run {ctx.client.run_id!r}"))
    records = [_record_dump(r) for r in ctx.client.provenance()]
    return ToolResult(ok=True, data={"run_id": a.run_id, "count": len(records), "records": records}, complete=True)


def tool_get_observation_units(ctx: ToolContext, a: GetObservationUnitsArgs) -> ToolResult:
    """Observation units (plots, plants, ...) for ONE exact study, saved as a managed artifact. Keeps
    level names/codes and the full position object so independence can be judged later (several rows
    from one plot are not several plots). Returns handle, preview, true row count, level counts and
    completeness. Never the full table in the message."""
    result = ctx.client.observation_units(a.study_db_id, approval=ctx.approval)
    rows = [_unit_row(r) for r in result.records]
    manifest, preview = _save_and_preview(ctx, result, rows, kind="observation_units", columns=UNIT_COLUMNS,
                                          study_ids=[a.study_db_id], variable_ids=[])
    levels = Counter(row["levelName"] or "(unstated)" for row in rows)
    data = {**_meta(result), "artifact_id": manifest.artifact_id, "row_count": manifest.row_count,
            "columns": preview.columns, "preview_rows": preview.rows, "displayed_rows": preview.displayed_rows,
            "levels": dict(sorted(levels.items()))}
    if result.status == "failed":
        return ToolResult(ok=False, error=_collection_error(result), data=data, request_ids=result.request_ids,
                          artifact_ids=[manifest.artifact_id], complete=False, warnings=result.warnings)
    return ToolResult(ok=True, data=data, request_ids=result.request_ids, artifact_ids=[manifest.artifact_id],
                      complete=result.complete, warnings=result.warnings)


def tool_export_metadata(ctx: ToolContext, a: ExportMetadataArgs) -> ToolResult:
    """Export the FULL, COMPLETE metadata collection for one entity (studies, variables, locations,
    programs, seasons) as a managed artifact with provenance and counts, so catalog questions can be
    answered from all rows without sending them through a model. If the collection cannot be proven
    complete within export_max_pages, the tool refuses (incomplete_data) with the counts.
    filters apply to studies only (exact IDs)."""
    shape, columns, kind = EXPORT_SHAPES[a.entity]
    if a.entity == "studies":
        f = a.filters or ExportFilters()
        filters = StudyFilters(location_id=f.location_id, season_id=f.season_id, study_type=f.study_type, program_id=f.program_id)
        result = ctx.client.studies(filters, approval=ctx.approval, max_pages=ctx.export_max_pages)
    else:
        if a.filters is not None and any(v is not None for v in a.filters.model_dump().values()):
            return ToolResult(ok=False, error=ToolError(code="invalid_argument", message="filters apply to entity='studies' only"))
        method = {"variables": ctx.client.variables, "locations": ctx.client.locations,
                  "programs": ctx.client.programs, "seasons": ctx.client.seasons}[a.entity]
        result = method(approval=ctx.approval, max_pages=ctx.export_max_pages)
    if result.status == "failed":
        return ToolResult(ok=False, error=_collection_error(result), request_ids=result.request_ids, complete=False)
    if not result.complete:
        msg = f"{a.entity} collection is {result.status}: {result.returned_count} of {result.reported_total} records within {ctx.export_max_pages} pages"
        return ToolResult(ok=False, error=ToolError(code="incomplete_data", message=msg), request_ids=result.request_ids,
                          complete=False, warnings=result.warnings)
    rows = [shape(r) for r in result.records]
    manifest, preview = _save_and_preview(ctx, result, rows, kind=kind, columns=columns, study_ids=[], variable_ids=[])
    data = {**_meta(result), "entity": a.entity, "artifact_id": manifest.artifact_id, "row_count": manifest.row_count,
            "columns": preview.columns, "preview_rows": preview.rows[:3], "displayed_rows": min(3, preview.displayed_rows),
            "filters": a.filters.model_dump() if a.filters else None}
    return ToolResult(ok=True, data=data, request_ids=result.request_ids, artifact_ids=[manifest.artifact_id],
                      complete=True, warnings=result.warnings)


def materialize_metadata_result(ctx: ToolContext, name: str, args: dict[str, Any], result: ToolResult) -> ToolResult:
    """Persist only returned lookup rows; called by the controller for metadata-only plans.

    No HTTP/model calls. A paged search cannot become a complete catalog merely because
    the upstream collection was complete. Existing artifacts are never duplicated.
    """
    if not result.ok or result.artifact_ids or not isinstance(result.data, dict):
        return result
    rows: list[dict[str, Any]]
    if name == "get_study":
        parsed = GetStudyArgs.model_validate(args)
        record = result.data.get("study")
        if not isinstance(record, dict) or str(record.get("studyDbId", "")) != parsed.study_db_id:
            return ToolResult(ok=False, complete=False, request_ids=result.request_ids,
                              error=ToolError(code="incomplete_data", message="study lookup returned no matching exact study ID"))
        rows = [_study_row(record)]
        kind, complete = "studies", result.complete is not False
    elif name in ("search_studies", "list_variables"):
        parser = SearchStudiesArgs if name == "search_studies" else ListVariablesArgs
        parsed = parser.model_validate(args)
        key, kind = ("candidates", "studies") if name == "search_studies" else ("variables", "variables")
        page = result.data.get(key)
        if not isinstance(page, list) or any(not isinstance(row, dict) for row in page):
            return result
        # These rows were already projected by the tool; reprojecting would lose nested-field columns.
        columns = EXPORT_SHAPES[kind][1]
        rows = [{column: row.get(column) for column in columns} for row in page]
        complete = (result.complete is True and parsed.offset == 0 and result.data.get("next_offset") is None
                    and result.data.get("total_matches") == len(rows) and result.data.get("returned") == len(rows))
    else:
        return result
    id_column = "studyDbId" if kind == "studies" else "observationVariableDbId"
    ids = [str(row[id_column]) for row in rows if row.get(id_column)]
    if len(ids) != len(rows) or len(set(ids)) != len(ids):
        return ToolResult(ok=False, complete=False, request_ids=result.request_ids,
                          error=ToolError(code="incomplete_data", message="lookup rows have missing or duplicate entity IDs"))
    manifest = ctx.registry.save_table(rows, kind=kind, columns=EXPORT_SHAPES[kind][1], complete=complete,
                                       source_request_ids=result.request_ids,
                                       study_ids=ids if kind == "studies" else [],
                                       variable_ids=ids if kind == "variables" else [],
                                       quality_notes=["Lookup evidence includes only rows actually returned by this tool; it is not an unfiltered catalog."])
    return result.model_copy(update={"artifact_ids": [manifest.artifact_id], "complete": complete})


# --------------------------------------------------------------------------
# ONE registry: names, docs, argument models, functions
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolSpec:
    name: str
    args_model: type[ToolArgs]
    fn: Callable[[ToolContext, Any], ToolResult]

    @property
    def description(self) -> str:
        return " ".join((self.fn.__doc__ or "").split())

    def input_schema(self) -> dict[str, Any]:
        return self.args_model.model_json_schema()


TOOLS: dict[str, ToolSpec] = {spec.name: spec for spec in [
    ToolSpec("server_info", NoArgs, tool_server_info),
    ToolSpec("search_studies", SearchStudiesArgs, tool_search_studies),
    ToolSpec("study_types", NoArgs, tool_study_types),
    ToolSpec("get_study", GetStudyArgs, tool_get_study),
    ToolSpec("list_variables", ListVariablesArgs, tool_list_variables),
    ToolSpec("get_observations", GetObservationsArgs, tool_get_observations),
    ToolSpec("list_locations", ListLocationsArgs, tool_list_locations),
    ToolSpec("list_programs", ListProgramsArgs, tool_list_programs),
    ToolSpec("list_seasons", ListSeasonsArgs, tool_list_seasons),
    ToolSpec("request_log", RequestLogArgs, tool_request_log),
    ToolSpec("get_observation_units", GetObservationUnitsArgs, tool_get_observation_units),
    ToolSpec("export_metadata", ExportMetadataArgs, tool_export_metadata),
]}
TOOL_NAMES = list(TOOLS)
assert len(TOOL_NAMES) == 12


def tool_schemas() -> list[dict[str, Any]]:
    """What a client sees: name, description, inputSchema — generated from the same models dispatch uses."""
    return [{"name": s.name, "description": s.description, "inputSchema": s.input_schema()} for s in TOOLS.values()]


def dispatch(ctx: ToolContext, name: str, arguments: dict[str, Any] | None) -> ToolResult:
    """Validate arguments against the tool's model, run it, and ALWAYS return a ToolResult."""
    spec = TOOLS.get(name)
    if spec is None:
        return ToolResult(ok=False, error=ToolError(code="invalid_argument", message=f"unknown tool {name!r}; known: {TOOL_NAMES}"))
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return ToolResult(ok=False, error=ToolError(code="invalid_argument", message="arguments must be a JSON object"))
    try:
        args = spec.args_model.model_validate(arguments)
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(p) for p in first.get("loc", ())) or "(root)"
        return ToolResult(ok=False, error=ToolError(code="invalid_argument", message=f"{loc}: {first.get('msg')}"))
    try:
        return spec.fn(ctx, args)
    except Exception as exc:  # noqa: BLE001 - every failure becomes a coded ToolResult, never a traceback to the caller
        return ToolResult(ok=False, error=_map_exception(exc))


# --------------------------------------------------------------------------
# MCP stdio server (protocol on stdout only)
# --------------------------------------------------------------------------

def build_server(ctx: ToolContext):
    """A low-level MCP Server whose tool list and dispatch both come from TOOLS."""
    import anyio
    from mcp import types
    from mcp.server.lowlevel import Server

    server = Server(SERVER_NAME, version=SERVER_VERSION,
                    instructions="Read-only BrAPI tools. Large tables come back as artifact handles, never inline. "
                                 "Completeness is always stated; do not treat incomplete collections as whole.")

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        return [types.Tool(name=s["name"], description=s["description"], inputSchema=s["inputSchema"]) for s in tool_schemas()]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        result = await anyio.to_thread.run_sync(dispatch, ctx, name, arguments or {})
        payload = json.loads(dump_json(result))
        return types.CallToolResult(content=[types.TextContent(type="text", text=json.dumps(payload))],
                                    structuredContent=payload, isError=not result.ok)

    return server


async def serve_stdio(ctx: ToolContext) -> None:
    from mcp.server.stdio import stdio_server

    server = build_server(ctx)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


# --------------------------------------------------------------------------
# Self-test: all twelve tools on the synthetic fixture, no MCP
# --------------------------------------------------------------------------

def run_selftest(ctx: ToolContext, log_fn: Callable[[str], None] = print) -> int:
    """Call every tool through dispatch and print twelve labelled checks. Returns 0 when all pass."""
    run_id = ctx.client.run_id
    checks: list[tuple[str, dict[str, Any], Callable[[ToolResult], bool], str]] = [
        ("server_info", {}, lambda r: r.ok and r.data["server_name"] == "SYNTHETIC fixture server", "synthetic server identified"),
        ("search_studies", {"name_contains": "s1"}, lambda r: r.ok and r.data["total_matches"] == 1 and r.data["candidates"][0]["studyDbId"] == "S1", "1 candidate S1, complete"),
        ("study_types", {}, lambda r: r.ok and r.data["counts"] == {"Synthetic Yield Trial": 1}, "type counts from a complete collection"),
        ("get_study", {"study_db_id": "S1"}, lambda r: r.ok and r.data["study"]["studyDbId"] == "S1" and r.data["provenance"]["request_id"], "one study with provenance"),
        ("list_variables", {"name_contains": "fresh"}, lambda r: r.ok and r.data["total_matches"] == 1 and r.data["variables"][0]["units"] == "t/ha", "V1 fresh root yield, t/ha"),
        ("get_observations", {"study_db_id": "S1", "variable_db_id": "V1"}, lambda r: r.ok and r.data["row_count"] == 6 and r.complete and r.data["displayed_rows"] == 5 and r.artifact_ids, "6 rows saved as an artifact, 5 previewed, complete"),
        ("list_locations", {}, lambda r: r.ok and r.data["total_matches"] == 1 and r.complete, "1 location candidate, complete"),
        ("list_programs", {}, lambda r: r.ok and r.data["returned"] == 1 and r.complete, "1 program, complete"),
        ("list_seasons", {}, lambda r: r.ok and r.data["returned"] == 1 and r.complete, "1 season, complete"),
        ("request_log", {"run_id": run_id}, lambda r: r.ok and r.data["count"] >= 9 and all(rec["origin"] == "cache" for rec in r.data["records"]), "request evidence, all from cache"),
        ("get_observation_units", {"study_db_id": "S1"}, lambda r: r.ok and r.data["row_count"] == 6 and r.data["levels"] == {"plot": 6}, "6 plots, level=plot"),
        ("export_metadata", {"entity": "variables"}, lambda r: r.ok and r.data["row_count"] == 2 and r.complete, "complete variables export as an artifact"),
    ]
    failures = 0
    for index, (name, args, check, detail) in enumerate(checks, start=1):
        result = dispatch(ctx, name, args)
        try:
            passed = bool(check(result))
        except Exception:  # noqa: BLE001 - a malformed result is a failed check, not a crash
            passed = False
        failures += 0 if passed else 1
        note = detail if passed else (f"{result.error.code}: {result.error.message}" if result.error else "unexpected data")
        log_fn(f"[{'PASS' if passed else 'FAIL'}] {index:>2}/12 {name:<22} {note}")
    log_fn(f"{'ALL 12 CHECKS PASSED' if failures == 0 else f'{failures} CHECK(S) FAILED'}  run_id={run_id}  live_requests="
           f"{sum(1 for r in ctx.client.provenance() if r.origin == 'live')}")
    return 0 if failures == 0 else 1


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m brapi_mcp_server", description=__doc__.split("\n\n")[0])
    p.add_argument("--offline", action="store_true", help="offline mode (the only mode this server runs in this version)")
    p.add_argument("--fixture", default=None, help="'synthetic' uses the invented server and seeds its cache")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--selftest", action="store_true", help="run all twelve tools and exit; does not start MCP")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)   # diagnostics never go to stdout
    try:
        ctx = build_context(offline=True, fixture=args.fixture, cache_dir=Path(args.cache_dir) if args.cache_dir else None,
                            out_dir=Path(args.out_dir) if args.out_dir else None, run_id=args.run_id)
    except (ValueError, BrapiClientError, ArtifactError) as exc:
        print(f"FAILED to build context: {exc}", file=sys.stderr)
        return 2
    if args.selftest:
        return run_selftest(ctx)
    if ctx.client.settings.mode != "offline":  # pragma: no cover - build_context forces offline
        print("refusing to serve: this version serves offline only", file=sys.stderr)
        return 2
    print(f"{SERVER_NAME} MCP server: offline, base={ctx.client.settings.base_url}, run_id={ctx.client.run_id}", file=sys.stderr)
    asyncio.run(serve_stdio(ctx))
    return 0


if __name__ == "__main__":
    sys.exit(main())
