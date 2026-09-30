"""
analyst_tools.py — eight FIXED calculation tools for the Analyst.

Plain-words summary:
* The Analyst never writes Python. It picks one of these eight functions and passes a
  registered artifact handle (art_xxxxxx_0001). The function reads the table through
  artifacts.py (hash-checked, path-safe), computes with plain arithmetic, and hands back
  typed claims plus the counts behind them. No HTTP, no eval/exec, no subprocess, no model.
* Every cell is TEXT until this module classifies it. A value token is one of three things:
    valid    — a plain decimal number ('10', '14.5', '-3', '2e3') that is finite
    missing  — an empty cell or an agreed missing token ('', 'NA', 'N/A', 'null', 'None', '.')
    invalid  — anything else, including 'bad', 'inf', '-inf', 'nan', '1e400' (overflows)
  Missing and invalid are counted, never turned into zero, and the original tokens survive.
* n needs a name. n_valid_values (how many numbers were averaged) is reported separately from
  n_independent_units (how many distinct plots/plants they came from). When the table cannot
  establish independence, the tool says unknown rather than calling rows "plots".
* Repeated measurements of one unit (same unit, variable, timepoint) or repeated observation
  IDs are detected. The default is to refuse; the caller must name an aggregation rule
  (duplicate_policy='mean_per_unit') and the result says that it was applied.
* Full precision is kept for sorting and for the `value` fields; `display` strings are rounded.
  Ties are reported, and ordering inside a tie is by group ID, so a rerun gives the same rank.
* An incomplete artifact (the source collection was not proven complete) yields no study-level
  claim: numeric tools refuse it unless allow_incomplete=True, and then every claim carries
  the caveat in its transformation text.
* Full results (every group, including excluded ones) are saved as analysis_result artifacts
  with their lineage in the manifest notes; a message only ever shows a preview.

Everyday example: a pocket calculator with eight labelled buttons, glued to a filing cabinet.
You may press a button and name a drawer; you may not rewire the calculator, open other
cabinets, or dictate the answer.
"""
from __future__ import annotations

import math
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from artifacts import ArtifactError, ArtifactRegistry
from contracts import ArtifactId, ArtifactManifest, Claim, Denominator, Exclusion, ToolError, ToolResult

__all__ = [
    "MISSING_TOKENS", "PLOTTING_ENABLED", "AnalystError", "classify_value", "ANALYST_TOOLS", "ANALYST_TOOL_NAMES",
    "analyst_tool_schemas", "dispatch_analyst", "AnalystTools",
]

# -- the value policy: explicit, visible in every numeric result -------------------------------
MISSING_TOKENS = frozenset({"", "NA", "N/A", "NULL", "null", "None", "."})
_NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")     # plain decimal numbers only; no '1_000', no '0x10'
VALUE_POLICY = {
    "missing_tokens": sorted(MISSING_TOKENS),
    "valid": "a plain decimal number (optional sign, digits, one decimal point, optional exponent) that is finite",
    "invalid": "every other token, including inf, -inf, nan and numbers that overflow to infinity",
    "ids": "IDs are exact text: '007' and '7' are different; only an empty cell is a missing ID",
}
PLOTTING_ENABLED = False          # matplotlib is not an approved dependency; plot_bar reports unavailable
DISPLAY_DECIMALS = 4

# column conventions of the observation tables written by brapi_mcp_server
VALUE_COL, UNIT_COL, OBS_ID_COL, VARIABLE_COL, TIMEPOINT_COL = (
    "value", "observationUnitDbId", "observationDbId", "observationVariableDbId", "observationTimeStamp")
LABEL_FOR = {"germplasmDbId": "germplasmName", "observationUnitDbId": "observationUnitName", "studyDbId": "studyName",
             "locationDbId": "locationName", "programDbId": "programName", "seasonDbId": "seasonName",
             "observationVariableDbId": "observationVariableName"}
MISSING_GROUP = "(missing group id)"
_HANDLE = re.compile(r"^art_[0-9a-f]{6}_[0-9]{4,}$")               # same shape artifacts.py issues
ANALYSIS_TARGET = ("descriptive means of valid values as recorded; not adjusted for trial design, environment, "
                   "missingness or clone composition; not evidence of genetic superiority")


class AnalystError(Exception):
    """A refusal with a fixed code (invalid_argument, not_found, incomplete_data, unsupported)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


# --------------------------------------------------------------------------
# Reading and classifying
# --------------------------------------------------------------------------

def classify_value(token: str | None) -> tuple[Literal["valid", "missing", "invalid"], float | None]:
    """One text token -> (class, number-or-None). Never guesses; the original token is left to the caller."""
    if token is None:
        return "missing", None
    text = token.strip()
    if text in MISSING_TOKENS:
        return "missing", None
    if not _NUMBER.match(text):
        return "invalid", None
    number = float(text)
    if not math.isfinite(number):
        return "invalid", None
    return "valid", number


@dataclass
class Table:
    manifest: ArtifactManifest
    rows: list[dict[str, str]]

    @property
    def columns(self) -> list[str]:
        return list(self.manifest.columns)


def _load(registry: ArtifactRegistry, artifact_id: Any) -> Table:
    if not isinstance(artifact_id, str) or not _HANDLE.match(artifact_id):
        raise AnalystError("invalid_argument", f"{artifact_id!r} is not an artifact handle (expected art_xxxxxx_0001); paths are not accepted")
    if artifact_id not in registry.artifact_ids():
        raise AnalystError("not_found", f"{artifact_id} is not registered in this run")
    manifest, rows = registry.load(artifact_id)
    return Table(manifest, rows)


def _require_column(table: Table, column: str, what: str = "column") -> None:
    if column not in table.manifest.columns:
        raise AnalystError("invalid_argument", f"{what} {column!r} is not a column of {table.manifest.artifact_id}; columns: {table.columns}")


def _single_variable(table: Table) -> str:
    """The one variable this table measures, or a refusal. Mixed variables are never averaged."""
    ids = set(table.manifest.variable_ids)
    if VARIABLE_COL in table.manifest.columns:
        ids |= {r.get(VARIABLE_COL, "") for r in table.rows if r.get(VARIABLE_COL, "")}
    if len(ids) > 1:
        raise AnalystError("unsupported", f"table holds more than one variable {sorted(ids)}; numeric tools work on exactly one")
    if not ids:
        raise AnalystError("unsupported", "cannot establish which variable the table measures (no variable_ids and no observationVariableDbId column)")
    return next(iter(ids))


def _display(number: float | None) -> str | None:
    if number is None:
        return None
    text = f"{number:.{DISPLAY_DECIMALS}f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None, "sd": None}
    return {"mean": statistics.fmean(values), "median": statistics.median(values), "min": min(values), "max": max(values),
            "sd": statistics.stdev(values) if len(values) >= 2 else None}


def _claim_component(part: str) -> str:
    # Plain components have no separator and may not occupy the encoding namespace.
    # UTF-8 hex is injective; punctuation, Unicode, underscores and empty text stay distinct.
    if re.fullmatch(r"[A-Za-z0-9-]+", part) and not part.startswith("x0"):
        return part
    return "x0" + part.encode("utf-8").hex()


def _claim_id(artifact_id: str, kind: str, *parts: str, namespace: str | None = None) -> str:
    tail = "_".join(_claim_component(p) for p in parts)
    # v2 cannot be a registry's six-hex prefix, so this namespace cannot overlap legacy IDs.
    head = (f"clm_v2_{artifact_id[4:]}_{_claim_component(namespace)}_{kind}" if namespace is not None
            else f"clm_{artifact_id[4:]}_{kind}")
    return head + (f"_{tail}" if tail else "")


def _duplicate_key(row: Mapping[str, str], columns: list[str]) -> tuple[str, ...] | None:
    unit = row.get(UNIT_COL, "")
    if not unit:
        return None
    return (unit, row.get(VARIABLE_COL, "") if VARIABLE_COL in columns else "", row.get(TIMEPOINT_COL, "") if TIMEPOINT_COL in columns else "")


@dataclass
class Classified:
    """Rows of one table sorted into valid / missing / invalid, with independence and duplicate facts."""

    valid: list[tuple[dict[str, str], float]]
    n_missing: int
    n_invalid: int
    n_rows: int
    units_known: bool                    # every valid row names an observation unit
    repeated_observation_ids: int        # extra copies of an observationDbId
    repeated_measurements: int           # extra rows for a (unit, variable, timepoint)
    duplicate_note: str | None

    @property
    def n_valid(self) -> int:
        return len(self.valid)

    @property
    def n_independent_units(self) -> int | None:
        if not self.units_known:
            return None
        return len({r[UNIT_COL] for r, _ in self.valid})


def _classify(table: Table, value_col: str) -> Classified:
    valid: list[tuple[dict[str, str], float]] = []
    n_missing = n_invalid = 0
    for row in table.rows:
        kind, number = classify_value(row.get(value_col))
        if kind == "valid":
            valid.append((row, number))      # type: ignore[arg-type]
        elif kind == "missing":
            n_missing += 1
        else:
            n_invalid += 1
    has_units = UNIT_COL in table.manifest.columns
    units_known = has_units and all(r.get(UNIT_COL, "") for r, _ in valid)
    rep_obs = 0
    if OBS_ID_COL in table.manifest.columns:
        counts = Counter(r.get(OBS_ID_COL, "") for r in table.rows if r.get(OBS_ID_COL, ""))
        rep_obs = sum(c - 1 for c in counts.values() if c > 1)
    rep_meas = 0
    if has_units:
        keys = Counter(k for k in (_duplicate_key(r, table.columns) for r in table.rows) if k is not None)
        rep_meas = sum(c - 1 for c in keys.values() if c > 1)
    note = None
    if rep_obs or rep_meas:
        note = (f"{rep_obs} repeated observation ID(s) and {rep_meas} repeated measurement(s) per unit/variable/timepoint; "
                "duplicate_policy='mean_per_unit' averages each unit first, duplicate_policy='reject' refuses")
    return Classified(valid, n_missing, n_invalid, len(table.rows), units_known, rep_obs, rep_meas, note)


def _apply_duplicate_policy(table: Table, classified: Classified, policy: str) -> tuple[list[tuple[dict[str, str], float]], str]:
    """Return (rows-with-values to analyse, transformation text). Refuses duplicates unless a rule is named."""
    if not (classified.repeated_observation_ids or classified.repeated_measurements):
        return classified.valid, "valid values used as recorded (one row per observation unit)"
    if policy == "reject":
        raise AnalystError("unsupported", f"{table.manifest.artifact_id}: {classified.duplicate_note}")
    if not classified.units_known:
        raise AnalystError("unsupported", "mean_per_unit needs an observation unit ID on every valid row; independence is unknown here")
    groups: dict[tuple[str, ...], list[tuple[dict[str, str], float]]] = {}
    for row, number in classified.valid:
        groups.setdefault(_duplicate_key(row, table.columns), []).append((row, number))   # type: ignore[arg-type]
    merged = [(rows[0][0], statistics.fmean(n for _, n in rows)) for rows in groups.values()]
    return merged, (f"duplicate_policy=mean_per_unit: {len(classified.valid)} valid rows averaged within "
                    f"{len(merged)} observation units before pooling")


def _completeness_gate(table: Table, allow_incomplete: bool) -> str | None:
    if table.manifest.complete:
        return None
    if not allow_incomplete:
        raise AnalystError("incomplete_data", f"{table.manifest.artifact_id} is not a complete collection; no study-level figure can be "
                                              "claimed from it (pass allow_incomplete=True for a clearly labelled partial description)")
    return "INCOMPLETE ARTIFACT: describes the rows fetched, not the study"


def _exclusions(classified: Classified) -> list[Exclusion]:
    out = []
    if classified.n_missing:
        out.append(Exclusion(reason="raw-missing value token", count=classified.n_missing))
    if classified.n_invalid:
        out.append(Exclusion(reason="invalid or non-finite value token", count=classified.n_invalid))
    return out


def _numeric_claims(table: Table, classified: Classified, used: list[tuple[dict[str, str], float]], transformation: str,
                    *, grouping: list[str], filters: dict[str, str], unit: str | None, denominator_name: str,
                    label_parts: tuple[str, ...] = (), claim_namespace: str | None = None) -> tuple[dict[str, float | None], list[Claim]]:
    values = [n for _, n in used]
    stats = _stats(values)
    n_units = len({r[UNIT_COL] for r, _ in used}) if classified.units_known else None
    denominator = Denominator(name=denominator_name, n=len(values))
    exclusions = _exclusions(classified)
    hashes = [table.manifest.sha256]
    claims: list[Claim] = []
    for kind in ("mean", "median", "min", "max", "sd"):
        value = stats[kind]
        reason = None
        if value is None:
            reason = "fewer than 2 valid values: sample SD undefined" if kind == "sd" and values else "no valid numeric values"
        claims.append(Claim(claim_id=_claim_id(table.manifest.artifact_id, kind, *label_parts, namespace=claim_namespace), kind=kind, value=value, unit=unit,
                            missing_reason=reason, denominator=denominator, n_independent_units=n_units,
                            source_artifact_hashes=hashes, filters=filters, grouping=grouping,
                            transformation=f"{kind} of valid values; {transformation}", exclusions=exclusions))
    claims.append(Claim(claim_id=_claim_id(table.manifest.artifact_id, "count", *label_parts, namespace=claim_namespace), kind="count", value=len(values),
                        denominator=denominator, n_independent_units=n_units, source_artifact_hashes=hashes, filters=filters,
                        grouping=grouping, transformation=f"count of valid values; {transformation}", exclusions=exclusions))
    return stats, claims


def _claims_json(claims: list[Claim]) -> list[dict[str, Any]]:
    return [c.model_dump(mode="json") for c in claims]


# --------------------------------------------------------------------------
# Argument models = the schemas (strict, extra keys refused)
# --------------------------------------------------------------------------

class ToolArgs(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


DuplicatePolicy = Literal["reject", "mean_per_unit"]
def _handle() -> Any:
    return Field(description="a registered artifact handle such as art_3f9a2c_0001 (never a path)")


class TableInfoArgs(ToolArgs):
    artifact_id: ArtifactId = _handle()


class NumericSummaryArgs(ToolArgs):
    artifact_id: ArtifactId = _handle()
    value: str = Field(default=VALUE_COL, description="the value column")
    duplicate_policy: DuplicatePolicy = Field(default="reject", description="what to do with repeated measurements of one unit")
    allow_incomplete: bool = Field(default=False, description="describe an incomplete artifact anyway, clearly labelled")


class GroupStatsArgs(ToolArgs):
    artifact_id: ArtifactId = _handle()
    by: str = Field(description="a stable ID column to group by (e.g. germplasmDbId); names are carried as labels")
    value: str = Field(default=VALUE_COL)
    top_n: int = Field(default=10, ge=1, le=1000,
                       description="how many included groups to show AND state as claims (best first); the artifact holds all. A question "
                                   "about every group (plots per clone) needs top_n at least the number of groups")
    ascending: bool = Field(default=False)
    min_independent_n: int = Field(default=2, ge=1, description="exclusion threshold on independent units; not a design correction")
    duplicate_policy: DuplicatePolicy = Field(default="reject")
    allow_incomplete: bool = Field(default=False)


FilterOp = Literal["eq", "ne", "contains", "gt", "ge", "lt", "le", "is_missing", "is_valid_number"]


class FilterRowsArgs(ToolArgs):
    artifact_id: ArtifactId = _handle()
    column: str
    op: FilterOp = Field(description="eq/ne/contains compare text (contains = literal, case-insensitive); gt/ge/lt/le compare valid numbers only")
    value: str = Field(default="", description="the comparison text; ignored by is_missing / is_valid_number")


class ConcatTablesArgs(ToolArgs):
    artifact_ids: list[ArtifactId] = Field(min_length=2, max_length=50, description="two or more compatible artifacts")


class MissingReportArgs(ToolArgs):
    artifact_id: ArtifactId = _handle()
    by: str | None = Field(default=None, description="optional ID column for per-group counts")
    value: str = Field(default=VALUE_COL)
    roster_artifact_id: ArtifactId | None = Field(default=None, description="an observation_units artifact for the same study: enables the expected-unit denominator")


class PlotBarArgs(ToolArgs):
    artifact_id: ArtifactId = _handle()
    x: str
    y: str
    top_n: int = Field(default=10, ge=1, le=100)


class CountRecordsArgs(ToolArgs):
    artifact_id: ArtifactId = _handle()
    group_by: str | None = Field(default=None, description="column whose value frequencies to count")
    distinct_by: str | None = Field(default=None, description="column whose distinct non-empty values to count")


# --------------------------------------------------------------------------
# The eight tools
# --------------------------------------------------------------------------

def tool_table_info(registry: ArtifactRegistry, a: TableInfoArgs) -> ToolResult:
    """What is in a registered table: kind, row count, completeness, study/variable IDs, measurement metadata
    (trait, unit, timepoint when stated), column meanings, quality notes, value-token counts (valid / raw-missing /
    invalid), duplicate and independence facts, and a five-row labelled preview. Start here before any calculation."""
    table = _load(registry, a.artifact_id)
    m = table.manifest
    data: dict[str, Any] = {
        "artifact_id": m.artifact_id, "kind": m.kind, "row_count": m.row_count, "complete": m.complete, "sha256": m.sha256,
        "study_ids": list(m.study_ids), "variable_ids": list(m.variable_ids), "measurement": m.measurement.model_dump(),
        "columns": dict(m.columns), "quality_notes": [n for n in m.quality_notes][:20], "source_request_count": len(m.source_request_ids),
        "value_policy": VALUE_POLICY,
    }
    if VALUE_COL in m.columns:
        c = _classify(table, VALUE_COL)
        data["value_counts"] = {"total_rows": c.n_rows, "valid": c.n_valid, "raw_missing": c.n_missing, "invalid_or_nonfinite": c.n_invalid}
        data["n_independent_units"] = c.n_independent_units
        data["independence"] = ("every valid row names an observation unit" if c.units_known
                                else "unknown: no observation unit ID on every valid row; rows are not plots")
        data["duplicates"] = {"repeated_observation_ids": c.repeated_observation_ids, "repeated_measurements_per_unit": c.repeated_measurements,
                              "note": c.duplicate_note or "none detected"}
    preview = registry.preview(m.artifact_id, 5)
    data["preview"] = {"displayed_rows": preview.displayed_rows, "total_rows": preview.total_rows, "rows": preview.rows}
    return ToolResult(ok=True, data=data, artifact_ids=[m.artifact_id], complete=m.complete)


def _timing_warnings(table: Table, value_column: str) -> list[str]:
    """Describe missing timing on valid source rows, without assigning dates or growth stages."""
    if table.manifest.kind != "observations":
        return []
    valid = [row for row in table.rows if classify_value(row.get(value_column))[0] == "valid"]
    if not valid:
        return []
    missing = sum(not row.get("observationTimeStamp", "").strip() for row in valid)
    warnings = []
    if missing:
        warnings.append(f"Measurement dates are not recorded for {missing} of {len(valid)} valid source rows; "
                        "the timing of those measurements is unknown.")
    if not table.manifest.measurement.timepoint:
        warnings.append("Growth stage or timepoint is not stated in the source metadata; a shared growth stage cannot be verified.")
    return warnings


def tool_numeric_summary(registry: ArtifactRegistry, a: NumericSummaryArgs) -> ToolResult:
    """Counts (total, valid, raw-missing, invalid/non-finite) and mean, median, min, max, sample SD of the valid values
    of ONE variable, as typed claims with a named denominator, n_independent_units (or unknown) and the exclusions.
    Refuses mixed variables, unresolved duplicates and incomplete artifacts (unless allow_incomplete). SD with fewer
    than 2 valid values is null. Values keep full precision; `display` is rounded."""
    table = _load(registry, a.artifact_id)
    _require_column(table, a.value, "value column")
    caveat = _completeness_gate(table, a.allow_incomplete)
    variable = _single_variable(table)
    classified = _classify(table, a.value)
    used, transformation = _apply_duplicate_policy(table, classified, a.duplicate_policy)
    if caveat:
        transformation = f"{caveat}; {transformation}"
    unit = table.manifest.measurement.unit
    denominator_name = ("valid numeric values, one per observation unit" if classified.units_known else "valid numeric values (rows; independence unknown)")
    stats, claims = _numeric_claims(table, classified, used, transformation, grouping=[], filters={}, unit=unit,
                                   denominator_name=denominator_name,
                                   label_parts=() if a.value == VALUE_COL else (a.value,),
                                   claim_namespace=None if a.value == VALUE_COL else "numeric-value")
    warnings = [w for w in (classified.duplicate_note if a.duplicate_policy != "reject" else None, caveat) if w]
    if unit is None:
        warnings.append("unit not stated in the artifact's measurement metadata; claims carry unit=null")
    warnings.extend(_timing_warnings(table, a.value))
    data = {
        "artifact_id": table.manifest.artifact_id, "sha256": table.manifest.sha256, "variable_id": variable, "unit": unit,
        "value_column": a.value, "counts": {"total_rows": classified.n_rows, "valid": classified.n_valid, "raw_missing": classified.n_missing,
                                            "invalid_or_nonfinite": classified.n_invalid, "values_used": len(used)},
        "n_independent_units": claims[0].n_independent_units, "stats": stats, "display": {k: _display(v) for k, v in stats.items()},
        "transformation": transformation, "exclusions": [e.model_dump() for e in _exclusions(classified)],
        "claims": _claims_json(claims), "value_policy": VALUE_POLICY, "analysis_target": ANALYSIS_TARGET,
    }
    return ToolResult(ok=True, data=data, artifact_ids=[table.manifest.artifact_id], complete=table.manifest.complete, warnings=warnings)


GROUP_RESULT_COLUMNS = {
    "group_id": "exact group ID text", "label": "name carried from the label column, if any", "included": "true/false",
    "exclusion_reason": "why the group was excluded ('' if included)", "rank": "rank among included groups (ties share a rank)",
    "n_rows": "source rows in the group", "n_valid_values": "valid numeric values used", "n_raw_missing": "raw-missing tokens",
    "n_invalid": "invalid or non-finite tokens", "n_independent_units": "distinct observation units among valid rows ('' = unknown)",
    "mean": "full precision", "median": "full precision", "min": "full precision", "max": "full precision", "sd": "sample SD, full precision ('' if undefined)",
}


def tool_group_stats(registry: ArtifactRegistry, a: GroupStatsArgs) -> ToolResult:
    """Per-group statistics of ONE variable, grouped by a stable ID column with names carried as labels. Each group
    reports n_valid_values and n_independent_units separately. Groups under min_independent_n are EXCLUDED with a
    reason (an exclusion threshold, not a design correction); rows with an empty group ID are excluded and counted.
    The FULL result (every group, included or not) is saved as an analysis_result artifact; the message shows top_n
    included groups ranked by mean at full precision, ties shared and listed. Descriptive only: not adjusted, not
    evidence of superiority."""
    table = _load(registry, a.artifact_id)
    _require_column(table, a.by, "group column")
    _require_column(table, a.value, "value column")
    if a.by.endswith("Name"):
        raise AnalystError("invalid_argument", f"group by the ID column, not {a.by!r}; names are carried as labels")
    caveat = _completeness_gate(table, a.allow_incomplete)
    variable = _single_variable(table)
    classified = _classify(table, a.value)
    used, transformation = _apply_duplicate_policy(table, classified, a.duplicate_policy)
    if caveat:
        transformation = f"{caveat}; {transformation}"
    unit = table.manifest.measurement.unit
    label_col = LABEL_FOR.get(a.by)
    label_col = label_col if label_col in table.manifest.columns else None

    # every source row is attributed to a group first (so missing/invalid counts are per group too)
    per_group_rows: dict[str, list[dict[str, str]]] = {}
    for row in table.rows:
        per_group_rows.setdefault(row.get(a.by, ""), []).append(row)
    per_group_used: dict[str, list[tuple[dict[str, str], float]]] = {}
    for row, number in used:
        per_group_used.setdefault(row.get(a.by, ""), []).append((row, number))

    results: list[dict[str, Any]] = []
    no_group_rows = len(per_group_rows.get("", []))
    for group_id in sorted(per_group_rows):
        if group_id == "":
            continue
        rows = per_group_rows[group_id]
        counts = Counter(classify_value(r.get(a.value))[0] for r in rows)
        values = [n for _, n in per_group_used.get(group_id, [])]
        n_units = len({r[UNIT_COL] for r, _ in per_group_used.get(group_id, [])}) if classified.units_known else None
        basis = n_units if n_units is not None else len(values)
        reason = ""
        if basis < a.min_independent_n:
            what = "independent units" if n_units is not None else "valid values (independence unknown)"
            reason = f"fewer than {a.min_independent_n} {what}: {basis}"
        labels = {r.get(label_col, "") for r in rows if r.get(label_col, "")} if label_col else set()
        results.append({"group_id": group_id, "label": "; ".join(sorted(labels)), "included": reason == "", "exclusion_reason": reason,
                        "n_rows": len(rows), "n_valid_values": len(values), "n_raw_missing": counts["missing"], "n_invalid": counts["invalid"],
                        "n_independent_units": n_units, **_stats(values)})

    included = [g for g in results if g["included"]]
    included.sort(key=lambda g: ((g["mean"] if a.ascending else -g["mean"]), g["group_id"]))
    for g in included:                                            # competition ranking: ties share a rank
        g["rank"] = 1 + sum(1 for o in included if (o["mean"] < g["mean"] if a.ascending else o["mean"] > g["mean"]))
    for g in results:
        g.setdefault("rank", None)
    tie_groups = [sorted(x["group_id"] for x in included if x["mean"] == v) for v in sorted({g["mean"] for g in included})]
    ties = [t for t in tie_groups if len(t) > 1]

    lineage = [f"lineage: group_stats by {a.by} on {table.manifest.artifact_id} sha256={table.manifest.sha256}",
               f"transformation: {transformation}", f"min_independent_n={a.min_independent_n} (exclusion threshold)",
               f"rows with an empty {a.by}: {no_group_rows} (excluded, not dropped silently)", f"analysis target: {ANALYSIS_TARGET}"]
    text_rows = [{**g, "included": "true" if g["included"] else "false",
                  "n_independent_units": "" if g["n_independent_units"] is None else str(g["n_independent_units"]),
                  "rank": "" if g["rank"] is None else str(g["rank"]),
                  **{k: ("" if g[k] is None else repr(g[k])) for k in ("mean", "median", "min", "max", "sd")}} for g in results]
    manifest = registry.save_table(text_rows, kind="analysis_result", columns=GROUP_RESULT_COLUMNS, complete=table.manifest.complete,
                                   source_request_ids=list(table.manifest.source_request_ids), study_ids=list(table.manifest.study_ids),
                                   variable_ids=list(table.manifest.variable_ids), measurement=table.manifest.measurement, quality_notes=lineage)

    exclusions = _exclusions(classified)
    if no_group_rows:
        exclusions.append(Exclusion(reason=f"rows with an empty {a.by}", count=no_group_rows))
    excluded_groups = [g for g in results if not g["included"]]
    if excluded_groups:
        exclusions.append(Exclusion(reason=f"groups under min_independent_n={a.min_independent_n}", count=len(excluded_groups)))
    shown = included[:a.top_n]
    claims: list[Claim] = []
    for g in shown:
        denominator = Denominator(name=f"valid numeric values in group {g['group_id']}" + ("" if classified.units_known else " (rows; independence unknown)"),
                                  n=g["n_valid_values"])
        claims.append(Claim(claim_id=_claim_id(table.manifest.artifact_id, "mean", a.by, g["group_id"],
                                                 *(() if a.value == VALUE_COL else (a.value,)),
                                                 namespace=None if a.value == VALUE_COL else "group-value"), kind="mean", value=g["mean"], unit=unit,
                            denominator=denominator, n_independent_units=g["n_independent_units"], source_artifact_hashes=[table.manifest.sha256],
                            filters={a.by: g["group_id"]}, grouping=[a.by], transformation=f"group mean of valid values; {transformation}",
                            exclusions=exclusions))
    data = {
        "artifact_id": table.manifest.artifact_id, "result_artifact_id": manifest.artifact_id, "variable_id": variable, "unit": unit, "by": a.by,
        "label_column": label_col, "groups_total": len(results), "groups_included": len(included), "groups_excluded": len(excluded_groups),
        "rows_without_group_id": no_group_rows, "order": "ascending" if a.ascending else "descending",
        "shown": [{**g, "display_mean": _display(g["mean"])} for g in shown],
        "excluded": [{"group_id": g["group_id"], "label": g["label"], "reason": g["exclusion_reason"]} for g in excluded_groups],
        "ties": ties, "tie_rule": "equal means share a rank; order inside a tie is by group ID",
        "transformation": transformation, "min_independent_n": a.min_independent_n, "claims": _claims_json(claims),
        "exclusions": [e.model_dump() for e in exclusions], "analysis_target": ANALYSIS_TARGET, "value_policy": VALUE_POLICY,
    }
    warnings = [w for w in (caveat, classified.duplicate_note if a.duplicate_policy != "reject" else None) if w]
    if not classified.units_known:
        warnings.append("independence unknown: min_independent_n was applied to valid values, not to units")
    warnings.extend(_timing_warnings(table, a.value))
    return ToolResult(ok=True, data=data, artifact_ids=[table.manifest.artifact_id, manifest.artifact_id], complete=table.manifest.complete, warnings=warnings)


def tool_filter_rows(registry: ArtifactRegistry, a: FilterRowsArgs) -> ToolResult:
    """Keep the rows where one column satisfies an allowlisted typed test and save them as a NEW artifact (same
    columns, lineage recorded). eq/ne compare exact text; contains is a literal case-insensitive substring (never a
    regular expression); gt/ge/lt/le compare valid numbers only and report how many rows had no valid number to
    compare; is_missing / is_valid_number test the value policy."""
    table = _load(registry, a.artifact_id)
    _require_column(table, a.column)
    needle = a.value
    catalog_comparison = (table.manifest.kind in {"studies", "variables", "locations", "programs", "seasons"}
                          and a.op in {"eq", "ne", "contains"} and bool(needle.strip()))
    unknown_cells = sum(not row.get(a.column, "").strip() or row.get(a.column, "").strip() == "[]"
                        for row in table.rows) if catalog_comparison else 0
    if table.rows and unknown_cells == len(table.rows):
        raise AnalystError("incomplete_data", f"{a.column} is missing for all {unknown_cells} catalog rows; "
                           "matching membership is unknown, not zero. Ask for a recorded field or more metadata.")
    threshold: float | None = None
    if a.op in ("gt", "ge", "lt", "le"):
        kind, threshold = classify_value(needle)
        if kind != "valid":
            raise AnalystError("invalid_argument", f"{a.op} needs a valid number to compare with, not {needle!r}")
    kept: list[dict[str, str]] = []
    dropped_non_numeric = 0
    for row in table.rows:
        cell = row.get(a.column, "")
        if catalog_comparison and (not cell.strip() or cell.strip() == "[]"):
            continue                    # absent metadata is neither a match nor a known non-match
        if a.op == "eq":
            keep = cell == needle
        elif a.op == "ne":
            keep = cell != needle
        elif a.op == "contains":
            keep = needle.lower() in cell.lower()
        elif a.op == "is_missing":
            keep = classify_value(cell)[0] == "missing"
        elif a.op == "is_valid_number":
            keep = classify_value(cell)[0] == "valid"
        else:
            kind, number = classify_value(cell)
            if kind != "valid":
                dropped_non_numeric += 1
                continue
            keep = {"gt": number > threshold, "ge": number >= threshold, "lt": number < threshold, "le": number <= threshold}[a.op]  # type: ignore[operator]
        if keep:
            kept.append(row)
    description = f"filter_rows: {a.column} {a.op} {needle!r}"
    lineage = [f"lineage: {description} on {table.manifest.artifact_id} sha256={table.manifest.sha256}",
               f"rows in: {len(table.rows)}, rows kept: {len(kept)}, rows without a valid number to compare: {dropped_non_numeric}"]
    membership_incomplete = bool(unknown_cells and a.column in {"programDbId", "programName"})
    warnings = ([f"{unknown_cells} of {len(table.rows)} catalog rows have no {a.column}; {len(kept)} known matches were found, "
                 "but the full matching count is unknown. Counts describe recorded matches only."
                 + (" The filtered table is incomplete." if membership_incomplete else "")] if unknown_cells else [])
    lineage.extend(warnings)
    manifest = registry.save_table(kept, kind=table.manifest.kind, columns=dict(table.manifest.columns),
                                   complete=table.manifest.complete and not membership_incomplete,
                                   source_request_ids=list(table.manifest.source_request_ids), study_ids=list(table.manifest.study_ids),
                                   variable_ids=list(table.manifest.variable_ids), measurement=table.manifest.measurement, quality_notes=lineage)
    preview = registry.preview(manifest.artifact_id, 5)
    data = {"source_artifact_id": table.manifest.artifact_id, "artifact_id": manifest.artifact_id, "filter": description,
            "rows_in": len(table.rows), "rows_out": len(kept), "dropped_non_numeric": dropped_non_numeric,
            "rows_with_unknown_filter_metadata": unknown_cells,
            "complete": manifest.complete, "preview": {"displayed_rows": preview.displayed_rows, "rows": preview.rows},
            "note": "counts describe recorded matches; incomplete program membership is blocked"}
    return ToolResult(ok=True, data=data, artifact_ids=[manifest.artifact_id], complete=manifest.complete, warnings=warnings)


def tool_concat_tables(registry: ArtifactRegistry, a: ConcatTablesArgs) -> ToolResult:
    """Stack two or more artifacts that are provably compatible: same columns, same kind, ONE shared variable, same
    measurement metadata (unit, method, scale, timepoint). Adds a source_artifact_id column so every row keeps its
    lineage; study IDs are the union. Refuses anything else with the reason. Complete only if every source is."""
    ids = list(a.artifact_ids)
    if len(set(ids)) != len(ids):
        raise AnalystError("invalid_argument", "the same artifact is listed twice")
    tables = [_load(registry, i) for i in ids]
    first = tables[0]
    variables: set[str] = set()
    for t in tables:
        if t.manifest.columns != first.manifest.columns:
            raise AnalystError("unsupported", f"incompatible schemas: {t.manifest.artifact_id} columns differ from {first.manifest.artifact_id}")
        if t.manifest.kind != first.manifest.kind:
            raise AnalystError("unsupported", f"incompatible kinds: {t.manifest.kind} vs {first.manifest.kind}")
        if t.manifest.measurement != first.manifest.measurement:
            mine, theirs = t.manifest.measurement.model_dump(), first.manifest.measurement.model_dump()
            differing = sorted(k for k in mine if mine[k] != theirs[k])
            raise AnalystError("unsupported", f"incompatible measurement metadata ({', '.join(differing)}): "
                                              f"{t.manifest.artifact_id} {mine} vs {first.manifest.artifact_id} {theirs}")
        variables |= set(t.manifest.variable_ids)
        if VARIABLE_COL in t.manifest.columns:
            variables |= {r.get(VARIABLE_COL, "") for r in t.rows if r.get(VARIABLE_COL, "")}
    if len(variables) > 1:
        raise AnalystError("unsupported", f"different variables cannot be stacked: {sorted(variables)}")
    if "source_artifact_id" in first.manifest.columns:
        raise AnalystError("unsupported", "the tables already carry a source_artifact_id column; stacking twice is not supported")
    rows: list[dict[str, str]] = []
    for t in tables:
        rows.extend({**r, "source_artifact_id": t.manifest.artifact_id} for r in t.rows)
    columns = {**first.manifest.columns, "source_artifact_id": "the artifact each row came from (lineage)"}
    study_ids = list(dict.fromkeys(s for t in tables for s in t.manifest.study_ids))
    request_ids = list(dict.fromkeys(r for t in tables for r in t.manifest.source_request_ids))
    lineage = [f"lineage: concat_tables of {t.manifest.artifact_id} sha256={t.manifest.sha256} rows={t.manifest.row_count} complete={t.manifest.complete}" for t in tables]
    repeated = 0
    if OBS_ID_COL in columns:
        counts = Counter(r.get(OBS_ID_COL, "") for r in rows if r.get(OBS_ID_COL, ""))
        repeated = sum(c - 1 for c in counts.values() if c > 1)
        if repeated:
            lineage.append(f"{repeated} repeated observation ID(s) across sources; numeric tools will require a duplicate rule")
    complete = all(t.manifest.complete for t in tables)
    manifest = registry.save_table(rows, kind="analysis_result", columns=columns, complete=complete, source_request_ids=request_ids,
                                   study_ids=study_ids, variable_ids=sorted(variables), measurement=first.manifest.measurement, quality_notes=lineage)
    data = {"artifact_id": manifest.artifact_id, "sources": ids, "row_count": manifest.row_count, "complete": complete, "study_ids": study_ids,
            "variable_ids": sorted(variables), "measurement": first.manifest.measurement.model_dump(), "repeated_observation_ids": repeated,
            "lineage": lineage}
    warnings = [] if complete else ["at least one source is incomplete; the stacked table is incomplete too"]
    return ToolResult(ok=True, data=data, artifact_ids=[manifest.artifact_id], complete=complete, warnings=warnings)


def tool_missing_report(registry: ArtifactRegistry, a: MissingReportArgs) -> ToolResult:
    """Raw-missing, invalid and valid counts for the value column, overall and per group. The denominator is rows
    unless a verified observation_units roster for the same study is supplied; then it also reports expected units,
    units with a valid value, units with only missing/invalid values, units absent from the table, and rows whose
    unit is not on the roster. Missing is never turned into zero."""
    table = _load(registry, a.artifact_id)
    _require_column(table, a.value, "value column")
    c = _classify(table, a.value)
    data: dict[str, Any] = {
        "artifact_id": table.manifest.artifact_id, "complete": table.manifest.complete,
        "overall": {"total_rows": c.n_rows, "valid": c.n_valid, "raw_missing": c.n_missing, "invalid_or_nonfinite": c.n_invalid},
        "denominators": {"rows": c.n_rows}, "value_policy": VALUE_POLICY,
    }
    if a.by is not None:
        _require_column(table, a.by, "group column")
        groups: dict[str, Counter] = {}
        for row in table.rows:
            groups.setdefault(row.get(a.by, "") or MISSING_GROUP, Counter())[classify_value(row.get(a.value))[0]] += 1
        data["by"] = a.by
        data["null_group_policy"] = f"rows with an empty {a.by} are counted under {MISSING_GROUP!r}, never dropped"
        data["groups"] = [{"group_id": g, "total_rows": sum(k.values()), "valid": k["valid"], "raw_missing": k["missing"], "invalid_or_nonfinite": k["invalid"]}
                          for g, k in sorted(groups.items())]
    if a.roster_artifact_id is None:
        data["expected_units"] = None
        data["expected_units_note"] = "no verified unit roster supplied; the only denominator is rows"
    else:
        roster = _load(registry, a.roster_artifact_id)
        if roster.manifest.kind != "observation_units":
            raise AnalystError("invalid_argument", f"roster {roster.manifest.artifact_id} is {roster.manifest.kind}, not observation_units")
        if not roster.manifest.complete:
            raise AnalystError("incomplete_data", f"roster {roster.manifest.artifact_id} is not a complete collection; expected units are unknown")
        if set(roster.manifest.study_ids) != set(table.manifest.study_ids):
            raise AnalystError("invalid_argument", f"roster studies {sorted(roster.manifest.study_ids)} differ from the table's {sorted(table.manifest.study_ids)}")
        _require_column(roster, UNIT_COL, "roster unit column")
        _require_column(table, UNIT_COL, "unit column")
        expected = {r.get(UNIT_COL, "") for r in roster.rows if r.get(UNIT_COL, "")}
        units_valid = {r.get(UNIT_COL, "") for r, _ in c.valid if r.get(UNIT_COL, "")}
        units_present = {r.get(UNIT_COL, "") for r in table.rows if r.get(UNIT_COL, "")}
        off_roster_rows = sum(1 for r in table.rows if r.get(UNIT_COL, "") and r.get(UNIT_COL, "") not in expected)
        data["expected_units"] = len(expected)
        data["units"] = {"with_valid_value": len(units_valid & expected), "present_without_valid_value": len((units_present - units_valid) & expected),
                         "absent_from_table": len(expected - units_present), "rows_with_unit_not_on_roster": off_roster_rows,
                         "absent_ids": sorted(expected - units_present)[:20]}
        data["denominators"]["expected_units"] = len(expected)
        data["roster_artifact_id"] = roster.manifest.artifact_id
    return ToolResult(ok=True, data=data, artifact_ids=[table.manifest.artifact_id] + ([a.roster_artifact_id] if a.roster_artifact_id else []),
                      complete=table.manifest.complete)


def tool_plot_bar(registry: ArtifactRegistry, a: PlotBarArgs) -> ToolResult:
    """Optional bar chart of a registered result. Plotting is NOT enabled in this build: matplotlib is not an approved
    dependency, so this tool returns unavailable and installs nothing. Use group_stats / count_records tables instead."""
    if not PLOTTING_ENABLED:
        return ToolResult(ok=False, error=ToolError(code="unsupported", message="plotting is unavailable: matplotlib is not an approved "
                                                                                "dependency of part2 and nothing was installed"),
                          data={"plotting_enabled": False, "requested": {"artifact_id": a.artifact_id, "x": a.x, "y": a.y, "top_n": a.top_n}})
    raise AnalystError("unsupported", "plotting is enabled by flag but not implemented in this build")   # pragma: no cover


def tool_count_records(registry: ArtifactRegistry, a: CountRecordsArgs) -> ToolResult:
    """Exact counts from a COMPLETE table (an incomplete one is refused): total rows, or value frequencies of one
    column (group_by), or the number of distinct non-empty values of one column (distinct_by). IDs are compared as
    exact text ('007' is not '7'). Empty cells form their own explicit group and are never dropped."""
    table = _load(registry, a.artifact_id)
    if not table.manifest.complete:
        raise AnalystError("incomplete_data", f"{table.manifest.artifact_id} is not a complete collection; a count from it would not be a count of the catalog")
    hashes = [table.manifest.sha256]
    denominator = Denominator(name=f"rows in the complete {table.manifest.kind} table", n=table.manifest.row_count)
    claims: list[Claim] = []
    data: dict[str, Any] = {"artifact_id": table.manifest.artifact_id, "kind": table.manifest.kind, "total_rows": table.manifest.row_count,
                            "null_group_policy": f"empty cells are counted under {MISSING_GROUP!r}, never dropped", "value_policy": VALUE_POLICY["ids"]}
    if a.group_by is None and a.distinct_by is None:
        claims.append(Claim(claim_id=_claim_id(table.manifest.artifact_id, "count", "rows"), kind="count", value=table.manifest.row_count,
                            denominator=denominator, source_artifact_hashes=hashes, transformation="row count of a complete table"))
    if a.group_by is not None:
        _require_column(table, a.group_by, "group_by column")
        counts = Counter(r.get(a.group_by, "") or MISSING_GROUP for r in table.rows)
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        data["group_by"] = a.group_by
        data["frequencies"] = [{"value": k, "count": n, "rank": 1 + sum(1 for _, m in ordered if m > n)} for k, n in ordered]
        ties = [sorted(k for k, m in ordered if m == n) for n in sorted({m for _, m in ordered})]
        data["ties"] = [t for t in ties if len(t) > 1]
        data["tie_rule"] = "equal counts share a rank; order inside a tie is by value text"
        for k, n in ordered:
            claims.append(Claim(claim_id=_claim_id(table.manifest.artifact_id, "count", a.group_by, k, namespace="frequency"), kind="count", value=n, denominator=denominator,
                                source_artifact_hashes=hashes, filters={a.group_by: k}, grouping=[a.group_by], transformation="frequency of exact text values"))
    if a.distinct_by is not None:
        _require_column(table, a.distinct_by, "distinct_by column")
        values = [r.get(a.distinct_by, "") for r in table.rows]
        distinct = sorted({v for v in values if v})
        data["distinct_by"] = a.distinct_by
        data["distinct_count"] = len(distinct)
        data["empty_cells"] = sum(1 for v in values if not v)
        data["distinct_values_preview"] = distinct[:20]
        claims.append(Claim(claim_id=_claim_id(table.manifest.artifact_id, "count", "distinct", a.distinct_by), kind="count", value=len(distinct),
                            denominator=denominator, source_artifact_hashes=hashes, grouping=[a.distinct_by],
                            transformation="number of distinct non-empty exact text values",
                            exclusions=[Exclusion(reason="empty cells", count=data["empty_cells"])] if data["empty_cells"] else []))
    data["claims"] = _claims_json(claims)
    return ToolResult(ok=True, data=data, artifact_ids=[table.manifest.artifact_id], complete=True)


# --------------------------------------------------------------------------
# ONE registry of the eight tools, schemas and dispatch
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class AnalystToolSpec:
    name: str
    args_model: type[ToolArgs]
    fn: Callable[[ArtifactRegistry, Any], ToolResult]

    @property
    def description(self) -> str:
        return " ".join((self.fn.__doc__ or "").split())

    def input_schema(self) -> dict[str, Any]:
        return self.args_model.model_json_schema()


ANALYST_TOOLS: dict[str, AnalystToolSpec] = {spec.name: spec for spec in [
    AnalystToolSpec("table_info", TableInfoArgs, tool_table_info),
    AnalystToolSpec("numeric_summary", NumericSummaryArgs, tool_numeric_summary),
    AnalystToolSpec("group_stats", GroupStatsArgs, tool_group_stats),
    AnalystToolSpec("filter_rows", FilterRowsArgs, tool_filter_rows),
    AnalystToolSpec("concat_tables", ConcatTablesArgs, tool_concat_tables),
    AnalystToolSpec("missing_report", MissingReportArgs, tool_missing_report),
    AnalystToolSpec("plot_bar", PlotBarArgs, tool_plot_bar),
    AnalystToolSpec("count_records", CountRecordsArgs, tool_count_records),
]}
ANALYST_TOOL_NAMES = list(ANALYST_TOOLS)
assert len(ANALYST_TOOL_NAMES) == 8


def analyst_tool_schemas() -> list[dict[str, Any]]:
    """[{name, description, inputSchema}] — the same models dispatch_analyst validates with."""
    return [{"name": s.name, "description": s.description, "inputSchema": s.input_schema()} for s in ANALYST_TOOLS.values()]


def _map_exception(exc: Exception) -> ToolError:
    if isinstance(exc, AnalystError):
        return ToolError(code=exc.code, message=exc.message[:500])   # type: ignore[arg-type]
    if isinstance(exc, ArtifactError):
        mapping = {"invalid_handle": "invalid_argument", "invalid_argument": "invalid_argument"}
        return ToolError(code=mapping.get(exc.code, "internal_error"), message=exc.message[:500])
    if isinstance(exc, ValidationError):
        first = exc.errors()[0]
        return ToolError(code="internal_error", message=f"contract violation at {list(first.get('loc', []))}: {first.get('msg')}")
    return ToolError(code="internal_error", message=f"{type(exc).__name__}: {str(exc)[:300]}")


def dispatch_analyst(registry: ArtifactRegistry, name: str, arguments: Mapping[str, Any] | None) -> ToolResult:
    """Validate arguments against the tool's model, run the fixed function, ALWAYS return a ToolResult."""
    spec = ANALYST_TOOLS.get(name)
    if spec is None:
        return ToolResult(ok=False, error=ToolError(code="invalid_argument", message=f"unknown tool {name!r}; known: {ANALYST_TOOL_NAMES}"))
    arguments = {} if arguments is None else arguments
    if not isinstance(arguments, Mapping):
        return ToolResult(ok=False, error=ToolError(code="invalid_argument", message="arguments must be a JSON object"))
    try:
        args = spec.args_model.model_validate(dict(arguments))
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(p) for p in first.get("loc", ())) or "(root)"
        return ToolResult(ok=False, error=ToolError(code="invalid_argument", message=f"{loc}: {first.get('msg')}"))
    try:
        return spec.fn(registry, args)
    except Exception as exc:  # noqa: BLE001 - every failure becomes a coded ToolResult
        return ToolResult(ok=False, error=_map_exception(exc))


class AnalystTools:
    """The eight tools behind the agent loop's ToolInterface (async schemas()/call()), in-process, no network."""

    def __init__(self, registry: ArtifactRegistry) -> None:
        self.registry = registry
        self.calls_made = 0

    async def schemas(self) -> list[dict[str, Any]]:
        return analyst_tool_schemas()

    async def call(self, name: str, args: Mapping[str, Any] | None) -> ToolResult:
        import anyio

        self.calls_made += 1
        return await anyio.to_thread.run_sync(dispatch_analyst, self.registry, name, dict(args or {}))


def main(argv: list[str] | None = None) -> int:   # pragma: no cover - tiny helper for humans
    """Print the eight tool schemas; there is no other command-line behaviour."""
    import json

    names = argv if argv else ANALYST_TOOL_NAMES
    for spec in analyst_tool_schemas():
        if spec["name"] in names:
            print(json.dumps(spec, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
