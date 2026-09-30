"""
contracts.py — the strict data contracts every part2 module must speak.

A contract is an agreement about the shape and meaning of data handed from
one file to the next, so the next file never has to guess. Every record here:

* is STRICT: no silent type conversion ("1" is not 1, 1 is not "1");
* FORBIDS EXTRA FIELDS: an unknown key is an error, which is how a model is
  stopped from smuggling in authority such as {"approved": true};
* is FROZEN: once built it cannot be changed, so an approval or a manifest
  cannot be edited after the fact;
* is VERSIONED: contract_version "1" is fixed on every record;
* carries string IDs (never integers) and NO NaN or infinity anywhere;
* says WHY a number is missing instead of writing 0.

There is deliberately no "ignore validation" switch. A malformed record is
rejected, not corrected.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timedelta
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

__all__ = [
    "ContractError",
    "ValidationError",
    "CONTRACT_VERSION",
    "AUTHORITY_KEYS",
    "StrictRecord",
    "StudyFilters",
    "RequestError",
    "RequestRecord",
    "CollectionResult",
    "MeasurementMeta",
    "ArtifactManifest",
    "ToolError",
    "ToolResult",
    "FetchApproval",
    "Denominator",
    "Exclusion",
    "Claim",
    "NormalizedMessage",
    "Usage",
    "StructuredError",
    "AgentResult",
    "ResolvedEntity",
    "StudyStatusEntry",
    "RetrievalReport",
    "AnalysisReport",
    "MetadataFact",
    "Clarification",
    "ResolvedScope",
    "Statistic",
    "PlanStep",
    "Plan",
    "FetchManifest",
    "RunCounts",
    "RunResult",
    "approval_mismatches",
    "MAX_PLAN_STEPS",
    "dump_json",
]

CONTRACT_VERSION = "1"


class ContractError(ValueError):
    """Raised by helper functions in this module (pydantic itself raises ValidationError)."""


# --------------------------------------------------------------------------
# Shared vocabularies. A Literal is a fixed list of allowed words; anything
# else is rejected — that is how an "unknown status" becomes an error.
# --------------------------------------------------------------------------

# Did we get the whole collection? complete / empty are the only "trust it" states.
CollectionStatus = Literal["complete", "empty", "incomplete", "unsupported", "failed", "unknown"]
# The per-study ledger adds "blocked": the fetch was refused by the approval guard.
StudyFetchStatus = Literal["complete", "empty", "incomplete", "unsupported", "failed", "unknown", "blocked"]
# How an agent run ended (the Task 3B vocabulary).
TerminalStatus = Literal["completed", "needs_clarification", "incomplete", "blocked", "failed", "limit_reached"]
# How a whole run ended (Stage 5): the agent vocabulary plus "canceled" (Ctrl+C or a human's cancel).
ExecutionStatus = Literal["completed", "needs_clarification", "incomplete", "blocked", "failed", "limit_reached", "canceled"]
# The human's separate verdict on a finished run. Software finishing is not a human agreeing.
Acceptance = Literal["pending", "accepted", "rejected"]
PlanStatus = Literal["draft", "needs_clarification", "ready"]
PlanAgent = Literal["retriever", "analyst"]          # the only agents a plan may name
FetchOrigin = Literal["snapshot", "live"]            # where observation data may come from
Origin = Literal["cache", "live"]
EndpointFamily = Literal[
    "serverinfo", "commoncropnames", "studies", "observationvariables",
    "observations", "observationunits", "locations", "programs", "seasons",
]
ArtifactKind = Literal[
    "studies", "variables", "locations", "programs", "seasons",
    "observations", "observation_units", "analysis_result", "snapshot",
]
ToolErrorCode = Literal[
    "invalid_argument", "not_found", "not_authorized", "incomplete_data",
    "unsupported", "upstream_error", "timeout", "budget_exhausted", "internal_error",
]
RequestErrorCode = Literal[
    "connection", "timeout", "http_error", "invalid_json", "not_brapi",
    "blocked", "cancelled", "too_large", "unknown",
]
ClaimKind = Literal["mean", "median", "min", "max", "sd", "count", "ratio", "rank"]
EntityKind = Literal["study", "variable", "location", "season", "program", "trait"]
MessageRole = Literal["system", "user", "assistant", "tool"]

# Keys that would carry AUTHORITY if a model could write them into a payload.
# Approval comes from controller code only, never from model text.
AUTHORITY_KEYS = frozenset({
    "fetch_approval", "approval", "approved", "approval_id", "authorized",
    "authorization", "allow_live", "live_access", "max_http_attempts",
    "max_observation_studies", "human_acceptance", "accepted",
})

_SECRET_HINT = re.compile(r"(token|api[_-]?key|password|secret|authorization)\s*=", re.IGNORECASE)


# --------------------------------------------------------------------------
# Validation helpers, attached to reusable field types below
# --------------------------------------------------------------------------

def _require_utc(value: datetime) -> datetime:
    """A time is only meaningful if it says which clock it is on. We require UTC."""
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("datetime must be timezone-aware UTC (tzinfo=timezone.utc)")
    return value


def _normalized_base_url(value: str) -> str:
    """Accept only a clean http(s) base: no credentials, query, fragment or trailing slash."""
    parts = urlsplit(value)
    if parts.scheme not in ("http", "https"):
        raise ValueError("base_url must start with http:// or https://")
    if not parts.hostname:
        raise ValueError("base_url has no host")
    if parts.username or parts.password or "@" in parts.netloc:
        raise ValueError("base_url must not contain credentials")
    if parts.query or parts.fragment:
        raise ValueError("base_url must not contain a query string or fragment")
    if value.endswith("/"):
        raise ValueError("base_url must not end with '/' (normalized form)")
    if parts.scheme != parts.scheme.lower() or parts.netloc != parts.netloc.lower():
        raise ValueError("base_url scheme and host must be lower-case (normalized form)")
    return value


def _managed_relative_path(value: str) -> str:
    """A path the APPLICATION chose, relative to its own folders. Never a user/model path."""
    if not value:
        raise ValueError("path must not be empty")
    if "\\" in value:
        raise ValueError("managed paths use '/' separators only")
    if value.startswith("/") or re.match(r"^[A-Za-z]:", value):
        raise ValueError("managed path must be relative, not absolute")
    if any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError("managed path must not contain '', '.' or '..' segments")
    return value


def _endpoint_path(value: str) -> str:
    if not value.startswith("/"):
        raise ValueError("endpoint must start with '/' (for example '/studies')")
    if "?" in value or "#" in value or "//" in value or ".." in value:
        raise ValueError("endpoint must be a plain path without query, fragment or traversal")
    return value


def _redacted_url(value: str) -> str:
    """The URL we may write in logs: no credentials, no secret-looking parameters."""
    parts = urlsplit(value)
    if parts.username or parts.password or "@" in parts.netloc:
        raise ValueError("redacted URL must not contain credentials")
    if _SECRET_HINT.search(value):
        raise ValueError("redacted URL still contains a secret-looking parameter")
    return value


def _unique(values: list[Any], label: str) -> list[Any]:
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must not contain duplicates")
    return values


def _walk(value: Any):
    """Yield every (key, item) pair nested anywhere inside a JSON-like value."""
    stack: list[tuple[Any, Any]] = [(None, value)]
    while stack:
        key, item = stack.pop()
        yield key, item
        if isinstance(item, dict):
            stack.extend(item.items())
        elif isinstance(item, (list, tuple)):
            stack.extend((None, sub) for sub in item)


def _reject_nonfinite(value: Any, where: str) -> None:
    for _, item in _walk(value):
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"{where} contains NaN or infinity; a missing number must be null with a reason")


def _reject_authority(value: Any, where: str) -> None:
    for key, _ in _walk(value):
        if isinstance(key, str) and key.lower() in AUTHORITY_KEYS:
            raise ValueError(f"{where} contains authority field {key!r}; approval never comes from a payload")


# Reusable field types. Annotated[...] = "this type, plus these checks".
IdStr = Annotated[str, Field(min_length=1, pattern=r"^\S+$")]          # a string ID, never an int
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
ArtifactId = Annotated[str, Field(pattern=r"^art_[A-Za-z0-9_-]+$")]
ApprovalId = Annotated[str, Field(pattern=r"^appr_[A-Za-z0-9_-]+$")]
ClaimId = Annotated[str, Field(pattern=r"^clm_[A-Za-z0-9_-]+$")]
Count = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(ge=1)]
HttpStatus = Annotated[int, Field(ge=100, le=599)]
Seconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
NonBlank = Annotated[str, Field(min_length=1, pattern=r"\S")]
UtcDatetime = Annotated[datetime, AfterValidator(_require_utc)]
BaseUrl = Annotated[str, AfterValidator(_normalized_base_url)]
EndpointPath = Annotated[str, AfterValidator(_endpoint_path)]
RedactedUrl = Annotated[str, AfterValidator(_redacted_url)]
ManagedPath = Annotated[str, AfterValidator(_managed_relative_path)]


# --------------------------------------------------------------------------
# Base record
# --------------------------------------------------------------------------

class StrictRecord(BaseModel):
    """Every contract inherits these rules. See the module docstring."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True, ser_json_inf_nan="strings")

    contract_version: Literal["1"] = CONTRACT_VERSION


# --------------------------------------------------------------------------
# 1. StudyFilters
# --------------------------------------------------------------------------

class StudyFilters(StrictRecord):
    """What the user is asking the studies endpoint for.

    Exact filters use resolved IDs. Discovery filters use human text
    (name_contains, location_name, season_name) and only find CANDIDATES.
    Giving an ID and a name for the same thing is rejected: resolve the name
    to an ID first, then filter — never silently combine the two.

    Everyday example: a library catalogue. "Shelf B12" is an exact ID; "title
    contains 'cassava'" is discovery. You would not say "shelf B12, but also
    any shelf whose name has 'east' in it" and expect one clear answer.
    """

    location_id: IdStr | None = None
    season_id: IdStr | None = None
    study_type: NonBlank | None = None
    program_id: IdStr | None = None
    name_contains: NonBlank | None = None
    location_name: NonBlank | None = None   # discovery candidate, not an exact filter
    season_name: NonBlank | None = None     # discovery candidate, not an exact filter

    @model_validator(mode="after")
    def _no_id_plus_name(self) -> "StudyFilters":
        if self.location_id is not None and self.location_name is not None:
            raise ValueError("location_id and location_name given together; resolve the name to an ID first")
        if self.season_id is not None and self.season_name is not None:
            raise ValueError("season_id and season_name given together; resolve the name to an ID first")
        return self

    @property
    def is_exact(self) -> bool:
        """True when no discovery text is present, so results can be treated as an exact filter."""
        return self.name_contains is None and self.location_name is None and self.season_name is None


# --------------------------------------------------------------------------
# 2. RequestRecord
# --------------------------------------------------------------------------

class RequestError(StrictRecord):
    """Why one request failed: a fixed code plus a message — never a bare string."""

    code: RequestErrorCode
    message: NonBlank
    retryable: bool = False


class RequestRecord(StrictRecord):
    """The receipt for one HTTP request (or one cache read) to a BrAPI server.

    Everyday example: a courier's delivery slip — who asked (run_id), what
    address (base_url + endpoint + params), when it was requested and
    delivered, whether it came from the warehouse (cache) or the road (live),
    the parcel's seal number (response_sha256) and, if it failed, the coded
    reason. The slip never contains the customer's password.
    """

    request_id: IdStr
    run_id: IdStr
    base_url: BaseUrl
    endpoint: EndpointPath
    params: dict[str, str | int | bool] = Field(default_factory=dict)
    requested_at_utc: UtcDatetime
    fetched_at_utc: UtcDatetime | None = None
    origin: Origin
    http_status: HttpStatus | None = None
    attempt: PositiveInt
    duration_seconds: Seconds
    response_sha256: Sha256 | None = None
    source_url_redacted: RedactedUrl
    error: RequestError | None = None

    @field_validator("params")
    @classmethod
    def _param_keys(cls, value: dict[str, str | int | bool]) -> dict[str, str | int | bool]:
        for key in value:
            if not key.strip():
                raise ValueError("parameter names must not be blank")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> "RequestRecord":
        if self.error is None:
            if self.response_sha256 is None:
                raise ValueError("a successful request must record response_sha256")
            if self.fetched_at_utc is None:
                raise ValueError("a successful request must record fetched_at_utc")
            if self.http_status is None or not (200 <= self.http_status < 300):
                raise ValueError("a request without error must have a 2xx http_status")
        if (self.origin == "live" and self.fetched_at_utc is not None
                and self.fetched_at_utc < self.requested_at_utc):
            raise ValueError("a live fetch cannot finish before it was requested")
        # origin == "cache": fetched_at_utc is the ORIGINAL server fetch, normally
        # earlier than this request — that is the whole point of a cache hit.
        return self

    @property
    def ok(self) -> bool:
        return self.error is None


# --------------------------------------------------------------------------
# 3. CollectionResult
# --------------------------------------------------------------------------

class CollectionResult(StrictRecord):
    """A set of records from one endpoint, plus PROOF of whether it is the whole set.

    status meanings:
      complete    — validated evidence says every record is here
      empty       — validated evidence says there are zero records
      incomplete  — more pages exist or a limit was hit; DO NOT analyse as a whole
      unsupported — the server does not offer this endpoint/filter
      failed      — the request(s) failed
      unknown     — the server gave no usable count; completeness cannot be claimed

    Everyday example: a class roster. 25 names on the sheet and the office
    says the class has 25 → complete. 25 names but the office says 40 →
    incomplete; you would not compute the class average from it.
    """

    records: list[dict[str, Any]] = Field(default_factory=list)
    returned_count: Count
    reported_total: Count | None = None
    complete: bool
    status: CollectionStatus
    request_ids: list[IdStr] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("records")
    @classmethod
    def _records_finite(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        _reject_nonfinite(value, "records")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> "CollectionResult":
        if self.returned_count != len(self.records):
            raise ValueError(f"returned_count {self.returned_count} != len(records) {len(self.records)}")
        if self.reported_total is not None and self.returned_count > self.reported_total:
            raise ValueError("returned_count exceeds reported_total")
        if self.complete and self.status not in ("complete", "empty"):
            raise ValueError(f"complete=True is only allowed with status complete/empty, not {self.status!r}")
        if self.status == "complete":
            if not self.complete:
                raise ValueError("status complete requires complete=True")
            if self.returned_count == 0:
                raise ValueError("status complete with zero records; use status empty")
            if self.reported_total is not None and self.reported_total != self.returned_count:
                raise ValueError("status complete but reported_total != returned_count")
        if self.status == "empty":
            if not self.complete or self.returned_count != 0:
                raise ValueError("status empty requires complete=True and zero records")
            if self.reported_total not in (None, 0):
                raise ValueError("status empty but the server reported a non-zero total")
        return self


# --------------------------------------------------------------------------
# 4. ArtifactManifest
# --------------------------------------------------------------------------

class MeasurementMeta(StrictRecord):
    """What a numeric column means: trait, method, scale, unit, timepoint. None = not stated."""

    trait: str | None = None
    method: str | None = None
    scale: str | None = None
    unit: str | None = None
    timepoint: str | None = None


class ArtifactManifest(StrictRecord):
    """The label on a saved table (an artifact) that the application created.

    Tools refer to files by artifact_id, never by a path a model typed. The
    manifest records where the file is (a managed relative path), its SHA-256
    fingerprint, how many rows, whether the source collection was complete,
    and which requests, studies and variables it came from.

    Everyday example: a museum crate label — contents, origin, item count,
    a tamper seal (the hash) and the shipping receipts it was built from.
    """

    artifact_id: ArtifactId
    run_id: IdStr
    kind: ArtifactKind
    relative_path: ManagedPath
    sha256: Sha256
    schema_version: NonBlank
    row_count: Count
    complete: bool
    source_request_ids: list[IdStr] = Field(default_factory=list)
    study_ids: list[IdStr] = Field(default_factory=list)
    variable_ids: list[IdStr] = Field(default_factory=list)
    measurement: MeasurementMeta = Field(default_factory=MeasurementMeta)
    columns: dict[str, str] = Field(default_factory=dict)   # column name -> meaning
    quality_notes: list[str] = Field(default_factory=list)
    created_at_utc: UtcDatetime

    @field_validator("study_ids", "variable_ids", "source_request_ids")
    @classmethod
    def _no_dups(cls, value: list[str]) -> list[str]:
        return _unique(value, "ID list")


# --------------------------------------------------------------------------
# 5. ToolResult
# --------------------------------------------------------------------------

class ToolError(StrictRecord):
    """A tool failure with a fixed code, a message, and optional details."""

    code: ToolErrorCode
    message: NonBlank
    details: dict[str, Any] | None = None

    @field_validator("details")
    @classmethod
    def _details_finite(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is not None:
            _reject_nonfinite(value, "error.details")
        return value


class ToolResult(StrictRecord):
    """What every MCP/analyst tool returns: ok with data, or not ok with a coded error.

    Everyday example: a waiter comes back either with the dish, or with a
    specific reason ("kitchen is out of cassava") — never a shrug.
    """

    ok: bool
    data: dict[str, Any] | list[Any] | None = None
    error: ToolError | None = None
    request_ids: list[IdStr] = Field(default_factory=list)
    artifact_ids: list[ArtifactId] = Field(default_factory=list)
    complete: bool | None = None       # None = completeness does not apply to this tool
    warnings: list[str] = Field(default_factory=list)

    @field_validator("data")
    @classmethod
    def _data_finite(cls, value: Any) -> Any:
        if value is not None:
            _reject_nonfinite(value, "data")
        return value

    @model_validator(mode="after")
    def _ok_xor_error(self) -> "ToolResult":
        if self.ok and self.error is not None:
            raise ValueError("ok=True cannot carry an error")
        if not self.ok and self.error is None:
            raise ValueError("ok=False requires a structured error (code + message)")
        return self


# --------------------------------------------------------------------------
# 6. FetchApproval
# --------------------------------------------------------------------------

class FetchApproval(StrictRecord):
    """A human-issued permission slip for live BrAPI access, created ONLY by controller code.

    A model cannot create one: model output is parsed into AgentResult,
    whose payload validator rejects authority keys, and the dispatcher
    accepts only FetchApproval objects built in-process by the controller
    (enforced in Stages 3 and 5). The record is frozen; a changed scope
    means a new approval.

    Everyday example: a signed hall pass that lists exactly which rooms
    (endpoint families), which lockers (study IDs), how many trips
    (max_http_attempts) and until what time (expires_at_utc).
    """

    approval_id: ApprovalId
    run_id: IdStr
    base_url: BaseUrl
    endpoint_families: list[EndpointFamily] = Field(min_length=1)
    observation_study_ids: list[IdStr] = Field(default_factory=list)
    observation_variable_ids: list[IdStr] = Field(default_factory=list)
    max_http_attempts: PositiveInt
    max_observation_studies: Count
    issued_at_utc: UtcDatetime
    expires_at_utc: UtcDatetime
    issued_by: Literal["human_controller"] = "human_controller"

    @field_validator("endpoint_families")
    @classmethod
    def _families_unique(cls, value: list[str]) -> list[str]:
        return _unique(value, "endpoint_families")

    @field_validator("observation_study_ids", "observation_variable_ids")
    @classmethod
    def _ids_unique(cls, value: list[str]) -> list[str]:
        return _unique(value, "ID list")

    @model_validator(mode="after")
    def _consistent(self) -> "FetchApproval":
        if self.expires_at_utc <= self.issued_at_utc:
            raise ValueError("expires_at_utc must be after issued_at_utc")
        if len(self.observation_study_ids) > self.max_observation_studies:
            raise ValueError("more observation studies listed than max_observation_studies allows")
        wants_observations = any(f in ("observations", "observationunits") for f in self.endpoint_families)
        if wants_observations and not self.observation_study_ids:
            raise ValueError("observation access requires exact observation_study_ids")
        if self.observation_study_ids and not wants_observations:
            raise ValueError("observation_study_ids given without an observation endpoint family")
        if "observations" in self.endpoint_families and not self.observation_variable_ids:
            raise ValueError("observations access requires exact observation_variable_ids")
        return self

    def is_valid_at(self, now: datetime) -> bool:
        return self.issued_at_utc <= _require_utc(now) < self.expires_at_utc

    def allows_family(self, family: str) -> bool:
        return family in self.endpoint_families

    def allows_study(self, study_id: str) -> bool:
        return study_id in self.observation_study_ids

    def allows_variable(self, variable_id: str) -> bool:
        return variable_id in self.observation_variable_ids


# --------------------------------------------------------------------------
# 7. Claim
# --------------------------------------------------------------------------

class Denominator(StrictRecord):
    """What 'n' means: a name plus the number. 'n=4' alone is not a denominator."""

    name: NonBlank          # e.g. "valid plot-level values"
    n: Count


class Exclusion(StrictRecord):
    """Something left out of a calculation and how many of them."""

    reason: NonBlank
    count: Count


class Claim(StrictRecord):
    """One numeric statement with its evidence, so a reviewer can check it.

    Everyday example: a nutrition label — "12 g sugar per 100 g": the value
    (12), the unit (g), the denominator (per 100 g), and, on a good label,
    how it was measured and what was left out.
    """

    claim_id: ClaimId
    kind: ClaimKind
    value: FiniteFloat | int | None
    unit: str | None = None
    missing_reason: str | None = None
    denominator: Denominator
    n_independent_units: Count | None = None      # None = independence not established
    source_artifact_hashes: list[Sha256] = Field(min_length=1)
    filters: dict[str, str] = Field(default_factory=dict)
    grouping: list[str] = Field(default_factory=list)
    transformation: NonBlank                       # e.g. "pooled mean of valid values"
    exclusions: list[Exclusion] = Field(default_factory=list)

    @field_validator("source_artifact_hashes")
    @classmethod
    def _hashes_unique(cls, value: list[str]) -> list[str]:
        return _unique(value, "source_artifact_hashes")

    @model_validator(mode="after")
    def _missing_has_reason(self) -> "Claim":
        if self.value is None and not (self.missing_reason and self.missing_reason.strip()):
            raise ValueError("a missing value must carry a non-empty missing_reason")
        if self.value is not None and self.missing_reason is not None:
            raise ValueError("missing_reason is only allowed when value is null")
        if self.kind == "count" and self.unit is not None:
            raise ValueError("a count has no unit")
        return self


# --------------------------------------------------------------------------
# 8. AgentResult
# --------------------------------------------------------------------------

class NormalizedMessage(StrictRecord):
    """One conversation event in a plain, serializable shape (no SDK objects)."""

    role: MessageRole
    content: str
    name: str | None = None                 # tool name for tool requests / results
    tool_call_id: str | None = None
    correlation_id: str | None = None


class Usage(StrictRecord):
    """Token counts as the provider reported them. Absent usage is None, never zeros."""

    prompt_tokens: Count
    completion_tokens: Count
    total_tokens: Count

    @model_validator(mode="after")
    def _adds_up(self) -> "Usage":
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("total_tokens != prompt_tokens + completion_tokens")
        return self


class StructuredError(StrictRecord):
    code: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]
    message: NonBlank
    stage: str | None = None


class AgentResult(StrictRecord):
    """How one agent run ended, with everything needed to audit it.

    Everyday example: a report card plus the attendance log — final grade
    (status), the work handed in (payload), every session (messages), which
    teacher was asked for vs who actually showed up (model requested vs
    reported), and how many questions were asked (call counts).
    """

    status: TerminalStatus
    payload: dict[str, Any] | None = None
    messages: list[NormalizedMessage] = Field(default_factory=list)
    model_requested: NonBlank
    model_reported: str | None = None
    model_calls: Count
    tool_calls: Count
    usage: Usage | None = None
    elapsed_seconds: Seconds
    log_path: ManagedPath | None = None
    errors: list[StructuredError] = Field(default_factory=list)

    @field_validator("payload")
    @classmethod
    def _payload_clean(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is not None:
            _reject_authority(value, "payload")
            _reject_nonfinite(value, "payload")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> "AgentResult":
        if self.status == "completed" and self.payload is None:
            raise ValueError("status completed requires a payload")
        if self.status == "failed" and not self.errors:
            raise ValueError("status failed requires at least one structured error")
        return self


# --------------------------------------------------------------------------
# 9. RetrievalReport
# --------------------------------------------------------------------------

class ResolvedEntity(StrictRecord):
    """A human phrase turned into one exact ID, with the evidence that resolved it."""

    kind: EntityKind
    id: IdStr
    label: NonBlank
    source_request_ids: list[IdStr] = Field(default_factory=list)


class StudyStatusEntry(StrictRecord):
    """One row of the per-study ledger. A zero-row or failed study stays on the ledger."""

    study_id: IdStr
    status: StudyFetchStatus
    artifact_id: ArtifactId | None = None
    reason: str | None = None
    request_ids: list[IdStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> "StudyStatusEntry":
        if self.status in ("complete", "empty") and self.artifact_id is None:
            raise ValueError("a complete/empty study must point at its artifact")
        if self.status not in ("complete", "empty") and not (self.reason and self.reason.strip()):
            raise ValueError(f"status {self.status!r} needs a reason")
        return self


class RetrievalReport(StrictRecord):
    """What the Retriever hands to the Analyst: resolved IDs, artifacts, and a per-study ledger.

    Everyday example: a librarian's fetch slip — which titles were found
    (resolved), which boxes were brought (artifacts), and for every title
    requested: found / empty shelf / refused / still on order (ledger).
    """

    status: TerminalStatus
    resolved: list[ResolvedEntity] = Field(default_factory=list)
    artifacts: list[ArtifactManifest] = Field(default_factory=list)
    study_ledger: list[StudyStatusEntry] = Field(default_factory=list)
    complete: bool
    request_ids: list[IdStr] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    log_refs: list[str] = Field(default_factory=list)
    usage: Usage | None = None

    @model_validator(mode="after")
    def _consistent(self) -> "RetrievalReport":
        _unique([e.study_id for e in self.study_ledger], "study_ledger study_ids")
        artifact_ids = _unique([a.artifact_id for a in self.artifacts], "artifact_ids")
        for entry in self.study_ledger:
            if entry.artifact_id is not None and entry.artifact_id not in artifact_ids:
                raise ValueError(f"ledger refers to unknown artifact {entry.artifact_id!r}")
        all_good = all(e.status in ("complete", "empty") for e in self.study_ledger)
        if self.complete and not all_good:
            raise ValueError("complete=True but a study on the ledger is not complete/empty")
        if self.complete and self.status != "completed":
            raise ValueError("complete=True requires status completed")
        if self.status == "completed" and not self.complete:
            raise ValueError("status completed requires complete=True; otherwise use incomplete")
        return self


# --------------------------------------------------------------------------
# 10. AnalysisReport
# --------------------------------------------------------------------------

class MetadataFact(StrictRecord):
    """A recorded text field, read by code from an exact entity row in a hashed artifact."""

    entity_kind: Literal["study", "variable"]
    entity_id: IdStr
    entity_label: str | None = None
    field: NonBlank
    value: str | None = None
    missing_reason: NonBlank | None = None
    source_artifact_id: ArtifactId
    source_artifact_hash: Sha256
    source_request_ids: list[IdStr] = Field(default_factory=list)
    source_fetched_at_utc: list[UtcDatetime] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> "MetadataFact":
        if self.value is None and self.missing_reason is None:
            raise ValueError("missing metadata requires a reason")
        if self.value is not None and self.missing_reason is not None:
            raise ValueError("recorded metadata cannot also carry a missing reason")
        if self.value == "":
            raise ValueError("an empty metadata cell must use value=None and a missing reason")
        _unique(self.source_request_ids, "metadata source request ids")
        return self


class AnalysisReport(StrictRecord):
    """What the Analyst hands back: typed claims, methods, exclusions and caveats.

    Everyday example: a lab report — the measured numbers (claims), the
    procedure (methods), the samples thrown out and why (exclusions), and
    the warnings a careful scientist adds (caveats).
    """

    status: TerminalStatus
    claims: list[Claim] = Field(default_factory=list)
    metadata_facts: list[MetadataFact] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    exclusions: list[Exclusion] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    result_artifact_ids: list[ArtifactId] = Field(default_factory=list)
    log_refs: list[str] = Field(default_factory=list)

    @field_validator("result_artifact_ids")
    @classmethod
    def _artifact_ids_unique(cls, value: list[str]) -> list[str]:
        return _unique(value, "result_artifact_ids")

    @model_validator(mode="after")
    def _consistent(self) -> "AnalysisReport":
        _unique([c.claim_id for c in self.claims], "claim_ids")
        _unique([(f.entity_kind, f.entity_id, f.field) for f in self.metadata_facts], "metadata entity fields")
        if self.status == "completed" and not (self.claims or self.metadata_facts):
            raise ValueError("status completed requires at least one claim or source-backed metadata fact")
        return self


# --------------------------------------------------------------------------
# 11. Plan (Stage 5): what a run intends to do, written down BEFORE any authority exists
# --------------------------------------------------------------------------

MAX_PLAN_STEPS = 12
PlanInputValue = str | int | bool | list[str]       # explicit typed inputs; never a blob of JSON text
_STEP_ID = Annotated[str, Field(pattern=r"^step_[A-Za-z0-9_-]+$")]
_PLAN_ID = Annotated[str, Field(pattern=r"^plan_[A-Za-z0-9_-]+$")]


def _reject_json_blob(value: Any, where: str) -> None:
    """A string that merely parses as a JSON object or list is not a typed input; it is smuggled structure."""
    for _, item in _walk(value):
        if isinstance(item, str):
            text = item.strip()
            if text[:1] in ("{", "[") and text[-1:] in ("}", "]"):
                try:
                    json.loads(text)
                except ValueError:
                    continue
                raise ValueError(f"{where} contains a JSON-looking string {text[:40]!r}; pass typed fields, not encoded JSON")


class Clarification(StrictRecord):
    """One question the run must have answered before it is ready. required=False is advisory only."""

    question: NonBlank
    answer: NonBlank | None = None
    required: bool = True


class ResolvedScope(StrictRecord):
    """The exact IDs a run will touch. Unknown is written as unknown, never estimated.

    Everyday example: a shopping list with aisle numbers. "Aisle 4, item 0012" is
    resolved; "something in the fruit section, maybe 6 items" is discovery_needed
    with matching_study_count=None — you do not write "6" until you have counted.
    """

    base_url: BaseUrl
    study_ids: list[IdStr] = Field(default_factory=list)
    variable_ids: list[IdStr] = Field(default_factory=list)
    location_ids: list[IdStr] = Field(default_factory=list)
    season_ids: list[IdStr] = Field(default_factory=list)
    filters: StudyFilters = Field(default_factory=StudyFilters)
    discovery_needed: bool = False
    discovery_families: list[EndpointFamily] = Field(default_factory=list)   # metadata families discovery may touch
    matching_study_count: Count | None = None                                # None = unknown until discovery ran

    @field_validator("study_ids", "variable_ids", "location_ids", "season_ids", "discovery_families")
    @classmethod
    def _scope_unique(cls, value: list[str]) -> list[str]:
        return _unique(value, "scope list")

    @model_validator(mode="after")
    def _consistent(self) -> "ResolvedScope":
        if self.discovery_needed and not self.discovery_families:
            raise ValueError("discovery_needed requires the endpoint families discovery may use")
        if not self.discovery_needed and self.discovery_families:
            raise ValueError("discovery_families given although discovery_needed is False")
        if any(f in ("observations", "observationunits") for f in self.discovery_families):
            raise ValueError("discovery is metadata only; observations need an approved fetch manifest")
        if self.matching_study_count is not None and self.discovery_needed:
            raise ValueError("matching_study_count cannot be known while discovery is still needed")
        return self


class Statistic(StrictRecord):
    """What number the run will compute and what its n means. A denominator is a NAME, not just a number."""

    kind: ClaimKind
    denominator: NonBlank                       # e.g. "valid plot-level values", "rows in the complete studies table"
    weighting: Literal["none", "equal_per_unit"] = "none"
    grouping: list[NonBlank] = Field(default_factory=list)
    min_independent_n: PositiveInt = 2
    duplicate_policy: Literal["reject", "mean_per_unit"] = "reject"


class PlanStep(StrictRecord):
    """One bounded action by one allowed agent, with explicit typed inputs and earlier-step dependencies."""

    step_id: _STEP_ID
    agent: PlanAgent
    action: NonBlank
    inputs: dict[NonBlank, PlanInputValue] = Field(default_factory=dict)
    depends_on: list[_STEP_ID] = Field(default_factory=list)
    max_tool_calls: Annotated[int, Field(ge=1, le=30)] = 10

    @field_validator("inputs")
    @classmethod
    def _inputs_clean(cls, value: dict[str, Any]) -> dict[str, Any]:
        _reject_authority(value, "step inputs")
        _reject_json_blob(value, "step inputs")
        _reject_nonfinite(value, "step inputs")
        return value

    @field_validator("depends_on")
    @classmethod
    def _deps_unique(cls, value: list[str]) -> list[str]:
        return _unique(value, "depends_on")


class Plan(StrictRecord):
    """The run's intention, validated before anything is fetched or computed.

    Steps are ordered; a step may depend only on EARLIER steps, which makes
    self-references, forward references and cycles impossible by construction.
    An analyst step must trace back to at least one retriever step — directly,
    or through the earlier analyst steps it depends on (export → filter → count
    is one chain of evidence; a count that depends on nothing analyses thin
    air). A plan carries no authority: no approval, no budget grant — those
    come from controller code and a FetchManifest.

    Everyday example: a recipe card written before shopping. It lists the
    dish, the servings, the steps in order, and what each step needs from an
    earlier one. It is not a receipt and it does not open the shop.
    """

    plan_schema: Literal["plan.v1"] = "plan.v1"
    plan_id: _PLAN_ID
    question: NonBlank
    interpretation: NonBlank
    clarifications: list[Clarification] = Field(default_factory=list)
    status: PlanStatus
    scope: ResolvedScope
    statistic: Statistic | None = None
    steps: list[PlanStep] = Field(min_length=1, max_length=MAX_PLAN_STEPS)
    created_at_utc: UtcDatetime

    @model_validator(mode="after")
    def _consistent(self) -> "Plan":
        ids = [s.step_id for s in self.steps]
        _unique(ids, "step_ids")
        seen: set[str] = set()
        evidence: dict[str, set[str]] = {}          # step_id -> the retriever steps it traces back to, directly or through analyst steps
        for step in self.steps:
            for dep in step.depends_on:
                if dep == step.step_id:
                    raise ValueError(f"{step.step_id} depends on itself")
                if dep not in seen:
                    where = "a later step" if dep in ids else "an unknown step"
                    raise ValueError(f"{step.step_id} depends on {dep!r}, which is {where}; dependencies must be earlier steps")
            traced: set[str] = set()
            for dep in step.depends_on:
                traced |= evidence[dep]
            if step.agent == "retriever":
                traced.add(step.step_id)
            elif not traced:
                raise ValueError(f"analyst step {step.step_id} must depend on at least one retriever step, directly or through earlier analyst steps")
            evidence[step.step_id] = traced
            seen.add(step.step_id)
        unanswered = [c.question for c in self.clarifications if c.required and c.answer is None]
        if unanswered and self.status != "needs_clarification":
            raise ValueError(f"required clarifications are unanswered; status must be needs_clarification: {unanswered[:2]}")
        if self.status == "ready":
            if self.scope.discovery_needed and not any(s.agent == "retriever" for s in self.steps):
                raise ValueError("a ready plan that needs discovery must include a retriever step")
        _reject_json_blob([self.question, self.interpretation, *(c.answer or "" for c in self.clarifications)], "plan text")
        return self


# --------------------------------------------------------------------------
# 12. FetchManifest and RunResult (Stage 5)
# --------------------------------------------------------------------------

class FetchManifest(StrictRecord):
    """The EXACT retrieval a human is asked to approve, hashed so the approval binds to it.

    A FetchApproval is issued for one manifest hash. Change one study ID, one
    variable, one budget number — the hash changes and the old approval no
    longer matches (see approval_mismatches). Snapshot manifests need no live
    approval because nothing leaves the machine.

    Everyday example: a purchase order. The signature is for THIS order; add
    one line item and you need a new signature.
    """

    manifest_schema: Literal["fetch_manifest.v1"] = "fetch_manifest.v1"
    run_id: IdStr
    base_url: BaseUrl
    origin: FetchOrigin
    snapshot_id: IdStr | None = None
    endpoint_families: list[EndpointFamily] = Field(min_length=1)
    observation_study_ids: list[IdStr] = Field(default_factory=list)
    observation_variable_ids: list[IdStr] = Field(default_factory=list)
    max_http_attempts: PositiveInt
    max_observation_studies: Count

    @field_validator("endpoint_families", "observation_study_ids", "observation_variable_ids")
    @classmethod
    def _manifest_unique(cls, value: list[str]) -> list[str]:
        return _unique(value, "manifest list")

    @model_validator(mode="after")
    def _consistent(self) -> "FetchManifest":
        if self.origin == "snapshot" and not self.snapshot_id:
            raise ValueError("a snapshot manifest must name the snapshot_id")
        if self.origin == "live" and self.snapshot_id is not None:
            raise ValueError("a live manifest cannot name a snapshot_id")
        if len(self.observation_study_ids) > self.max_observation_studies:
            raise ValueError("more observation studies listed than max_observation_studies allows")
        wants_observations = any(f in ("observations", "observationunits") for f in self.endpoint_families)
        if wants_observations and not self.observation_study_ids:
            raise ValueError("observation access requires exact observation_study_ids")
        if "observations" in self.endpoint_families and not self.observation_variable_ids:
            raise ValueError("observations access requires exact observation_variable_ids")
        if self.observation_study_ids and not wants_observations:
            raise ValueError("observation_study_ids given without an observation endpoint family")
        return self

    def sha256(self) -> str:
        """The fingerprint an approval binds to: canonical JSON of every field."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def approval_mismatches(manifest: FetchManifest, approval: FetchApproval) -> list[str]:
    """Every way the approval fails to cover exactly this manifest. Empty list = it matches.

    Exact equality on purpose: a wider approval (more studies, more attempts) is
    also a mismatch, because the human approved a different fetch.
    """
    problems: list[str] = []
    if manifest.origin != "live":
        problems.append(f"manifest origin is {manifest.origin!r}; a live approval does not apply to it")
    if approval.run_id != manifest.run_id:
        problems.append(f"run_id differs: approval {approval.run_id!r} vs manifest {manifest.run_id!r}")
    if approval.base_url != manifest.base_url:
        problems.append(f"base_url differs: approval {approval.base_url!r} vs manifest {manifest.base_url!r}")
    if sorted(approval.endpoint_families) != sorted(manifest.endpoint_families):
        problems.append(f"endpoint_families differ: approval {sorted(approval.endpoint_families)} vs manifest {sorted(manifest.endpoint_families)}")
    if sorted(approval.observation_study_ids) != sorted(manifest.observation_study_ids):
        problems.append(f"observation_study_ids differ: approval {sorted(approval.observation_study_ids)} vs manifest {sorted(manifest.observation_study_ids)}")
    if sorted(approval.observation_variable_ids) != sorted(manifest.observation_variable_ids):
        problems.append(f"observation_variable_ids differ: approval {sorted(approval.observation_variable_ids)} vs manifest {sorted(manifest.observation_variable_ids)}")
    if approval.max_http_attempts != manifest.max_http_attempts:
        problems.append(f"max_http_attempts differ: approval {approval.max_http_attempts} vs manifest {manifest.max_http_attempts}")
    if approval.max_observation_studies != manifest.max_observation_studies:
        problems.append(f"max_observation_studies differ: approval {approval.max_observation_studies} vs manifest {manifest.max_observation_studies}")
    return problems


class RunCounts(StrictRecord):
    """What actually happened, counted by code: real attempts and distinct studies, never a model's estimate."""

    model_calls: Count = 0
    tool_calls: Count = 0
    live_attempts: Count = 0
    cached_requests: Count = 0
    distinct_studies_fetched: Count = 0
    elapsed_seconds: Seconds = 0.0


class RunResult(StrictRecord):
    """Everything one run produced, linked: plan -> manifest -> approval -> retrieval -> analysis -> artifacts.

    execution_status says how far the SOFTWARE got; human_acceptance says what the
    HUMAN decided about the draft. Both are needed: a run can complete with
    acceptance still pending (evaluation runs do exactly that), and a human may
    reject a run that completed.

    Everyday example: a contractor's job file — the quote (plan), the signed
    work order (manifest + approval), the invoices (retrieval and analysis),
    the photos (artifacts), and two separate boxes: "work finished" and
    "customer signed off".
    """

    run_schema: Literal["run_result.v1"] = "run_result.v1"
    run_id: IdStr
    plan: Plan
    manifest: FetchManifest | None = None
    approval_id: ApprovalId | None = None
    approved_manifest_sha256: Sha256 | None = None
    catalog_manifest: FetchManifest | None = None
    catalog_approval: FetchApproval | None = None
    approved_catalog_manifest_sha256: Sha256 | None = None
    catalog_requests: list[RequestRecord] = Field(default_factory=list)
    retrieval: RetrievalReport | None = None
    analysis: AnalysisReport | None = None
    artifacts: list[ArtifactManifest] = Field(default_factory=list)
    execution_status: ExecutionStatus
    human_acceptance: Acceptance = "pending"
    counts: RunCounts = Field(default_factory=RunCounts)
    model_requested: NonBlank | None = None
    model_reported: str | None = None
    started_at_utc: UtcDatetime
    finished_at_utc: UtcDatetime | None = None
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> "RunResult":
        if self.manifest is not None and self.manifest.run_id != self.run_id:
            raise ValueError("manifest.run_id differs from run_id")
        if (self.approval_id is None) != (self.approved_manifest_sha256 is None):
            raise ValueError("approval_id and approved_manifest_sha256 must be given together")
        if self.approved_manifest_sha256 is not None:
            if self.manifest is None:
                raise ValueError("an approval is recorded but there is no manifest")
            if self.manifest.sha256() != self.approved_manifest_sha256:
                raise ValueError("the manifest changed after approval: its hash differs from approved_manifest_sha256")
        # Catalog setup is a separate metadata-only permission. Its proven attempts stay
        # in the total, but can never stand in for approval to retrieve observations.
        catalog_live_attempts = 0
        if (self.catalog_approval is None) != (self.approved_catalog_manifest_sha256 is None):
            raise ValueError("catalog_approval and approved_catalog_manifest_sha256 must be given together")
        if self.catalog_manifest is None:
            if self.catalog_approval is not None or self.catalog_requests:
                raise ValueError("catalog approval or requests were recorded without a catalog manifest")
        else:
            catalog = self.catalog_manifest
            if catalog.run_id != self.run_id + "_catalog":
                raise ValueError("catalog manifest run_id does not belong to this run")
            if catalog.base_url != self.plan.scope.base_url or (self.manifest is not None and catalog.base_url != self.manifest.base_url):
                raise ValueError("catalog manifest server differs from the run's source server")
            if (catalog.origin != "live" or not set(catalog.endpoint_families) <= {"studies", "observationvariables"}
                    or catalog.observation_study_ids or catalog.observation_variable_ids or catalog.max_observation_studies != 0):
                raise ValueError("catalog manifest must grant metadata-only studies and variables, with no observation authority")
            if self.catalog_approval is not None:
                if catalog.sha256() != self.approved_catalog_manifest_sha256:
                    raise ValueError("catalog manifest changed after approval: its hash differs")
                mismatches = approval_mismatches(catalog, self.catalog_approval)
                if mismatches:
                    raise ValueError("catalog approval does not match its manifest: " + "; ".join(mismatches))
            for record in self.catalog_requests:
                if record.run_id != catalog.run_id or record.base_url != catalog.base_url:
                    raise ValueError("catalog request run_id or server differs from its manifest")
                family = {"/studies": "studies", "/variables": "observationvariables"}.get(record.endpoint)
                if family is None or family not in catalog.endpoint_families:
                    raise ValueError("catalog request is outside its metadata-only approved endpoint scope")
                if record.origin == "live":
                    if self.catalog_approval is None:
                        raise ValueError("catalog live attempts were counted but no catalog approval is recorded")
                    if not self.catalog_approval.is_valid_at(record.requested_at_utc):
                        raise ValueError("catalog request occurred outside its approval validity period")
                    catalog_live_attempts += 1
            _unique([record.request_id for record in self.catalog_requests], "catalog request ids")
            if catalog_live_attempts > catalog.max_http_attempts:
                raise ValueError("catalog live attempts exceed its approved HTTP attempt budget")
        if catalog_live_attempts > self.counts.live_attempts:
            raise ValueError("catalog live attempts exceed the run's total live attempts")
        if self.counts.live_attempts > catalog_live_attempts and self.approval_id is None:
            raise ValueError("live attempts were counted but no approval is recorded for non-catalog requests")
        if self.analysis is not None and self.retrieval is None:
            raise ValueError("an analysis without a retrieval has no evidence")
        artifact_ids = _unique([a.artifact_id for a in self.artifacts], "artifact ids")
        by_id = {a.artifact_id: a for a in self.artifacts}
        hashes = {a.sha256 for a in self.artifacts}
        if self.retrieval is not None:
            for a in self.retrieval.artifacts:
                if a.artifact_id not in by_id or by_id[a.artifact_id].sha256 != a.sha256:
                    raise ValueError(f"retrieval artifact {a.artifact_id} is missing from the run's artifacts or has a different hash")
        if self.analysis is not None:
            for rid in self.analysis.result_artifact_ids:
                if rid not in artifact_ids:
                    raise ValueError(f"analysis result artifact {rid} is not among the run's artifacts")
            for claim in self.analysis.claims:
                unsupported = [h for h in claim.source_artifact_hashes if h not in hashes]
                if unsupported:
                    raise ValueError(f"claim {claim.claim_id} cites artifact hashes that no run artifact has: {unsupported[0][:12]}...")
            for fact in self.analysis.metadata_facts:
                artifact = by_id.get(fact.source_artifact_id)
                if artifact is None or artifact.sha256 != fact.source_artifact_hash:
                    raise ValueError("metadata fact cites an unknown artifact or a different hash")
                if not artifact.complete or artifact.kind != {"study": "studies", "variable": "variables"}[fact.entity_kind]:
                    raise ValueError("metadata fact requires a complete artifact of the matching entity kind")
                if fact.field not in artifact.columns:
                    raise ValueError("metadata fact field is absent from the source artifact schema")
                if fact.source_request_ids != artifact.source_request_ids:
                    raise ValueError("metadata fact request provenance differs from its source artifact")
                scoped_ids = artifact.study_ids if fact.entity_kind == "study" else artifact.variable_ids
                if scoped_ids and fact.entity_id not in scoped_ids:
                    raise ValueError("metadata fact entity is outside its source artifact scope")
                if self.retrieval is None or fact.source_artifact_id not in {a.artifact_id for a in self.retrieval.artifacts}:
                    raise ValueError("metadata fact source was not supplied by this run's retrieval")
        if self.execution_status == "completed":
            if self.retrieval is None or self.retrieval.status != "completed":
                raise ValueError("execution_status completed requires a completed retrieval")
            if self.analysis is None or self.analysis.status != "completed":
                raise ValueError("execution_status completed requires a completed analysis")
        if self.human_acceptance == "accepted" and self.execution_status != "completed":
            raise ValueError("a human can only accept a run whose execution completed")
        if self.finished_at_utc is not None and self.finished_at_utc < self.started_at_utc:
            raise ValueError("finished_at_utc is before started_at_utc")
        return self


# --------------------------------------------------------------------------
# Serialization helper
# --------------------------------------------------------------------------

def dump_json(record: StrictRecord) -> str:
    """Serialize a record to JSON, refusing if a NaN/Infinity float somehow got through."""
    try:
        _reject_nonfinite(record.model_dump(), type(record).__name__)
    except ValueError as exc:
        raise ContractError(str(exc)) from exc
    return record.model_dump_json()
