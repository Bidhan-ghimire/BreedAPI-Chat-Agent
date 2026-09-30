"""
brapi_client.py — the ONLY module that talks to a BrAPI server.

Plain-words summary:
* One transport function does every HTTP call. Everything else in part2
  goes through this client; nothing else imports requests to reach a server.
* GET only, one approved base URL, allowlisted BrAPI paths. A model can never
  hand us a full URL or a path with ".." in it.
* Offline is the default: a cache miss in offline mode is an error, not a
  reason to go online.
* Live access needs a FetchApproval (a permission slip built by controller
  code), checked before EVERY attempt, retries included.
* Every attempt — success, failure, cache hit — gets a RequestRecord.
* A collection is only called complete when the server's own counts prove it.

Typed return values (what you get back):
* get()               -> (parsed JSON dict, RequestRecord)
* study()             -> (one study record dict, RequestRecord)
* get_all(), studies(), variables(), find_variables(), observations(),
  observation_units(), locations(), programs(), seasons()
                      -> CollectionResult (records + returned/reported counts +
                         complete flag + status + request_ids + warnings)
* records_to_df()     -> pandas DataFrame where EVERY cell is text (str) or None;
                         numbers are never converted here, so raw tokens survive.
Errors are raised as BrapiClientError subclasses with a fixed .code.
"""
from __future__ import annotations

import json
import os
import re
import math
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Literal

from cache_store import CacheEntry, CacheIdentity, CacheStore, sha256_bytes
from contracts import (
    CollectionResult,
    CollectionStatus,
    FetchApproval,
    RequestError,
    RequestRecord,
    StudyFilters,
)

__all__ = [
    "PART2_DIR",
    "Settings",
    "load_settings",
    "BrapiClientError",
    "NotAuthorized",
    "OfflineCacheMiss",
    "TransportError",
    "TransportRequest",
    "TransportResponse",
    "RequestsTransport",
    "BrapiClient",
    "records_to_df",
    "ID_FIELDS",
]

PART2_DIR = Path(__file__).resolve().parent

Mode = Literal["offline", "live"]


# --------------------------------------------------------------------------
# Settings — loaded relative to THIS file, never the caller's working directory
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    base_url: str
    mode: Mode = "offline"
    snapshot_id: str | None = None
    cache_dir: Path = PART2_DIR / "cache"
    connect_timeout: float = 10.0        # seconds to establish a connection
    read_timeout: float = 30.0           # seconds of silence tolerated during ONE read
    max_bytes: int = 10_000_000          # response byte cap
    max_attempts: int = 3                # per metadata request (observations: 1)
    operation_deadline: float = 120.0    # whole-operation budget, seconds
    page_size: int = 1000                # requested page size; the server may cap it


def load_settings(env_file: Path | None = None, environ: Mapping[str, str] | None = None) -> Settings:
    """Read part2/.env (values never printed), then let real environment variables win.

    No token variable is read: this is the public-only version.
    """
    from dotenv import dotenv_values

    # Offline tests and hosted deployments may explicitly ignore the local secrets file.
    # Explicit fixture paths still work for configuration-parser tests.
    environment = dict(os.environ if environ is None else environ)
    ignore_default = env_file is None and environment.get("PART2_IGNORE_DOTENV", "").strip().lower() == "true"
    env_file = PART2_DIR / ".env" if env_file is None else env_file
    values: dict[str, str] = {}
    if not ignore_default and env_file.is_file():
        values.update({k: v for k, v in dotenv_values(env_file).items() if v is not None})
    values.update(environment)

    base_url = values.get("BRAPI_BASE_URL", "https://cassavabase.org/brapi/v2").strip().rstrip("/")
    mode = values.get("BRAPI_MODE", "offline").strip().lower()
    if mode not in ("offline", "live"):
        raise BrapiClientError("invalid_argument", f"BRAPI_MODE must be offline or live, not {mode!r}")
    snapshot = values.get("BRAPI_SNAPSHOT_ID", "").strip() or None
    cache_dir = Path(values["BRAPI_CACHE_DIR"]) if values.get("BRAPI_CACHE_DIR") else PART2_DIR / "cache"
    return Settings(base_url=base_url, mode=mode, snapshot_id=snapshot, cache_dir=cache_dir)


# --------------------------------------------------------------------------
# Errors and the transport boundary
# --------------------------------------------------------------------------

class BrapiClientError(Exception):
    """Every client failure carries a fixed code plus a plain message."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class NotAuthorized(BrapiClientError):
    def __init__(self, message: str) -> None:
        super().__init__("not_authorized", message)


class OfflineCacheMiss(BrapiClientError):
    def __init__(self, message: str) -> None:
        super().__init__("offline_cache_miss", message)


class TransportError(Exception):
    """Raised by a transport. code is a RequestError code: connection, timeout, too_large, ..."""

    def __init__(self, code: str, message: str, retryable: bool = False) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.retryable = retryable


@dataclass(frozen=True)
class TransportRequest:
    url: str
    params: dict[str, str | int | bool]
    connect_timeout: float
    read_timeout: float
    max_bytes: int
    deadline_remaining: float
    cancelled: Callable[[], bool] | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class TransportResponse:
    status: int
    headers: dict[str, str]
    body: bytes


Transport = Callable[[TransportRequest], TransportResponse]


class RequestsTransport:
    """The real HTTP transport. Streams the body so the byte cap and deadline are enforceable.

    NOT exercised by the offline tests; first used by the approved live probe (Task 1D).
    """

    CHUNK = 64 * 1024

    def __init__(self) -> None:
        import requests

        self._requests = requests
        self._session = requests.Session()
        self._session.trust_env = False   # no proxies or .netrc credentials picked up silently

    def __call__(self, req: TransportRequest) -> TransportResponse:
        rq = self._requests
        if req.cancelled is not None and req.cancelled():
            raise TransportError("cancelled", "request cancelled before HTTP")
        started = time.monotonic()
        deadline_at = started + req.deadline_remaining
        try:
            response = self._session.get(
                req.url, params=req.params, allow_redirects=False, stream=True,
                timeout=(req.connect_timeout, req.read_timeout),
                headers={"Accept": "application/json", "User-Agent": "part2-brapi-client/1"},
            )
        except rq.exceptions.Timeout as exc:
            raise TransportError("timeout", f"no reply within {req.read_timeout:.0f}s", retryable=True) from exc
        except rq.exceptions.ConnectionError as exc:
            raise TransportError("connection", "could not reach the server", retryable=True) from exc
        except rq.exceptions.RequestException as exc:
            raise TransportError("unknown", type(exc).__name__) from exc
        try:
            chunks: list[bytes] = []
            size = 0
            try:
                for chunk in response.iter_content(chunk_size=self.CHUNK):
                    if req.cancelled is not None and req.cancelled():
                        raise TransportError("cancelled", "request cancelled while reading")
                    size += len(chunk)
                    if size > req.max_bytes:
                        raise TransportError("too_large", f"response exceeds {req.max_bytes} bytes")
                    chunks.append(chunk)
                    if time.monotonic() > deadline_at:
                        raise TransportError("timeout", "operation deadline passed while reading")
            except rq.exceptions.Timeout as exc:
                raise TransportError("timeout", "read stalled", retryable=True) from exc
            except rq.exceptions.ConnectionError as exc:
                raise TransportError("connection", "connection dropped while reading", retryable=True) from exc
            return TransportResponse(response.status_code, dict(response.headers), b"".join(chunks))
        finally:
            response.close()


# --------------------------------------------------------------------------
# Allowlist: the only paths this client will ever request
# --------------------------------------------------------------------------

_ID = r"[A-Za-z0-9_.:-]+"
_ALLOWED: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^/serverinfo$"), "serverinfo"),
    (re.compile(r"^/commoncropnames$"), "commoncropnames"),
    (re.compile(r"^/studies$"), "studies"),
    (re.compile(rf"^/studies/{_ID}$"), "studies"),
    (re.compile(r"^/variables$"), "observationvariables"),
    (re.compile(rf"^/variables/{_ID}$"), "observationvariables"),
    (re.compile(r"^/observations$"), "observations"),
    (re.compile(r"^/observationunits$"), "observationunits"),
    (re.compile(r"^/locations$"), "locations"),
    (re.compile(r"^/programs$"), "programs"),
    (re.compile(r"^/seasons$"), "seasons"),
]
_ID_RE = re.compile(rf"^{_ID}$")

# The stable ID field of a record, by endpoint. Used to detect repeated pages and duplicates.
ID_FIELDS: dict[str, str] = {
    "/studies": "studyDbId",
    "/variables": "observationVariableDbId",
    "/observations": "observationDbId",
    "/observationunits": "observationUnitDbId",
    "/locations": "locationDbId",
    "/programs": "programDbId",
    "/seasons": "seasonDbId",
}

_RESERVED_PARAMS = {"page", "pageSize"}


def _family_for(endpoint: str) -> str:
    if not isinstance(endpoint, str) or ".." in endpoint or "//" in endpoint or "?" in endpoint or "#" in endpoint:
        raise BrapiClientError("invalid_argument", f"endpoint {endpoint!r} is not a plain allowlisted path")
    for pattern, family in _ALLOWED:
        if pattern.match(endpoint):
            return family
    raise BrapiClientError("invalid_argument", f"endpoint {endpoint!r} is not on the allowlist")


def _check_id(value: str, label: str) -> str:
    if not isinstance(value, str) or not _ID_RE.match(value) or ".." in value or set(value) <= {"."}:
        raise BrapiClientError("invalid_argument", f"{label} must be a plain ID string, got {value!r}")
    return value


# --------------------------------------------------------------------------
# The client
# --------------------------------------------------------------------------

@dataclass
class _Page:
    data: list[dict[str, Any]]
    current_page: int | None
    page_size: int | None
    total_count: int | None
    total_pages: int | None
    result_null: bool


class BrapiClient:
    """See the module docstring. One instance = one run (one run_id, one request log)."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: Transport | None = None,
        cache: CacheStore | None = None,
        run_id: str | None = None,
        clock: Callable[[], float] = time.monotonic,
        now_utc: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        sleeper: Callable[[float], None] = time.sleep,
        remaining_time: Callable[[], float] | None = None,
    ) -> None:
        self.settings = settings
        self.cache = cache if cache is not None else CacheStore(settings.cache_dir)
        self._transport = transport
        self.run_id = run_id or f"run_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:6]}"
        self._clock = clock
        self._now_utc = now_utc
        self._sleep = sleeper
        self.remaining_time = remaining_time
        self._cancelled = threading.Event()
        self.request_log: list[RequestRecord] = []
        self._counter = 0
        self._live_attempts = 0                 # counts failed attempts too
        self._studies_fetched: set[str] = set()

    def cancel_pending_requests(self) -> None:
        """Permanently stop this run's later requests; an in-flight GET still finishes/logs.

        Safe to call from the event-loop thread while the server worker is busy.
        Create a new client for a new run; cancellation is never reset by reapproval.
        """
        self._cancelled.set()

    def _check_cancelled(self) -> None:
        if self._cancelled.is_set():
            raise BrapiClientError("cancelled", "this run's requests were cancelled")

    def _remaining(self, deadline_at: float) -> float:
        remaining = deadline_at - self._clock()
        if self.remaining_time is not None:
            supplied = float(self.remaining_time())
            if not math.isfinite(supplied):
                raise BrapiClientError("timeout", "invalid remaining work allowance")
            remaining = min(remaining, supplied)
        return remaining

    def _retry_wait(self, seconds: float) -> None:
        self._check_cancelled()
        if self._sleep is time.sleep:
            # Unlike a blocking sleep, the stop hook wakes a production retry wait.
            self._cancelled.wait(max(0.0, seconds))
        else:
            self._sleep(seconds)  # retain deterministic clock/sleeper injection in tests
        self._check_cancelled()

    # -- provenance ----------------------------------------------------------

    def provenance(self) -> list[RequestRecord]:
        """Every attempt and cache use so far, in order."""
        return list(self.request_log)

    def _new_request_id(self) -> str:
        self._counter += 1
        return f"req_{self.run_id}_{self._counter:04d}"

    # -- authorization: checked before EVERY live attempt --------------------

    def _authorize(self, approval: FetchApproval | None, family: str,
                   study_id: str | None = None, variable_id: str | None = None) -> FetchApproval:
        if self.settings.mode != "live":
            raise NotAuthorized("live access is off (BRAPI_MODE=offline)")
        if approval is None:
            raise NotAuthorized("no FetchApproval for this live request")
        if approval.base_url != self.settings.base_url:
            raise NotAuthorized("approval is for a different server")
        if not approval.is_valid_at(self._now_utc()):
            raise NotAuthorized("approval has expired or is not yet valid")
        if not approval.allows_family(family):
            raise NotAuthorized(f"approval does not cover endpoint family {family!r}")
        if self._live_attempts >= approval.max_http_attempts:
            raise NotAuthorized(f"HTTP attempt budget {approval.max_http_attempts} is used up")
        if family in ("observations", "observationunits"):
            if study_id is None or not approval.allows_study(study_id):
                raise NotAuthorized(f"study {study_id!r} is not in the approval")
            if family == "observations" and (variable_id is None or not approval.allows_variable(variable_id)):
                raise NotAuthorized(f"variable {variable_id!r} is not in the approval")
            if study_id not in self._studies_fetched and len(self._studies_fetched) >= approval.max_observation_studies:
                raise NotAuthorized(f"distinct observation studies limit {approval.max_observation_studies} reached")
        return approval

    # -- one GET -------------------------------------------------------------

    def get(
        self,
        endpoint: str,
        params: Mapping[str, str | int | bool] | None = None,
        *,
        approval: FetchApproval | None = None,
        expensive: bool = False,
        deadline_at: float | None = None,
        study_id: str | None = None,
        variable_id: str | None = None,
    ) -> tuple[dict[str, Any], RequestRecord]:
        """Cache first; then, only in live mode with a valid approval, bounded HTTP attempts.

        expensive=True (observations) means: never retry a timeout automatically.
        Returns the parsed JSON envelope and the RequestRecord of the successful attempt.
        Raises BrapiClientError with a code when nothing valid could be obtained.
        """
        self._check_cancelled()
        family = _family_for(endpoint)
        identity = CacheIdentity(base_url=self.settings.base_url, endpoint=endpoint, params=dict(params or {}))
        deadline_at = self._clock() + self.settings.operation_deadline if deadline_at is None else deadline_at

        # 1. cache
        hit = self.cache.get(identity)
        if hit is not None:
            entry, body = hit
            record = entry.request_record(request_id=self._new_request_id(), run_id=self.run_id,
                                          requested_at_utc=self._now_utc())
            self.request_log.append(record)
            return self._parse_envelope(body, record), record

        # 2. offline: stop here, no HTTP fallback
        if self.settings.mode != "live":
            raise OfflineCacheMiss(f"{endpoint} with {dict(identity.params)} is not cached and BRAPI_MODE=offline")

        # 3. live: bounded attempts, approval re-checked each time
        max_attempts = 1 if expensive else self.settings.max_attempts
        last_error: RequestError | None = None
        attempt = 0
        while attempt < max_attempts:
            self._check_cancelled()
            attempt += 1
            approval = self._authorize(approval, family, study_id, variable_id)
            remaining = self._remaining(deadline_at)
            if remaining <= 0:
                last_error = RequestError(code="timeout", message="operation deadline passed before the attempt")
                break
            record, body, retry_after = self._attempt(identity, attempt, remaining)
            self.request_log.append(record)
            if record.error is None:
                if study_id is not None and family in ("observations", "observationunits"):
                    self._studies_fetched.add(study_id)
                return self._parse_envelope(body, record), record
            last_error = record.error
            self._check_cancelled()
            if not record.error.retryable or attempt >= max_attempts:
                break
            remaining = self._remaining(deadline_at)
            if retry_after is not None:
                if retry_after > remaining:
                    last_error = RequestError(code="http_error", message=f"Retry-After {retry_after:.0f}s exceeds remaining budget")
                    break
                self._retry_wait(retry_after)          # the server asked us to wait; we wait, within budget
            else:
                self._retry_wait(min(0.5 * attempt, max(0.0, remaining)))   # short, bounded back-off
        assert last_error is not None
        raise BrapiClientError(last_error.code, f"{endpoint}: {last_error.message} (after {attempt} attempt(s))")

    def _attempt(self, identity: CacheIdentity, attempt: int, remaining: float) -> tuple[RequestRecord, bytes | None, float | None]:
        """One HTTP attempt. Always returns a RequestRecord; never raises for server-side problems."""
        self._check_cancelled()
        if self._transport is None:
            self._transport = RequestsTransport()
        self._live_attempts += 1
        url = f"{identity.base_url}{identity.endpoint}"
        redacted = _redacted_url(url, identity.params)
        requested_at = self._now_utc()
        started = self._clock()
        common = dict(request_id=self._new_request_id(), run_id=self.run_id, base_url=identity.base_url,
                      endpoint=identity.endpoint, params=dict(identity.params), requested_at_utc=requested_at,
                      origin="live", attempt=attempt, source_url_redacted=redacted)
        req = TransportRequest(url=url, params=dict(identity.params), connect_timeout=min(self.settings.connect_timeout, remaining),
                               read_timeout=min(self.settings.read_timeout, remaining), max_bytes=self.settings.max_bytes,
                               deadline_remaining=remaining, cancelled=self._cancelled.is_set)
        try:
            response = self._transport(req)
        except TransportError as exc:
            duration = max(0.0, self._clock() - started)
            err = RequestError(code=exc.code if exc.code in _REQUEST_CODES else "unknown", message=exc.message, retryable=exc.retryable)
            return RequestRecord(**common, duration_seconds=duration, error=err), None, None
        duration = max(0.0, self._clock() - started)
        fetched_at = self._now_utc()
        status = response.status
        if 300 <= status < 400:
            err = RequestError(code="http_error", message=f"HTTP {status} redirect refused (would leave the approved origin)")
            return RequestRecord(**common, duration_seconds=duration, http_status=status, error=err), None, None
        if status == 429 or 500 <= status < 600:
            retry_after = _parse_retry_after(response.headers) if status == 429 else None
            err = RequestError(code="http_error", message=f"HTTP {status}", retryable=True)
            return RequestRecord(**common, duration_seconds=duration, http_status=status, error=err), None, retry_after
        if not (200 <= status < 300):
            err = RequestError(code="http_error", message=f"HTTP {status}")
            return RequestRecord(**common, duration_seconds=duration, http_status=status, error=err), None, None
        body = response.body
        try:
            self._parse_envelope(body, None)
        except BrapiClientError as exc:
            err = RequestError(code=exc.code if exc.code in _REQUEST_CODES else "unknown", message=exc.message)
            return RequestRecord(**common, duration_seconds=duration, http_status=status, error=err), None, None
        entry: CacheEntry = self.cache.put(identity, body, http_status=status, fetched_at_utc=fetched_at,
                                           source_url_redacted=redacted,
                                           content_type=response.headers.get("Content-Type"))
        record = RequestRecord(**common, duration_seconds=duration, http_status=status,
                               fetched_at_utc=fetched_at, response_sha256=entry.response_sha256)
        return record, body, None

    @staticmethod
    def _parse_envelope(body: bytes, record: RequestRecord | None) -> dict[str, Any]:
        """A BrAPI reply is a JSON object with 'metadata' and 'result'. Anything else is not BrAPI."""
        try:
            parsed = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise BrapiClientError("invalid_json", "reply was not valid JSON")
        if not isinstance(parsed, dict) or "metadata" not in parsed or "result" not in parsed:
            raise BrapiClientError("not_brapi", "reply lacks the BrAPI metadata/result envelope")
        if parsed["result"] is not None and not isinstance(parsed["result"], dict):
            raise BrapiClientError("not_brapi", "result is neither an object nor null")
        return parsed

    # -- collections ----------------------------------------------------------

    def get_all(
        self,
        endpoint: str,
        params: Mapping[str, str | int | bool] | None = None,
        *,
        approval: FetchApproval | None = None,
        max_pages: int = 1,
        page_size: int | None = None,
        expensive: bool = False,
        study_id: str | None = None,
        variable_id: str | None = None,
    ) -> CollectionResult:
        """Fetch up to max_pages pages and say honestly whether the collection is complete."""
        params = dict(params or {})
        if _RESERVED_PARAMS & params.keys():
            raise BrapiClientError("invalid_argument", "page and pageSize are managed by the client; do not pass them")
        if max_pages < 1:
            raise BrapiClientError("invalid_argument", "max_pages must be at least 1")
        page_size = self.settings.page_size if page_size is None else page_size
        id_field = ID_FIELDS.get(endpoint)
        deadline_at = self._clock() + self.settings.operation_deadline

        records: list[dict[str, Any]] = []
        request_ids: list[str] = []
        warnings: list[str] = []
        seen_ids: dict[str, int] = {}
        previous_ids: list[str] | None = None
        reported_total: int | None = None
        status: CollectionStatus = "unknown"
        uncertain = False
        null_with_count = False
        pages_fetched = 0
        more_pages = False

        for page_number in range(max_pages):
            page_params = {**params, "page": page_number, "pageSize": page_size}
            log_before = len(self.request_log)
            try:
                envelope, record = self.get(endpoint, page_params, approval=approval, expensive=expensive,
                                            deadline_at=deadline_at, study_id=study_id, variable_id=variable_id)
            except BrapiClientError as exc:
                warnings.append(f"page {page_number}: {exc}")
                request_ids.extend(r.request_id for r in self.request_log[log_before:])   # failed attempts count too
                if reported_total is not None and reported_total < len(records):
                    reported_total = None
                return CollectionResult(records=records, returned_count=len(records), reported_total=reported_total,
                                        complete=False, status="failed", request_ids=request_ids, warnings=warnings)
            request_ids.append(record.request_id)
            pages_fetched += 1
            page = self._read_page(envelope, warnings)

            if page.total_count is not None:
                if reported_total is not None and page.total_count != reported_total:
                    warnings.append(f"totalCount changed during paging ({reported_total} -> {page.total_count}); "
                                    "the collection is not an atomic snapshot")
                    uncertain = True
                reported_total = page.total_count

            if page.result_null:
                if not records and (page.total_count == 0 or page.total_pages == 0):
                    status = "empty"
                    more_pages = False
                    break
                warnings.append("result is null but the metadata does not say the collection is empty")
                uncertain = True
                null_with_count = True
                more_pages = False
                break
            if page.current_page is not None and page.current_page != page_number:
                warnings.append(f"server returned page {page.current_page} when page {page_number} was requested")
                uncertain = True

            page_ids: list[str] = []
            if id_field is not None:
                for row in page.data:
                    value = row.get(id_field)
                    if not isinstance(value, str) or not value:
                        warnings.append(f"a record has no string {id_field}")
                        uncertain = True
                        continue
                    page_ids.append(value)
                if previous_ids is not None and page_ids and page_ids == previous_ids:
                    warnings.append(f"page {page_number} repeats page {page_number - 1}; stopping")
                    uncertain = True
                    more_pages = False
                    break
                previous_ids = page_ids
                for value in page_ids:
                    seen_ids[value] = seen_ids.get(value, 0) + 1

            records.extend(page.data)

            got = len(page.data)
            more_pages = _more_pages_expected(page, got, page_size, len(records))
            if page.page_size is not None and got > page.page_size:
                warnings.append(f"server returned {got} records but reported pageSize {page.page_size}")
                uncertain = True
            if not more_pages:
                break

        duplicates = {k: v for k, v in seen_ids.items() if v > 1}
        if duplicates:
            warnings.append(f"{len(duplicates)} duplicate {id_field} value(s) across pages, kept as received (e.g. {sorted(duplicates)[:3]})")
            uncertain = True

        if status != "empty":
            if null_with_count:
                status = "unknown"          # the server contradicted itself; nothing can be claimed
            elif more_pages:
                status = "incomplete"
                warnings.append(f"stopped after {pages_fetched} page(s) with more available (max_pages={max_pages})")
            elif reported_total is None:
                status = "unknown"
                warnings.append("the server reported no totalCount; completeness cannot be proven")
            elif reported_total != len(records):
                status = "incomplete" if len(records) < reported_total else "unknown"
                warnings.append(f"returned {len(records)} records but the server reports {reported_total}")
            elif uncertain:
                status = "unknown"
            elif len(records) == 0:
                status = "empty"
            else:
                status = "complete"

        complete = status in ("complete", "empty")
        if reported_total is not None and reported_total < len(records):
            reported_total = None   # contract forbids returned > reported; the warning above explains it
        return CollectionResult(records=records, returned_count=len(records), reported_total=reported_total,
                                complete=complete, status=status, request_ids=request_ids, warnings=warnings)

    @staticmethod
    def _read_page(envelope: dict[str, Any], warnings: list[str]) -> _Page:
        metadata = envelope.get("metadata")
        pagination = metadata.get("pagination") if isinstance(metadata, dict) else None
        if not isinstance(pagination, dict):
            warnings.append("reply has no metadata.pagination")
            pagination = {}
        result = envelope.get("result")
        if result is None:
            data: list[dict[str, Any]] = []
            result_null = True
        else:
            raw = result.get("data")
            if raw is None:
                raw = []
                warnings.append("result.data is missing")
            if not isinstance(raw, list) or not all(isinstance(r, dict) for r in raw):
                raise BrapiClientError("not_brapi", "result.data is not a list of records")
            data = raw
            result_null = False
        return _Page(
            data=data,
            current_page=_int_or_none(pagination.get("currentPage")),
            page_size=_int_or_none(pagination.get("pageSize")),
            total_count=_int_or_none(pagination.get("totalCount")),
            total_pages=_int_or_none(pagination.get("totalPages")),
            result_null=result_null,
        )

    # -- named endpoints -------------------------------------------------------

    def studies(self, filters: StudyFilters | None = None, *, approval: FetchApproval | None = None,
                max_pages: int = 1) -> CollectionResult:
        """Study records. Exact filters go to the server; name_contains is filtered here, on a
        collection that must itself be complete for the answer to be complete.

        Offline, a FILTERED request is a different cache key from the complete catalog a snapshot holds, so
        it would miss. In that one case the exact filters are applied here to the cached complete catalog
        instead, and the result is complete only because the catalog is (a subset of a complete list is
        complete; a subset of an incomplete list is not). Live requests are untouched."""
        filters = filters or StudyFilters()
        if filters.location_name is not None or filters.season_name is not None:
            raise BrapiClientError("invalid_argument", "resolve location_name/season_name to IDs before filtering studies")
        params: dict[str, str | int | bool] = {}
        if filters.location_id is not None:
            params["locationDbId"] = filters.location_id
        if filters.season_id is not None:
            params["seasonDbId"] = filters.season_id
        if filters.study_type is not None:
            params["studyType"] = filters.study_type
        if filters.program_id is not None:
            params["programDbId"] = filters.program_id
        result = self.get_all("/studies", params, approval=approval, max_pages=max_pages)
        if params and self.settings.mode != "live" and result.status == "failed" and result.request_ids == [] \
                and all("offline_cache_miss" in w for w in result.warnings):
            result = self._filter_cached_catalog(params, approval=approval, max_pages=max_pages)
        if filters.name_contains is None:
            return result
        needle = filters.name_contains.lower()
        kept = [r for r in result.records if needle in str(r.get("studyName", "")).lower()]
        return CollectionResult(
            records=kept, returned_count=len(kept),
            reported_total=len(kept) if result.complete else None,
            complete=result.complete, status=_derived_status(result.status, kept),
            request_ids=result.request_ids,
            warnings=result.warnings + [f"name_contains {filters.name_contains!r} applied client-side to {result.returned_count} records"],
        )

    def _filter_cached_catalog(self, params: Mapping[str, str | int | bool], *, approval: FetchApproval | None, max_pages: int) -> CollectionResult:
        """Offline only: the exact study filters applied to the cached complete catalog, with the same meaning the server gives them
        (locationDbId and programDbId equal, seasonDbId one of the study's seasons, studyType equal ignoring case)."""
        base = self.get_all("/studies", {}, approval=approval, max_pages=max_pages)
        if base.status == "failed":
            return CollectionResult(records=[], returned_count=0, reported_total=None, complete=False, status="failed", request_ids=base.request_ids,
                                    warnings=base.warnings + [f"offline: the filtered request {dict(params)} is not cached and neither is the study catalog"])
        wanted_type = str(params["studyType"]).strip().casefold() if "studyType" in params else None

        def keep(r: dict[str, Any]) -> bool:
            if "locationDbId" in params and str(r.get("locationDbId")) != str(params["locationDbId"]):
                return False
            if "programDbId" in params and str(r.get("programDbId")) != str(params["programDbId"]):
                return False
            if "seasonDbId" in params:
                seasons = r.get("seasons") if isinstance(r.get("seasons"), list) else []
                if str(params["seasonDbId"]) not in {str(s) for s in seasons}:
                    return False
            if wanted_type is not None and str(r.get("studyType") or "").strip().casefold() != wanted_type:
                return False
            return True

        kept = [r for r in base.records if keep(r)]
        note = (f"offline: the filtered request {dict(params)} is not cached; exact filters applied client-side to the cached catalog of "
                f"{base.returned_count} studies, which is {base.status}")
        if base.complete:
            return CollectionResult(records=kept, returned_count=len(kept), reported_total=len(kept), complete=True,
                                    status="complete" if kept else "empty", request_ids=base.request_ids, warnings=base.warnings + [note])
        return CollectionResult(records=kept, returned_count=len(kept), reported_total=None, complete=False,
                                status=base.status if base.status in ("incomplete", "unknown") else "incomplete", request_ids=base.request_ids,
                                warnings=base.warnings + [note + "; a subset of an incomplete list cannot be called complete"])

    def study(self, study_id: str, *, approval: FetchApproval | None = None) -> tuple[dict[str, Any], RequestRecord]:
        _check_id(study_id, "study_id")
        envelope, record = self.get(f"/studies/{study_id}", approval=approval)
        result = envelope["result"]
        if not isinstance(result, dict) or not result:
            raise BrapiClientError("not_found", f"study {study_id!r}: empty result")
        return result, record

    def variables(self, *, approval: FetchApproval | None = None, max_pages: int = 1) -> CollectionResult:
        return self.get_all("/variables", approval=approval, max_pages=max_pages)

    def find_variables(self, name_contains: str, *, approval: FetchApproval | None = None,
                       max_pages: int = 1) -> CollectionResult:
        """Literal, case-insensitive substring match on observationVariableName — discovery, not an exact ID."""
        if not name_contains.strip():
            raise BrapiClientError("invalid_argument", "name_contains must not be blank")
        base = self.variables(approval=approval, max_pages=max_pages)
        needle = name_contains.lower()
        kept = [r for r in base.records if needle in str(r.get("observationVariableName", "")).lower()]
        return CollectionResult(
            records=kept, returned_count=len(kept), reported_total=len(kept) if base.complete else None,
            complete=base.complete, status=_derived_status(base.status, kept), request_ids=base.request_ids,
            warnings=base.warnings + [f"literal case-insensitive match {name_contains!r} on {base.returned_count} variables"],
        )

    def observations(self, study_id: str, variable_id: str, *, approval: FetchApproval | None = None,
                     paging: bool = False, max_pages: int = 1) -> CollectionResult:
        """One authorized page first. If more exist: incomplete and STOP, unless paging=True
        (a separately approved bounded sequential mode). Timeouts are never retried."""
        _check_id(study_id, "study_id")
        _check_id(variable_id, "variable_id")
        return self.get_all("/observations", {"studyDbId": study_id, "observationVariableDbId": variable_id},
                            approval=approval, max_pages=max_pages if paging else 1, expensive=True,
                            study_id=study_id, variable_id=variable_id)

    def observation_units(self, study_id: str, *, approval: FetchApproval | None = None,
                          paging: bool = False, max_pages: int = 1) -> CollectionResult:
        _check_id(study_id, "study_id")
        return self.get_all("/observationunits", {"studyDbId": study_id}, approval=approval,
                            max_pages=max_pages if paging else 1, expensive=True, study_id=study_id)

    def locations(self, *, approval: FetchApproval | None = None, max_pages: int = 1) -> CollectionResult:
        return self.get_all("/locations", approval=approval, max_pages=max_pages)

    def programs(self, *, approval: FetchApproval | None = None, max_pages: int = 1) -> CollectionResult:
        return self.get_all("/programs", approval=approval, max_pages=max_pages)

    def seasons(self, *, approval: FetchApproval | None = None, max_pages: int = 1) -> CollectionResult:
        return self.get_all("/seasons", approval=approval, max_pages=max_pages)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

_REQUEST_CODES = {"connection", "timeout", "http_error", "invalid_json", "not_brapi", "blocked", "cancelled", "too_large", "unknown"}


def _redacted_url(url: str, params: Mapping[str, str | int | bool]) -> str:
    """The URL as it may appear in logs. Our requests carry no secrets; this keeps it that way."""
    if not params:
        return url
    from urllib.parse import urlencode

    return f"{url}?{urlencode({k: params[k] for k in sorted(params)})}"


def _parse_retry_after(headers: Mapping[str, str]) -> float | None:
    value = None
    for key, val in headers.items():
        if key.lower() == "retry-after":
            value = val
            break
    if value is None:
        return None
    value = value.strip()
    if re.fullmatch(r"\d+", value):
        return float(value)
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _more_pages_expected(page: _Page, got: int, requested_size: int, total_so_far: int) -> bool:
    """Decide from the server's own numbers whether another page exists."""
    if page.total_pages is not None and page.current_page is not None:
        return page.current_page + 1 < page.total_pages
    if page.total_count is not None:
        return total_so_far < page.total_count
    effective = page.page_size if page.page_size is not None else requested_size
    return got >= effective and got > 0


def _derived_status(base_status: CollectionStatus, kept: list[dict[str, Any]]) -> CollectionStatus:
    if base_status in ("complete", "empty"):
        return "complete" if kept else "empty"
    return base_status


def records_to_df(collection: CollectionResult, columns: list[str] | None = None):
    """A pandas DataFrame with EVERY cell as text (str) or None.

    Nested objects/lists become JSON text. Nothing is converted to a number here:
    '007' stays '007', '' stays '', 'bad' stays 'bad'. Later stages validate and convert.
    """
    import pandas as pd

    rows: list[dict[str, str | None]] = []
    for record in collection.records:
        row: dict[str, str | None] = {}
        for key, value in record.items():
            if value is None:
                row[key] = None
            elif isinstance(value, (dict, list)):
                row[key] = json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False)
            elif isinstance(value, bool):
                row[key] = "true" if value else "false"
            else:
                row[key] = str(value)
        rows.append(row)
    frame = pd.DataFrame(rows, columns=columns, dtype=object)
    return frame
