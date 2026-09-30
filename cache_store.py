"""
cache_store.py — safe request identity (the cache KEY) and evidence storage.

Two different fingerprints, on purpose:

* The KEY is the SHA-256 of the QUESTION: which server (full base URL, path
  included), which endpoint, which validated parameters, which cache format
  version and which access namespace. Same question → same key, on any day.
* The RESPONSE HASH is the SHA-256 of the ANSWER bytes the server sent.
  The same question asked twice can give two different answers (the data
  changed), so one key can hold several VERSIONS, each with its own hash.

Everyday example: the key is the exact wording of a library request slip;
the response hash is the seal on the box that came back. Two slips with the
same wording go to the same shelf; each delivery keeps its own seal.

Storage layout (under the cache root the application chose, e.g. part2/cache):

    <key[:2]>/<key>/v0001/response.bin   exact bytes as received
    <key[:2]>/<key>/v0001/entry.json     metadata (identity, hashes, times)
    <key[:2]>/<key>/v0002/...            a refresh: new version, old one untouched

Rules enforced here:
* keys come from canonical JSON, never from "key=value&..." strings;
* only str / int / bool parameter values are accepted;
* an entry is written to temporary files and moved into place atomically,
  and entry.json is written LAST — so an interrupted write never looks like
  a valid entry;
* a cache hit re-checks identity, key and response hash before returning;
* anything corrupt raises CacheCorruptError (optionally quarantining the
  bad version) — it is never turned into an empty response;
* a stored evidence snapshot is never overwritten; a refresh adds a version.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from pydantic import Field, ValidationError, field_validator

from contracts import (
    BaseUrl,
    Count,
    EndpointPath,
    HttpStatus,
    ManagedPath,
    PositiveInt,
    RedactedUrl,
    RequestRecord,
    Sha256,
    StrictRecord,
    UtcDatetime,
)

__all__ = [
    "CACHE_FORMAT_VERSION",
    "PUBLIC_NAMESPACE",
    "CacheError",
    "CacheCorruptError",
    "CacheIdentity",
    "CacheEntry",
    "canonical_json",
    "cache_key",
    "sha256_bytes",
    "CacheStore",
]

CACHE_FORMAT_VERSION = "1"
PUBLIC_NAMESPACE = "public"

_ENTRY_FILE = "entry.json"
_BODY_FILE = "response.bin"
_VERSION_DIR = re.compile(r"^v(\d{4,})$")


class CacheError(Exception):
    """Something the cache refuses to do (wrong namespace, bad argument, ...)."""


class CacheCorruptError(CacheError):
    """A stored entry failed validation. `path` says which one."""

    def __init__(self, message: str, path: Path | None = None) -> None:
        super().__init__(message if path is None else f"{message} [{path}]")
        self.path = path


ParamValue = str | int | bool
Namespace = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]{0,31}$")]
FormatVersion = Annotated[str, Field(pattern=r"^[0-9]+$")]


# --------------------------------------------------------------------------
# Identity and key
# --------------------------------------------------------------------------

class CacheIdentity(StrictRecord):
    """Everything that makes one request different from another. Hashed into the key."""

    base_url: BaseUrl                      # normalized; path component kept (…/brapi/v2)
    endpoint: EndpointPath
    params: dict[str, ParamValue] = Field(default_factory=dict)
    format_version: FormatVersion = CACHE_FORMAT_VERSION
    access_namespace: Namespace = PUBLIC_NAMESPACE

    @field_validator("params")
    @classmethod
    def _param_names(cls, value: dict[str, ParamValue]) -> dict[str, ParamValue]:
        for key in value:
            if not key.strip():
                raise ValueError("parameter names must not be blank")
        return value


def canonical_json(identity: CacheIdentity) -> str:
    """One fixed text form of the identity: sorted keys, no spaces, ASCII only, no NaN."""
    payload = {
        "access_namespace": identity.access_namespace,
        "base_url": identity.base_url,
        "endpoint": identity.endpoint,
        "format_version": identity.format_version,
        "params": {k: identity.params[k] for k in sorted(identity.params)},
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def cache_key(identity: CacheIdentity) -> str:
    """SHA-256 of the canonical JSON — 64 hex characters."""
    return hashlib.sha256(canonical_json(identity).encode("ascii")).hexdigest()


def sha256_bytes(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


# --------------------------------------------------------------------------
# Stored entry metadata (this is what entry.json contains)
# --------------------------------------------------------------------------

class CacheEntry(StrictRecord):
    """Metadata for one stored version. Carries everything a RequestRecord needs on a cache hit."""

    key: Sha256
    version: PositiveInt
    identity: CacheIdentity
    http_status: HttpStatus
    fetched_at_utc: UtcDatetime        # when the server actually answered (preserved on every hit)
    stored_at_utc: UtcDatetime         # when this version was written to disk
    source_url_redacted: RedactedUrl
    response_sha256: Sha256
    byte_length: Count
    content_type: str | None = None
    relative_path: ManagedPath         # "<key[:2]>/<key>/vNNNN", relative to the cache root

    def request_record(
        self,
        *,
        request_id: str,
        run_id: str,
        requested_at_utc: datetime,
        attempt: int = 1,
        duration_seconds: float = 0.0,
    ) -> RequestRecord:
        """Build the receipt for a cache hit. The original fetch time is kept, not replaced."""
        return RequestRecord(
            request_id=request_id,
            run_id=run_id,
            base_url=self.identity.base_url,
            endpoint=self.identity.endpoint,
            params=dict(self.identity.params),
            requested_at_utc=requested_at_utc,
            fetched_at_utc=self.fetched_at_utc,
            origin="cache",
            http_status=self.http_status,
            attempt=attempt,
            duration_seconds=duration_seconds,
            response_sha256=self.response_sha256,
            source_url_redacted=self.source_url_redacted,
        )


# --------------------------------------------------------------------------
# The store
# --------------------------------------------------------------------------

def _write_atomic(target: Path, data: bytes) -> None:
    """Write to a temporary file beside the target, flush to disk, then move it into place."""
    tmp = target.with_name(f"{target.name}.tmp-{uuid.uuid4().hex}")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, target)   # atomic on the same volume, Windows included


class CacheStore:
    """Versioned, hash-checked storage of exact responses under one root folder."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    # -- locating ----------------------------------------------------------

    def _check_scope(self, identity: CacheIdentity) -> str:
        if identity.access_namespace != PUBLIC_NAMESPACE:
            raise CacheError(f"access namespace {identity.access_namespace!r} is out of scope; only 'public' is served")
        if identity.format_version != CACHE_FORMAT_VERSION:
            raise CacheError(f"cache format version {identity.format_version!r} is not this store's version {CACHE_FORMAT_VERSION!r}")
        return cache_key(identity)

    def key_dir(self, identity: CacheIdentity) -> Path:
        key = cache_key(identity)
        return self.root / key[:2] / key

    def versions(self, identity: CacheIdentity) -> list[int]:
        """Version numbers that have a completed entry.json (interrupted writes are not listed)."""
        key_dir = self.key_dir(identity)
        if not key_dir.is_dir():
            return []
        found = []
        for child in key_dir.iterdir():
            match = _VERSION_DIR.match(child.name)
            if match and child.is_dir() and (child / _ENTRY_FILE).is_file():
                found.append(int(match.group(1)))
        return sorted(found)

    def _next_version(self, key_dir: Path) -> int:
        """One more than any version directory ever created here — including corrupt or half-written ones."""
        highest = 0
        if key_dir.is_dir():
            for child in key_dir.iterdir():
                match = re.match(r"^v(\d{4,})", child.name)
                if match:
                    highest = max(highest, int(match.group(1)))
        return highest + 1

    # -- writing -------------------------------------------------------------

    def put(
        self,
        identity: CacheIdentity,
        body: bytes,
        *,
        http_status: int,
        fetched_at_utc: datetime,
        source_url_redacted: str,
        content_type: str | None = None,
    ) -> CacheEntry:
        """Store exact response bytes as a NEW version. Never overwrites an earlier version."""
        if not isinstance(body, (bytes, bytearray)):
            raise CacheError("body must be bytes (the exact response), not text")
        key = self._check_scope(identity)
        key_dir = self.root / key[:2] / key
        key_dir.mkdir(parents=True, exist_ok=True)
        version = self._next_version(key_dir)
        version_dir = key_dir / f"v{version:04d}"
        version_dir.mkdir(parents=False, exist_ok=False)   # a version is created exactly once

        body = bytes(body)
        entry = CacheEntry(
            key=key,
            version=version,
            identity=identity,
            http_status=http_status,
            fetched_at_utc=fetched_at_utc,
            stored_at_utc=datetime.now(timezone.utc),
            source_url_redacted=source_url_redacted,
            response_sha256=sha256_bytes(body),
            byte_length=len(body),
            content_type=content_type,
            relative_path=f"{key[:2]}/{key}/v{version:04d}",
        )
        _write_atomic(version_dir / _BODY_FILE, body)                     # bytes first ...
        _write_atomic(version_dir / _ENTRY_FILE, entry.model_dump_json(indent=2).encode("utf-8"))  # ... entry last
        return entry

    # -- reading -------------------------------------------------------------

    def get(self, identity: CacheIdentity, *, quarantine: bool = False) -> tuple[CacheEntry, bytes] | None:
        """Return the latest valid version, or None on a miss. Corruption raises, never returns empty."""
        key = self._check_scope(identity)
        versions = self.versions(identity)
        if not versions:
            return None
        version_dir = self.root / key[:2] / key / f"v{versions[-1]:04d}"
        try:
            return self._load_checked(version_dir, identity, key)
        except CacheCorruptError:
            if quarantine:
                self._quarantine(version_dir)
            raise

    def get_version(self, identity: CacheIdentity, version: int) -> tuple[CacheEntry, bytes]:
        key = self._check_scope(identity)
        version_dir = self.root / key[:2] / key / f"v{version:04d}"
        if not (version_dir / _ENTRY_FILE).is_file():
            raise CacheError(f"no completed version {version} for this identity")
        return self._load_checked(version_dir, identity, key)

    def _load_checked(self, version_dir: Path, identity: CacheIdentity, key: str) -> tuple[CacheEntry, bytes]:
        entry_path = version_dir / _ENTRY_FILE
        try:
            entry = CacheEntry.model_validate_json(entry_path.read_bytes())
        except (OSError, ValueError, ValidationError) as exc:   # ValueError covers bad JSON
            raise CacheCorruptError(f"entry.json unreadable or malformed: {type(exc).__name__}", entry_path) from exc
        if entry.key != key or canonical_json(entry.identity) != canonical_json(identity):
            raise CacheCorruptError("entry identity does not match the requested identity", entry_path)
        expected_dir = f"{key[:2]}/{key}/v{entry.version:04d}"
        if entry.relative_path != expected_dir or version_dir.name != f"v{entry.version:04d}":
            raise CacheCorruptError("entry version/path does not match its location", entry_path)
        body_path = version_dir / _BODY_FILE
        try:
            body = body_path.read_bytes()
        except OSError as exc:
            raise CacheCorruptError("response.bin missing or unreadable", body_path) from exc
        if len(body) != entry.byte_length or sha256_bytes(body) != entry.response_sha256:
            raise CacheCorruptError("response bytes do not match the stored hash", body_path)
        return entry, body

    def _quarantine(self, version_dir: Path) -> Path:
        target = version_dir.with_name(f"{version_dir.name}.corrupt-{uuid.uuid4().hex[:8]}")
        os.replace(version_dir, target)
        return target
