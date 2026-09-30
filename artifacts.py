"""
artifacts.py — managed artifacts: safe handles instead of file paths.

Plain-words summary:
* A tool never receives or returns a file path. It gets an opaque handle such
  as art_3f9a2c_0001. This registry maps the handle to a file the APPLICATION
  created, inside a per-run folder under part2/out, and to its manifest
  (hash, schema, source request IDs, completeness, column meanings).
* Filenames are generated (0001_observations.csv), never built from study IDs
  or names a model chose.
* Nothing is overwritten. A saved table is an immutable evidence snapshot;
  loading it re-checks the SHA-256 hash.
* Every cell is stored as TEXT. '007' stays '007', '' stays '', 'bad' stays
  'bad'. A JSON null is written as an empty cell and that policy is recorded.
  A non-finite float (NaN, inf) is written as null (empty) and the manifest
  records the row, column and reason.
* Input is resolved ONLY from this registry: this run's artifacts, or a
  snapshot imported explicitly from another run of the SAME server. Paths,
  URLs, '..', absolute paths and symlink escapes are rejected.
* A zero-row table still has columns and a manifest, and a zero-row study
  stays on the study-status ledger.

Everyday example: a coat-check. You hand over a coat and get ticket 0042.
The ticket says nothing about where the coat hangs; only the attendant maps
ticket -> hook, and only tickets issued by this coat-check are honoured.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import Field, ValidationError

from contracts import (
    ArtifactId,
    ArtifactKind,
    ArtifactManifest,
    CollectionResult,
    Count,
    IdStr,
    MeasurementMeta,
    StrictRecord,
    StudyFetchStatus,
    StudyStatusEntry,
)

__all__ = [
    "ArtifactError",
    "ArtifactPreview",
    "ArtifactRegistry",
    "NULL_POLICY",
    "cell_text",
]

NULL_POLICY = "JSON null is written as an empty cell; the raw token '' is also an empty cell. See quality_notes for non-finite values."
_HANDLE_RE = re.compile(r"^art_[0-9a-f]{6}_[0-9]{4,}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
_REGISTRY_FILE = "registry.json"


class ArtifactError(Exception):
    """Every registry failure carries a fixed code plus a plain message."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class ArtifactPreview(StrictRecord):
    """What a tool may show in a message: a FEW rows plus the true total. Never the whole table."""

    artifact_id: ArtifactId
    kind: ArtifactKind
    total_rows: Count
    displayed_rows: Count
    complete: bool
    columns: list[str]
    rows: list[dict[str, str | None]]
    quality_notes: list[str] = Field(default_factory=list)


def cell_text(value: Any) -> tuple[str | None, str | None]:
    """Turn one raw value into (text or None, reason-if-null). Never converts numbers."""
    if value is None:
        return None, None
    if isinstance(value, bool):
        return ("true" if value else "false"), None
    if isinstance(value, float):
        if not math.isfinite(value):
            return None, f"non-finite float {value!r} replaced by null"
        return repr(value), None
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False), None
    return str(value), None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new_file(path: Path, data: bytes) -> None:
    """Create a file that must not exist yet (O_EXCL), write, flush, fsync. Never overwrites."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    try:
        fd = os.open(path, flags, 0o644)
    except FileExistsError as exc:
        raise ArtifactError("exists", f"refusing to overwrite {path.name}") from exc
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


class ArtifactRegistry:
    """One registry per run. Files live under <out_root>/<run_id>/artifacts/."""

    def __init__(self, out_root: Path | str, run_id: str, *, base_url: str) -> None:
        if not _RUN_ID_RE.match(run_id) or ".." in run_id or set(run_id) <= {"."}:
            raise ArtifactError("invalid_argument", f"run_id {run_id!r} is not a plain identifier")
        self.out_root = Path(out_root).resolve()
        self.run_id = run_id
        self.base_url = base_url
        self.run_dir = self.out_root / run_id
        self.artifact_dir = self.run_dir / "artifacts"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._prefix = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:6]
        self._manifests: dict[str, ArtifactManifest] = {}
        self._roots: dict[str, Path] = {}            # artifact_id -> the run dir that owns the file
        self._ledger: dict[str, StudyStatusEntry] = {}
        self._imported_runs: list[str] = []
        self._seq = 0
        if (self.run_dir / _REGISTRY_FILE).is_file():
            self._load_registry_file()

    # -- saving ----------------------------------------------------------------

    def save_table(
        self,
        records: list[dict[str, Any]],
        *,
        kind: ArtifactKind,
        columns: dict[str, str],
        collection: CollectionResult | None = None,
        complete: bool | None = None,
        source_request_ids: list[str] | None = None,
        study_ids: list[str] | None = None,
        variable_ids: list[str] | None = None,
        measurement: MeasurementMeta | None = None,
        quality_notes: list[str] | None = None,
        schema_version: str = "1",
    ) -> ArtifactManifest:
        """Write records as a text-only CSV with the given columns, register it, return the manifest.

        columns maps column name -> meaning; the CSV has exactly these columns in this order.
        complete comes from the collection when given; it must be stated explicitly otherwise.
        """
        if not columns:
            raise ArtifactError("invalid_argument", "a table needs at least one defined column, even when empty")
        for name in columns:
            if not isinstance(name, str) or not name.strip() or "\n" in name or "\r" in name:
                raise ArtifactError("invalid_argument", f"bad column name {name!r}")
        if collection is not None:
            if collection.returned_count != len(records):
                raise ArtifactError("invalid_argument", "records do not match the collection's returned_count")
            complete = collection.complete
            source_request_ids = list(source_request_ids or collection.request_ids)
        if complete is None:
            raise ArtifactError("invalid_argument", "state complete=True/False explicitly when no collection is given")

        notes = list(quality_notes or [])
        notes.append(NULL_POLICY)
        header = list(columns)
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(header)
        dropped_fields: set[str] = set()
        for row_index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ArtifactError("invalid_argument", f"record {row_index} is not an object")
            dropped_fields.update(k for k in record if k not in columns)
            out_row = []
            for name in header:
                text, reason = cell_text(record.get(name))
                if reason is not None:
                    notes.append(f"row {row_index} column {name!r}: {reason}")
                out_row.append("" if text is None else text)
            writer.writerow(out_row)
        if dropped_fields:
            notes.append(f"fields outside the declared columns were not stored: {sorted(dropped_fields)[:20]}")
        data = buffer.getvalue().encode("utf-8")

        self._seq += 1
        artifact_id = f"art_{self._prefix}_{self._seq:04d}"
        filename = f"{self._seq:04d}_{kind}.csv"          # generated: never a study ID or a model's choice
        path = self.artifact_dir / filename
        _write_new_file(path, data)                        # refuses if it somehow exists
        manifest = ArtifactManifest(
            artifact_id=artifact_id,
            run_id=self.run_id,
            kind=kind,
            relative_path=f"{self.run_id}/artifacts/{filename}",
            sha256=hashlib.sha256(data).hexdigest(),
            schema_version=schema_version,
            row_count=len(records),
            complete=bool(complete),
            source_request_ids=list(source_request_ids or []),
            study_ids=list(study_ids or []),
            variable_ids=list(variable_ids or []),
            measurement=measurement or MeasurementMeta(),
            columns=dict(columns),
            quality_notes=notes,
            created_at_utc=datetime.now(timezone.utc),
        )
        _write_new_file(path.with_suffix(".manifest.json"), manifest.model_dump_json(indent=2).encode("utf-8"))
        self._manifests[artifact_id] = manifest
        self._roots[artifact_id] = self.out_root
        self._save_registry_file()
        return manifest

    # -- the study ledger --------------------------------------------------------

    def record_study(self, study_id: str, status: StudyFetchStatus, *, artifact_id: str | None = None,
                     reason: str | None = None, request_ids: list[str] | None = None) -> StudyStatusEntry:
        """A study stays on the ledger whatever happened to it: complete, empty, failed, blocked ..."""
        if artifact_id is not None:
            self._check_handle(artifact_id)
            if artifact_id not in self._manifests:
                raise ArtifactError("invalid_handle", f"{artifact_id} is not registered in this run")
        entry = StudyStatusEntry(study_id=study_id, status=status, artifact_id=artifact_id, reason=reason,
                                 request_ids=list(request_ids or []))
        self._ledger[study_id] = entry
        self._save_registry_file()
        return entry

    def ledger(self) -> list[StudyStatusEntry]:
        return [self._ledger[k] for k in sorted(self._ledger)]

    # -- resolving handles -------------------------------------------------------

    @staticmethod
    def _check_handle(handle: Any) -> str:
        if not isinstance(handle, str) or not _HANDLE_RE.match(handle):
            raise ArtifactError("invalid_handle", f"{handle!r} is not an artifact handle (expected art_xxxxxx_0001)")
        return handle

    def manifest(self, artifact_id: str) -> ArtifactManifest:
        self._check_handle(artifact_id)
        manifest = self._manifests.get(artifact_id)
        if manifest is None:
            raise ArtifactError("invalid_handle", f"{artifact_id} is not registered in this run or its imported snapshots")
        return manifest

    def artifact_ids(self) -> list[str]:
        return sorted(self._manifests)

    def _safe_path(self, manifest: ArtifactManifest) -> Path:
        """The real file behind a manifest, proven to sit inside its run root (no traversal, no symlink escape)."""
        root = self._roots[manifest.artifact_id]
        candidate = (root / manifest.relative_path)
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise ArtifactError("missing", f"{manifest.artifact_id}: file is missing") from exc
        if not resolved.is_relative_to(root.resolve()):
            raise ArtifactError("escape", f"{manifest.artifact_id}: file resolves outside the run root")
        if candidate.is_symlink():
            raise ArtifactError("escape", f"{manifest.artifact_id}: artifact path is a symlink")
        return resolved

    def load(self, artifact_id: str) -> tuple[ArtifactManifest, list[dict[str, str]]]:
        """Read a table back, verifying its hash. Every cell is text."""
        manifest = self.manifest(artifact_id)
        path = self._safe_path(manifest)
        if _sha256_file(path) != manifest.sha256:
            raise ArtifactError("hash_mismatch", f"{artifact_id}: file bytes do not match the manifest hash")
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header != list(manifest.columns):
                raise ArtifactError("schema_mismatch", f"{artifact_id}: CSV header differs from the manifest columns")
            rows = [dict(zip(header, row)) for row in reader]
        if len(rows) != manifest.row_count:
            raise ArtifactError("schema_mismatch", f"{artifact_id}: {len(rows)} rows on disk, manifest says {manifest.row_count}")
        return manifest, rows

    def preview(self, artifact_id: str, max_rows: int = 5) -> ArtifactPreview:
        """A few rows for a message. total_rows is the real count; displayed_rows is what is shown."""
        if max_rows < 0:
            raise ArtifactError("invalid_argument", "max_rows must be >= 0")
        manifest, rows = self.load(artifact_id)
        shown = rows[:max_rows]
        return ArtifactPreview(
            artifact_id=artifact_id, kind=manifest.kind, total_rows=manifest.row_count,
            displayed_rows=len(shown), complete=manifest.complete, columns=list(manifest.columns),
            rows=[dict(row) for row in shown],                  # exactly the stored text; '' means empty/missing
            quality_notes=[n for n in manifest.quality_notes if n != NULL_POLICY][:10],
        )

    # -- importing a snapshot from another run of the same server -----------------

    def import_run(self, other_run_id: str) -> list[str]:
        """Explicitly bring another run's artifacts (same server only) into scope, verifying every hash."""
        if not _RUN_ID_RE.match(other_run_id):
            raise ArtifactError("invalid_argument", f"run_id {other_run_id!r} is not a plain identifier")
        other_dir = self.out_root / other_run_id
        registry_path = other_dir / _REGISTRY_FILE
        if not registry_path.is_file():
            raise ArtifactError("missing", f"run {other_run_id!r} has no registry under {self.out_root}")
        try:
            data = json.loads(registry_path.read_text(encoding="utf-8"))
            # validate from JSON text: strict JSON mode accepts ISO datetimes, python mode would not
            manifests = [ArtifactManifest.model_validate_json(json.dumps(m)) for m in data["artifacts"].values()]
        except (ValueError, KeyError, TypeError, ValidationError) as exc:
            raise ArtifactError("corrupt", f"registry of run {other_run_id!r} is unreadable: {type(exc).__name__}") from exc
        if data.get("base_url") != self.base_url:
            raise ArtifactError("server_mismatch",
                                f"run {other_run_id!r} belongs to {data.get('base_url')!r}, this run uses {self.base_url!r}")
        imported = []
        for manifest in manifests:
            if manifest.artifact_id in self._manifests:
                raise ArtifactError("exists", f"{manifest.artifact_id} is already registered")
            self._roots[manifest.artifact_id] = self.out_root
            self._manifests[manifest.artifact_id] = manifest
            try:
                self.load(manifest.artifact_id)          # verifies path containment and hash
            except ArtifactError:
                del self._manifests[manifest.artifact_id]
                del self._roots[manifest.artifact_id]
                raise
            imported.append(manifest.artifact_id)
        self._imported_runs.append(other_run_id)
        self._save_registry_file()
        return imported

    # -- persistence of the registry index -----------------------------------------

    def _save_registry_file(self) -> None:
        own = {aid: json.loads(m.model_dump_json()) for aid, m in self._manifests.items() if m.run_id == self.run_id}
        payload = {
            "run_id": self.run_id,
            "base_url": self.base_url,
            "artifacts": own,
            "imported_runs": list(self._imported_runs),
            "ledger": [json.loads(e.model_dump_json()) for e in self.ledger()],
        }
        # no sort_keys: the order of a manifest's `columns` IS the CSV header order and must survive a reload
        _write_atomic(self.run_dir / _REGISTRY_FILE, json.dumps(payload, indent=2).encode("utf-8"))

    def _load_registry_file(self) -> None:
        try:
            data = json.loads((self.run_dir / _REGISTRY_FILE).read_text(encoding="utf-8"))
            if data.get("base_url") != self.base_url:
                raise ArtifactError("server_mismatch", f"existing registry for run {self.run_id!r} belongs to another server")
            for aid, raw in data.get("artifacts", {}).items():
                manifest = ArtifactManifest.model_validate_json(json.dumps(raw))
                self._manifests[aid] = manifest
                self._roots[aid] = self.out_root
                self._seq = max(self._seq, int(aid.rsplit("_", 1)[1]))
            for raw in data.get("ledger", []):
                entry = StudyStatusEntry.model_validate_json(json.dumps(raw))
                self._ledger[entry.study_id] = entry
        except (ValueError, KeyError, TypeError, ValidationError) as exc:
            raise ArtifactError("corrupt", f"registry of run {self.run_id!r} is unreadable: {type(exc).__name__}") from exc
        for other in data.get("imported_runs", []):
            self.import_run(other)
