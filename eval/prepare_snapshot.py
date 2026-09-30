"""
eval/prepare_snapshot.py — freeze the evaluation input: an immutable, hash-verified SNAPSHOT.

Plain-words summary:
* A snapshot is a photocopy of the database replies, locked away. Once written, nothing in it
  changes; every file carries a SHA-256 fingerprint, the manifest lists them all, and the manifest
  itself is fingerprinted (MANIFEST.sha256). Before anyone grades against a snapshot, every
  fingerprint is recomputed and compared (verify_snapshot). A changed byte, a missing file or a
  manifest from another server is refused — there is no "probably fine".
* The offline path (--fixture synthetic) builds the snapshot from the COMPLETE artifacts already in the
  local cache: it asks the ordinary BrAPI client, offline, for each collection, keeps only what the
  client itself calls complete or empty, and copies the exact response bytes. Anything not cached is
  recorded as not_in_source — never fetched, never guessed. No live request can happen here.
* Layout of snapshots/<ID>/:
    manifest.json        source server, versions, every response with its hash, variables and units,
                         every study's status, the transformations that turn bytes into rows
    MANIFEST.sha256      fingerprint of manifest.json
    responses/<sha>.bin  the exact bytes of each reply (for grading and hashing)
    cache/               the same bytes as a one-version CacheStore, so the application can read the
                         snapshot exactly as it reads its cache (used by run.py --snapshot in 6B)
  The snapshot ID is content-addressed: snap_<fixture>_<12 hex of the sorted response hashes and
  the server>. Preparing the same source twice gives the same ID and reuses the folder.
* The --live path (Step 6.3) is interactive and incompatible with --auto. It prepares ONE study and ONE
  variable from the real server named in .env, in two separately approved steps, each typed "yes" by the
  human at the keyboard (an EOF is a cancel, never a yes):
    step 1  METADATA DISCOVERY: GET /serverinfo, the complete /studies, /variables, /locations and /seasons
            catalogs (public metadata, no measurements). The study and the trait named in --question are
            resolved by literal, case-insensitive match — exact ID, else exact name, else a UNIQUE name
            containing the phrase; anything ambiguous or missing stops the preparation with the candidates
            listed. The application resolves names only against a COMPLETE catalog, so an incomplete study or
            variable catalog stops it too.
    step 2  OBSERVATION RETRIEVAL for exactly the resolved study/variable pair plus that study's plot roster,
            after the human has seen the resolved IDs, names and unit.
  Each step runs under a FetchApproval built here, in code, after the "yes": the client re-checks it before
  every HTTP attempt, and --max-http-attempts is one ceiling for the whole preparation (both steps, retries
  included). It refuses before asking anything unless ACCESS_NOTES.md records a review date and
  BRAPI_MODE=live was set on purpose. The result is sealed exactly like the synthetic snapshot, with
  source.synthetic=false, the question, the resolved pair, the two approvals and the full request log in the
  manifest. Real bytes stay under git-ignored folders (cache/live and snapshots/).

Everyday example: an exam board seals the question paper in a numbered envelope, photographs the seal
and logs the number. On exam day the seal is checked before the envelope is opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from brapi_client import PART2_DIR, BrapiClient, BrapiClientError, Settings, load_settings
from cache_store import CACHE_FORMAT_VERSION, CacheIdentity, CacheStore
from contracts import CONTRACT_VERSION, CollectionResult, FetchApproval
from explore_cassavabase import ACCESS_NOTES_PATH, SYNTHETIC_BASE, access_notes_review_date, seed_synthetic

__all__ = [
    "SNAPSHOT_SCHEMA", "SNAPSHOTS_DIR", "SnapshotCheck", "prepare_synthetic_snapshot", "verify_snapshot", "load_manifest",
    "snapshot_id_for", "code_versions", "main", "LiveOutcome", "prepare_live_snapshot", "parse_preparation_question",
]

SNAPSHOT_SCHEMA = "snapshot.v1"
SNAPSHOTS_DIR = PART2_DIR / "snapshots"
MANIFEST = "manifest.json"
MANIFEST_HASH = "MANIFEST.sha256"
RESPONSES_DIR = "responses"
CACHE_DIR = "cache"
LIVE_LABEL = "live"                  # snapshot IDs of real data start with snap_live_
CATALOG_MAX_PAGES = 20               # the application's own export ceiling; a catalog needing more cannot be used by it anyway
APPROVAL_MINUTES = 30                # how long each typed approval stays valid
MAX_READ_TIMEOUT = 600.0             # --read-timeout ceiling: ten minutes of silence is the most we ever wait for one reply
# exit codes of the live path
EXIT_OK, EXIT_FAILED, EXIT_REFUSED, EXIT_NO_TERMINAL, EXIT_STOPPED, EXIT_CANCELLED = 0, 1, 2, 3, 4, 130
NO_TERMINAL = ("CANCELLED: no interactive terminal to receive your approval. Run this command yourself in a terminal; "
               "the helper must not answer for you.")
# how bytes become rows — written into every manifest so a grader knows the rules that produced the numbers
TRANSFORMATIONS = [
    "each response file is the exact JSON body the server (or fixture) sent; records are result.data[] (or result for a single object)",
    "every cell is kept as its raw text token; nothing is converted or rounded when stored",
    "a value token is valid when it is a plain finite decimal number; '', NA, N/A, NULL, null, None and . are raw-missing; every other token (inf, nan, 'bad', overflow) is invalid",
    "a collection counts as complete only when the client's pagination check says complete or empty; anything else is recorded with its status and not treated as the whole set",
    "observation units are plots; several observation rows can share one unit, so n_valid_values and n_independent_units are counted separately",
]
CODE_FILES = ["contracts.py", "cache_store.py", "brapi_client.py", "explore_cassavabase.py", "artifacts.py", "brapi_mcp_server.py",
              "analyst_tools.py", "eval/prepare_snapshot.py", "eval/ground_truth.py", "eval/questions.json"]


class SnapshotError(Exception):
    """Preparation or verification refused; the message says why."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_versions() -> dict[str, Any]:
    """What produced this snapshot: file fingerprints and library versions (never secrets)."""
    import pandas
    import pydantic

    files = {name: _sha256_file(PART2_DIR / name) for name in CODE_FILES if (PART2_DIR / name).is_file()}
    lock = PART2_DIR / "requirements.lock.txt"
    return {
        "contract_version": CONTRACT_VERSION, "cache_format_version": CACHE_FORMAT_VERSION, "snapshot_schema": SNAPSHOT_SCHEMA,
        "python": platform.python_version(), "pydantic": pydantic.VERSION, "pandas": pandas.__version__,
        "requirements_lock_sha256": _sha256_file(lock) if lock.is_file() else None, "code_sha256": files,
    }


def snapshot_id_for(fixture: str, base_url: str, response_hashes: list[str]) -> str:
    digest = _sha256((base_url + "\n" + "\n".join(sorted(response_hashes))).encode("utf-8"))
    return f"snap_{fixture}_{digest[:12]}"


# --------------------------------------------------------------------------
# Preparation from the offline cache (the only path that produces bytes in this version)
# --------------------------------------------------------------------------

@dataclass
class _Response:
    endpoint: str
    params: dict[str, Any]
    request_id: str
    http_status: int | None
    fetched_at_utc: str | None
    source_url_redacted: str
    content_type: str | None
    sha256: str
    data: bytes
    collection: dict[str, Any] | None = None


@dataclass
class _Prepared:
    responses: list[_Response] = field(default_factory=list)
    studies: list[dict[str, Any]] = field(default_factory=list)
    variables: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _collection_summary(result: CollectionResult) -> dict[str, Any]:
    return {"status": result.status, "complete": result.complete, "returned": result.returned_count, "reported_total": result.reported_total,
            "warnings": list(result.warnings)}


def _capture(client: BrapiClient, prepared: _Prepared, collection: CollectionResult | None, since: int) -> list[_Response]:
    """Copy the exact bytes behind the client's newest request records.

    A successful record — a cache hit, or a live reply the client has just stored in its cache — is read back
    from the cache and its hash re-checked. A failed attempt is noted, never guessed at.
    """
    captured: list[_Response] = []
    for record in client.request_log[since:]:
        if record.error is not None or record.origin not in ("cache", "live"):
            prepared.notes.append(f"{record.endpoint} {record.params}: {record.origin} {record.error.code if record.error else ''} — not in the snapshot")
            continue
        identity = CacheIdentity(base_url=client.settings.base_url, endpoint=record.endpoint, params=dict(record.params))
        got = client.cache.get(identity)
        if got is None:
            raise SnapshotError(f"{record.endpoint} was served from cache but the cache entry cannot be read back")
        entry, data = got
        if _sha256(data) != entry.response_sha256 or entry.response_sha256 != record.response_sha256:
            raise SnapshotError(f"{record.endpoint}: cache bytes do not match their recorded hash; refusing to snapshot corrupt evidence")
        captured.append(_Response(endpoint=record.endpoint, params=dict(record.params), request_id=record.request_id, http_status=entry.http_status,
                                  fetched_at_utc=entry.fetched_at_utc.isoformat(), source_url_redacted=entry.source_url_redacted,
                                  content_type=entry.content_type, sha256=entry.response_sha256, data=data,
                                  collection=_collection_summary(collection) if collection is not None else None))
    prepared.responses.extend(captured)
    return captured


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return str(value)


def _sub(record: dict[str, Any], key: str, inner: str) -> str | None:
    """record[key][inner] as text when record[key] is an object, else None."""
    value = record.get(key)
    return _text(value.get(inner)) if isinstance(value, dict) else None


def _variable_entry(v: dict[str, Any]) -> dict[str, Any]:
    return {"variable_id": _text(v.get("observationVariableDbId")), "name": _text(v.get("observationVariableName")),
            "trait": _sub(v, "trait", "traitName"), "method": _sub(v, "method", "methodName"), "scale": _sub(v, "scale", "scaleName"),
            "unit": _sub(v, "scale", "units"), "data_type": _sub(v, "scale", "dataType")}


def _study_entry(study: dict[str, Any]) -> dict[str, Any]:
    return {"study_id": _text(study.get("studyDbId")), "name": _text(study.get("studyName")), "study_type": _text(study.get("studyType")),
            "location_id": _text(study.get("locationDbId")), "location_name": _text(study.get("locationName")),
            "seasons": [str(s) for s in (study.get("seasons") or [])] if isinstance(study.get("seasons"), list) else [],
            "observations": {}, "observation_units": None}


def _collection_entry(result: CollectionResult, captured: list[_Response]) -> dict[str, Any]:
    """What the manifest records about one fetched collection; nothing captured means nothing in the source."""
    if not captured:
        return {"status": "not_in_source", "rows": 0, "response_sha256": None}
    return {"status": result.status, "complete": result.complete, "rows": result.returned_count, "response_sha256": captured[0].sha256,
            "warnings": list(result.warnings)}


def _seal(prepared: _Prepared, *, label: str, base_url: str, source: dict[str, Any], snapshots_dir: Path, now: datetime,
          extra: dict[str, Any] | None = None) -> tuple[str, Path, dict[str, Any]]:
    """Write snapshots/<ID>/ from the captured responses, or reuse an existing verified folder with the same ID."""
    hashes = sorted({r.sha256 for r in prepared.responses})
    snapshot_id = snapshot_id_for(label, base_url, hashes)
    target = snapshots_dir / snapshot_id
    if (target / MANIFEST).is_file():
        check = verify_snapshot(target, expected_base_url=base_url)
        if check.ok:
            return snapshot_id, target, check.manifest
        raise SnapshotError(f"{snapshot_id} exists but fails verification: {check.problems[0]}")

    manifest: dict[str, Any] = {
        "schema": SNAPSHOT_SCHEMA, "snapshot_id": snapshot_id, "created_at_utc": now.isoformat(),
        "source": source,
        "versions": code_versions(),
        "responses": [{"endpoint": r.endpoint, "params": r.params, "request_id": r.request_id, "http_status": r.http_status, "fetched_at_utc": r.fetched_at_utc,
                       "source_url_redacted": r.source_url_redacted, "content_type": r.content_type, "sha256": r.sha256, "byte_length": len(r.data),
                       "file": f"{RESPONSES_DIR}/{r.sha256}.bin", "collection": r.collection} for r in prepared.responses],
        "variables": prepared.variables,
        "studies": prepared.studies,
        "transformations": TRANSFORMATIONS,
        "notes": prepared.notes,
    }
    manifest.update(extra or {})
    target.mkdir(parents=True, exist_ok=False)
    (target / RESPONSES_DIR).mkdir()
    snap_cache = CacheStore(target / CACHE_DIR)
    seen: set[str] = set()
    for r in prepared.responses:
        if r.sha256 not in seen:
            (target / RESPONSES_DIR / f"{r.sha256}.bin").write_bytes(r.data)
            seen.add(r.sha256)
        identity = CacheIdentity(base_url=base_url, endpoint=r.endpoint, params=r.params)
        if snap_cache.get(identity) is None:
            snap_cache.put(identity, r.data, http_status=r.http_status or 200, fetched_at_utc=datetime.fromisoformat(r.fetched_at_utc) if r.fetched_at_utc else now,
                           source_url_redacted=r.source_url_redacted, content_type=r.content_type)
    cache_files = sorted(str(p.relative_to(target)).replace("\\", "/") for p in (target / CACHE_DIR).rglob("*") if p.is_file())
    manifest["cache_files"] = [{"file": f, "sha256": _sha256_file(target / f)} for f in cache_files]
    body = json.dumps(manifest, indent=2, ensure_ascii=True, allow_nan=False).encode("utf-8")
    (target / MANIFEST).write_bytes(body)
    (target / MANIFEST_HASH).write_text(_sha256(body) + "\n", encoding="utf-8")
    return snapshot_id, target, manifest


def prepare_synthetic_snapshot(*, cache_dir: Path | None = None, snapshots_dir: Path | None = None, fixture: str = "synthetic",
                               now: datetime | None = None, after_seed=None) -> tuple[str, Path, dict[str, Any]]:
    """Build (or reuse) the snapshot of the synthetic fixture from the offline cache. Returns (id, dir, manifest).

    after_seed (harness only): called with the CacheStore once the fixture is seeded, so a test can make a
    malformed reply the newest cache version and prove that its status is recorded, not hidden.
    """
    if fixture != "synthetic":
        raise SnapshotError(f"unknown fixture {fixture!r}; only 'synthetic' exists")
    now = now or datetime.now(timezone.utc)
    cache_dir = Path(cache_dir) if cache_dir else PART2_DIR / "cache" / "synthetic"
    snapshots_dir = Path(snapshots_dir) if snapshots_dir else SNAPSHOTS_DIR
    cache = CacheStore(cache_dir)
    seed_synthetic(cache, SYNTHETIC_BASE)
    if after_seed is not None:
        after_seed(cache)
    settings = Settings(base_url=SYNTHETIC_BASE, mode="offline", cache_dir=cache_dir)
    client = BrapiClient(settings, cache=cache, run_id=f"snap_prep_{now:%Y%m%dT%H%M%S}")
    prepared = _Prepared()

    mark = len(client.request_log)
    client.get("/serverinfo")
    _capture(client, prepared, None, mark)
    collections: dict[str, CollectionResult] = {}
    for name, call in (("studies", client.studies), ("variables", client.variables), ("locations", client.locations),
                       ("programs", client.programs), ("seasons", client.seasons)):
        mark = len(client.request_log)
        result = call()
        collections[name] = result
        _capture(client, prepared, result, mark)
        if not (result.complete or result.status == "empty"):
            prepared.notes.append(f"{name}: collection is {result.status}, not complete; questions needing the whole {name} list are unavailable")

    studies = collections["studies"].records if collections["studies"].complete else []
    variables = collections["variables"].records if collections["variables"].complete else []
    prepared.variables = [_variable_entry(v) for v in variables if v.get("observationVariableDbId")]
    for study in studies:
        entry = _study_entry(study)
        study_id = entry["study_id"]
        if not study_id:
            continue
        for variable in prepared.variables:
            mark = len(client.request_log)
            result = client.observations(study_id, variable["variable_id"])
            entry["observations"][variable["variable_id"]] = _collection_entry(result, _capture(client, prepared, result, mark))
        mark = len(client.request_log)
        units = client.observation_units(study_id)
        entry["observation_units"] = _collection_entry(units, _capture(client, prepared, units, mark))
        prepared.studies.append(entry)

    source = {"base_url": SYNTHETIC_BASE, "fixture": fixture, "mode": "offline", "synthetic": True, "prepared_run_id": client.run_id,
              "note": "invented teaching data from tests/fixtures/brapi; not CassavaBase, not real breeding data"}
    return _seal(prepared, label=fixture, base_url=SYNTHETIC_BASE, source=source, snapshots_dir=snapshots_dir, now=now)


# --------------------------------------------------------------------------
# Verification: every hash, every time, before any grading
# --------------------------------------------------------------------------

@dataclass
class SnapshotCheck:
    ok: bool
    problems: list[str]
    manifest: dict[str, Any]
    manifest_sha256: str | None = None


def load_manifest(snapshot_dir: Path) -> dict[str, Any]:
    return json.loads((Path(snapshot_dir) / MANIFEST).read_text(encoding="utf-8"))


def verify_snapshot(snapshot_dir: Path, *, expected_base_url: str | None = None) -> SnapshotCheck:
    """Recompute and compare every fingerprint. Any problem means the snapshot must not be graded against."""
    snapshot_dir = Path(snapshot_dir)
    problems: list[str] = []
    manifest_path = snapshot_dir / MANIFEST
    if not manifest_path.is_file():
        return SnapshotCheck(False, [f"{MANIFEST} is missing in {snapshot_dir}"], {})
    body = manifest_path.read_bytes()
    manifest_sha = _sha256(body)
    try:
        manifest = json.loads(body.decode("utf-8"))
    except ValueError as exc:
        return SnapshotCheck(False, [f"{MANIFEST} is not valid JSON: {exc}"], {})
    if manifest.get("schema") != SNAPSHOT_SCHEMA:
        problems.append(f"manifest schema is {manifest.get('schema')!r}, expected {SNAPSHOT_SCHEMA!r}")
    recorded = (snapshot_dir / MANIFEST_HASH).read_text(encoding="utf-8").strip() if (snapshot_dir / MANIFEST_HASH).is_file() else None
    if recorded is None:
        problems.append(f"{MANIFEST_HASH} is missing")
    elif recorded != manifest_sha:
        problems.append(f"{MANIFEST} has been modified: its hash {manifest_sha[:12]}... differs from {MANIFEST_HASH} {recorded[:12]}...")
    if manifest.get("snapshot_id") != snapshot_dir.name:
        problems.append(f"manifest snapshot_id {manifest.get('snapshot_id')!r} does not match the folder name {snapshot_dir.name!r}")
    base_url = (manifest.get("source") or {}).get("base_url")
    if expected_base_url is not None and base_url != expected_base_url:
        problems.append(f"source mismatch: snapshot is from {base_url!r}, expected {expected_base_url!r}")
    seen_files: set[str] = set()
    for r in manifest.get("responses", []):
        path = snapshot_dir / r.get("file", "")
        if not path.is_file():
            problems.append(f"missing response file {r.get('file')!r} ({r.get('endpoint')})")
            continue
        if r.get("file") in seen_files:
            continue
        seen_files.add(r["file"])
        actual = _sha256_file(path)
        if actual != r.get("sha256"):
            problems.append(f"hash mismatch for {r.get('file')!r} ({r.get('endpoint')}): file {actual[:12]}... manifest {str(r.get('sha256'))[:12]}...")
        if not path.name.startswith(str(r.get("sha256"))):
            problems.append(f"response file {r.get('file')!r} is not named by its hash")
    for c in manifest.get("cache_files", []):
        path = snapshot_dir / c["file"]
        if not path.is_file():
            problems.append(f"missing cache file {c['file']!r}")
        elif _sha256_file(path) != c["sha256"]:
            problems.append(f"hash mismatch for cache file {c['file']!r}")
    return SnapshotCheck(not problems, problems, manifest, manifest_sha)


# --------------------------------------------------------------------------
# Live preparation (Step 6.3): one study, one variable, two typed approvals
# --------------------------------------------------------------------------

_STUDY_WORD = re.compile(r"\b(?:study|trial)\s+([A-Za-z0-9_.:-]+)", re.IGNORECASE)
_TRAIT_WORD = re.compile(r"^\s*(?:please\s+)?(?:prepare|fetch|get|retrieve|freeze|snapshot|collect)\s+(?:the\s+)?(.+?)\s+"
                         r"(?:data|values|observations|measurements|records)\b", re.IGNORECASE)


def parse_preparation_question(question: str) -> tuple[str | None, str | None]:
    """'Prepare fresh-root-yield data for study 19ayt18highYldIB.' -> ('19ayt18highYldIB', 'fresh root yield').

    Only two things are read out of the sentence: the token after 'study'/'trial', and the phrase between the
    leading verb and 'data'. Hyphens and underscores in the phrase become spaces. Anything else is None and
    must be given with --study / --trait. What was read is printed and confirmed before any request.
    """
    study = trait = None
    match = _STUDY_WORD.search(question)
    if match:
        study = match.group(1).rstrip(".,;:!?") or None
    match = _TRAIT_WORD.search(question)
    if match:
        trait = " ".join(re.split(r"[-_\s]+", match.group(1).strip())).strip() or None
    return study, trait


def _fold(value: Any) -> str:
    return str(value).casefold() if value is not None else ""


def _resolve_unique(records: list[dict[str, Any]], *, id_field: str, name_field: str, wanted: str,
                    what: str) -> tuple[dict[str, Any] | None, str | None, list[dict[str, Any]]]:
    """Exact ID wins; else an exact (case-insensitive) name; else a UNIQUE name containing the phrase.
    Returns (record, reason when unresolved, candidates shown to the human) — the same rule the baseline uses."""
    wanted = wanted.strip()
    target = wanted.casefold()
    by_id = [r for r in records if _text(r.get(id_field)) == wanted]
    if len(by_id) == 1:
        return by_id[0], None, by_id
    exact = [r for r in records if _fold(r.get(name_field)) == target]
    if len(exact) == 1:
        return exact[0], None, exact
    if len(exact) > 1:
        return None, f"ambiguous: {len(exact)} {what}s are named {wanted!r}", exact
    partial = [r for r in records if target and target in _fold(r.get(name_field))]
    if len(partial) == 1:
        return partial[0], None, partial
    if not partial:
        return None, f"no {what} matches {wanted!r} (as exact ID, exact name or a name containing it, case-insensitive)", []
    return None, f"ambiguous: {len(partial)} {what}s contain {wanted!r}", partial


def _approval_summary(approval: FetchApproval) -> dict[str, Any]:
    return json.loads(approval.model_dump_json())


def _ask_yes(ask: Callable[[str], str], prompt: str) -> str | None:
    """The human's answer, lower-cased; None when no keyboard is attached (EOF) — which is never a yes."""
    try:
        return ask(prompt).strip().lower()
    except EOFError:
        return None


@dataclass
class LiveOutcome:
    exit_code: int
    reason: str
    snapshot_id: str | None = None
    snapshot_dir: Path | None = None
    manifest: dict[str, Any] | None = None


def prepare_live_snapshot(*, question: str, study: str | None = None, trait: str | None = None, max_fetches: int = 1, max_http_attempts: int = 20,
                          read_timeout: float | None = None, ask: Callable[[str], str] = input, log: Callable[[str], None] = print,
                          settings: Settings | None = None, client_factory: Callable[[Settings], BrapiClient] | None = None,
                          cache_dir: Path | None = None, snapshots_dir: Path | None = None, access_notes: Path = ACCESS_NOTES_PATH,
                          now: datetime | None = None) -> LiveOutcome:
    """Prepare one study/variable from the real server in two typed-approval steps. See the module docstring.

    read_timeout: seconds of silence tolerated during ONE reply (the client's default is 30). A big study can take a slow
    server longer than that to answer /observations; the human raises it on purpose, for this one command, up to
    MAX_READ_TIMEOUT. The whole-operation deadline is widened to hold at least two such waits. Nothing else changes: an
    observation request is still tried once, and the attempt ceiling still applies.
    settings/client_factory/cache_dir/snapshots_dir/access_notes/now are for the harness: tests pass a client with a fake
    transport and never reach a socket. Every real run goes through load_settings() and the ordinary client.
    """
    # 0. what the question asks for, said out loud before anything else
    parsed_study, parsed_trait = parse_preparation_question(question)
    study = (study or parsed_study or "").strip()
    trait = (trait or parsed_trait or "").strip()
    log("LIVE PREPARATION (interactive) — nothing has been requested from any server yet")
    log(f"  question      : {question}")
    log(f"  study wanted  : {study or '(not found in the question: pass --study NAME_OR_ID)'}")
    log(f"  trait wanted  : {trait or '(not found in the question: pass --trait PHRASE_OR_ID)'}")
    if not study or not trait:
        log("REFUSED: the question must name one study ('... study NAME') and one trait ('Prepare TRAIT data ...'), or pass --study and --trait")
        return LiveOutcome(EXIT_REFUSED, "study or trait not identified")
    if max_fetches < 1 or max_http_attempts < 1:
        log("REFUSED: --max-fetches and --max-http-attempts must be at least 1")
        return LiveOutcome(EXIT_REFUSED, "limits below 1")
    if read_timeout is not None and not (1 <= read_timeout <= MAX_READ_TIMEOUT):
        log(f"REFUSED: --read-timeout must be between 1 and {MAX_READ_TIMEOUT:.0f} seconds")
        return LiveOutcome(EXIT_REFUSED, "read timeout out of range")
    # 1. preconditions, checked before the first prompt: a refusal here costs nothing and asks nothing
    reviewed = access_notes_review_date(access_notes)
    if reviewed is None:
        log(f"REFUSED: {access_notes.name} has no 'Reviewed by Bidhan on: YYYY-MM-DD' line; read it and record your review first")
        return LiveOutcome(EXIT_REFUSED, "access notes not reviewed")
    settings = settings or load_settings()
    if settings.mode != "live":
        log("REFUSED: BRAPI_MODE is not 'live'. Set BRAPI_MODE=live for this one command, on purpose (the client stays offline otherwise).")
        return LiveOutcome(EXIT_REFUSED, "BRAPI_MODE is not live")
    now = now or datetime.now(timezone.utc)
    cache_dir = Path(cache_dir) if cache_dir else settings.cache_dir / LIVE_LABEL
    snapshots_dir = Path(snapshots_dir) if snapshots_dir else SNAPSHOTS_DIR
    settings = replace(settings, cache_dir=cache_dir)
    if read_timeout is not None:
        settings = replace(settings, read_timeout=float(read_timeout), operation_deadline=max(settings.operation_deadline, 2.0 * read_timeout))
    client = client_factory(settings) if client_factory else BrapiClient(settings, run_id=f"snap_live_{now:%Y%m%dT%H%M%S}")
    base_url, run_id = client.settings.base_url, client.run_id
    catalog_pages = min(CATALOG_MAX_PAGES, max_http_attempts)
    issued, expires = now - timedelta(minutes=1), now + timedelta(minutes=APPROVAL_MINUTES)

    def live_attempts() -> int:
        return sum(1 for r in client.request_log if r.origin == "live")

    # 2. STEP 1 — metadata discovery, approved by a typed yes
    discovery = FetchApproval(approval_id=f"appr_{run_id}_discovery", run_id=run_id, base_url=base_url,
                              endpoint_families=["serverinfo", "studies", "observationvariables", "locations", "seasons"],
                              max_http_attempts=max_http_attempts, max_observation_studies=0, issued_at_utc=issued, expires_at_utc=expires)
    log("")
    log("STEP 1 of 2 — METADATA DISCOVERY (public catalogs, no measurements)")
    log(f"  server        : {base_url}   (ACCESS_NOTES reviewed {reviewed})")
    log(f"  requests      : GET /serverinfo; GET /studies, /variables, /locations, /seasons — complete catalogs, at most {catalog_pages} page(s) each")
    log(f"  purpose       : resolve study {study!r} and trait {trait!r} by literal case-insensitive match; the application resolves names only against a complete catalog")
    log(f"  ceiling       : {max_http_attempts} HTTP attempt(s) for the WHOLE preparation (both steps, retries included); approval valid {APPROVAL_MINUTES} minutes")
    log("  never         : observations, any study's measurements, or an endpoint outside these five")
    answer = _ask_yes(ask, "Type yes to approve STEP 1 (anything else cancels): ")
    if answer is None:
        log(NO_TERMINAL)
        return LiveOutcome(EXIT_NO_TERMINAL, "no interactive terminal")
    if answer != "yes":
        log("CANCELLED: nothing was sent")
        return LiveOutcome(EXIT_CANCELLED, "cancelled before step 1")

    prepared = _Prepared()
    mark = len(client.request_log)
    try:
        client.get("/serverinfo", approval=discovery)
    except BrapiClientError as exc:
        prepared.notes.append(f"/serverinfo: {exc}")
    _capture(client, prepared, None, mark)
    catalogs: dict[str, CollectionResult] = {}
    for name, call in (("studies", client.studies), ("variables", client.variables), ("locations", client.locations), ("seasons", client.seasons)):
        mark = len(client.request_log)
        result = call(approval=discovery, max_pages=catalog_pages)
        catalogs[name] = result
        _capture(client, prepared, result, mark)
        total = result.reported_total if result.reported_total is not None else "?"
        log(f"  {name:<12}: {result.status} — {result.returned_count} of {total} record(s), {len(result.request_ids)} request(s)")
        for warning in result.warnings[:3]:
            log(f"                 note: {warning}")
        if not result.complete:
            prepared.notes.append(f"{name}: collection is {result.status}, not complete; questions needing the whole {name} list are unavailable")
    log(f"  attempts used : {live_attempts()} of {max_http_attempts}")
    for name in ("studies", "variables"):
        if not catalogs[name].complete:
            log(f"STOPPED: the {name} catalog is {catalogs[name].status}, not complete, and the application can resolve names only against a complete "
                f"catalog. Read the notes above; raise --max-http-attempts on purpose if a page budget was the cause, or stop here.")
            return LiveOutcome(EXIT_STOPPED, f"{name} catalog not complete")

    study_rec, why, shown = _resolve_unique(catalogs["studies"].records, id_field="studyDbId", name_field="studyName", wanted=study, what="study")
    if study_rec is None:
        log(f"STOPPED: {why}")
        for r in shown[:10]:
            log(f"    candidate study {_text(r.get('studyDbId'))!r}: {_text(r.get('studyName'))!r} ({_text(r.get('studyType'))}, {_text(r.get('locationName'))})")
        log("  Re-run with --study giving the exact ID or exact name.")
        return LiveOutcome(EXIT_STOPPED, why or "study unresolved")
    var_rec, why, shown = _resolve_unique(catalogs["variables"].records, id_field="observationVariableDbId", name_field="observationVariableName",
                                          wanted=trait, what="variable")
    if var_rec is None:
        log(f"STOPPED: {why}")
        for v in shown[:10]:
            entry = _variable_entry(v)
            log(f"    candidate variable {entry['variable_id']!r}: {entry['name']!r} (trait {entry['trait']!r}, unit {entry['unit']!r})")
        log("  Re-run with --trait giving the exact ID or exact name.")
        return LiveOutcome(EXIT_STOPPED, why or "variable unresolved")
    study_entry, variable_entry = _study_entry(study_rec), _variable_entry(var_rec)
    study_id, variable_id = study_entry["study_id"], variable_entry["variable_id"]
    if not study_id or not variable_id:
        log("STOPPED: the resolved study or variable record has no ID")
        return LiveOutcome(EXIT_STOPPED, "resolved record without ID")

    # 3. STEP 2 — observation retrieval for exactly the resolved pair, approved by a second typed yes
    log("")
    log("STEP 2 of 2 — OBSERVATION RETRIEVAL for exactly this pair")
    log(f"  study         : {study_id}  {study_entry['name']!r}  type {study_entry['study_type']!r}  location {study_entry['location_name']!r}  seasons {study_entry['seasons']}")
    log(f"  variable      : {variable_id}  {variable_entry['name']!r}  trait {variable_entry['trait']!r}  method {variable_entry['method']!r}  "
        f"scale {variable_entry['scale']!r}  unit {variable_entry['unit']!r}")
    log(f"  requests      : GET /observations?studyDbId={study_id}&observationVariableDbId={variable_id} and GET /observationunits?studyDbId={study_id}, "
        f"at most {max_fetches} page(s) of {client.settings.page_size} each")
    log(f"  ceiling       : {live_attempts()} of {max_http_attempts} attempt(s) used so far; one study only; approval valid {APPROVAL_MINUTES} minutes")
    log(f"  patience      : up to {client.settings.read_timeout:.0f} s of silence per reply; an observation request is tried once")
    answer = _ask_yes(ask, "Type yes to approve STEP 2 (anything else cancels): ")
    if answer is None:
        log(NO_TERMINAL)
        return LiveOutcome(EXIT_NO_TERMINAL, "no interactive terminal")
    if answer != "yes":
        log("CANCELLED: no observation request was sent; the catalog replies stay in the live cache, no snapshot was written")
        return LiveOutcome(EXIT_CANCELLED, "cancelled before step 2")
    retrieval = FetchApproval(approval_id=f"appr_{run_id}_observations", run_id=run_id, base_url=base_url,
                              endpoint_families=["observations", "observationunits"], observation_study_ids=[study_id],
                              observation_variable_ids=[variable_id], max_http_attempts=max_http_attempts, max_observation_studies=1,
                              issued_at_utc=issued, expires_at_utc=expires)
    try:
        mark = len(client.request_log)
        observations = client.observations(study_id, variable_id, approval=retrieval, paging=True, max_pages=max_fetches)
        obs_captured = _capture(client, prepared, observations, mark)
        mark = len(client.request_log)
        units = client.observation_units(study_id, approval=retrieval, paging=True, max_pages=max_fetches)
        units_captured = _capture(client, prepared, units, mark)
    except BrapiClientError as exc:                                   # e.g. an ID the client will not put in a URL
        log(f"FAILED: {exc}")
        return LiveOutcome(EXIT_FAILED, str(exc))
    for name, result in (("observations", observations), ("units", units)):
        total = result.reported_total if result.reported_total is not None else "?"
        log(f"  {name:<12}: {result.status} — {result.returned_count} of {total} row(s), {len(result.request_ids)} request(s)")
        for warning in result.warnings[:3]:
            log(f"                 note: {warning}")
    log(f"  attempts used : {live_attempts()} of {max_http_attempts}")
    if not obs_captured:
        log("FAILED: no observation reply was obtained; nothing is sealed (the request log above says why)")
        return LiveOutcome(EXIT_FAILED, "observations failed")
    if not observations.complete:
        log(f"  WARNING: the observations are {observations.status}, not complete; they are sealed WITH that status and the baseline will "
            "treat questions needing the whole set as unavailable")
    study_entry["observations"][variable_id] = _collection_entry(observations, obs_captured)
    study_entry["observation_units"] = _collection_entry(units, units_captured)
    prepared.studies.append(study_entry)
    prepared.variables = [_variable_entry(v) for v in catalogs["variables"].records if v.get("observationVariableDbId")]
    prepared.notes.append("studies[] lists only the study whose observations were retrieved; the complete study catalog is in the /studies responses")

    # 4. seal, verify, report
    host = urlsplit(base_url).hostname or base_url
    source = {"base_url": base_url, "fixture": None, "mode": "live", "synthetic": False, "prepared_run_id": run_id, "question": question,
              "resolved": {"study_id": study_id, "study_name": study_entry["name"], "variable_id": variable_id, "variable_name": variable_entry["name"],
                           "trait": variable_entry["trait"], "unit": variable_entry["unit"]},
              "access_notes_reviewed": reviewed, "approvals": [_approval_summary(discovery), _approval_summary(retrieval)],
              "note": f"real data from {host}; kept under git-ignored folders; ACCESS_NOTES.md governs its use and redistribution"}
    request_log = [json.loads(r.model_dump_json()) for r in client.request_log]
    try:
        snapshot_id, target, manifest = _seal(prepared, label=LIVE_LABEL, base_url=base_url, source=source, snapshots_dir=snapshots_dir, now=now,
                                              extra={"request_log": request_log})
    except SnapshotError as exc:
        log(f"FAILED: {exc}")
        return LiveOutcome(EXIT_FAILED, str(exc))
    check = verify_snapshot(target, expected_base_url=base_url)
    log("")
    _print_summary(snapshot_id, target, manifest, check, log)
    return LiveOutcome(EXIT_OK if check.ok else EXIT_FAILED, "sealed and verified" if check.ok else "sealed but verification failed",
                       snapshot_id=snapshot_id, snapshot_dir=target, manifest=manifest)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _print_summary(snapshot_id: str, target: Path, manifest: dict[str, Any], check: SnapshotCheck, log: Callable[[str], None] = print) -> None:
    log(f"snapshot_id: {snapshot_id}")
    log(f"snapshot_dir: {target}")
    log(f"responses: {len(manifest['responses'])}  studies: {len(manifest['studies'])}  variables: {len(manifest['variables'])}  verified: {check.ok}")
    for problem in check.problems:
        log(f"  problem: {problem}")
    for study in manifest["studies"]:
        obs = ", ".join(f"{v}:{d['status']}({d['rows']})" for v, d in study["observations"].items())
        log(f"  {study['study_id']}: observations [{obs}]  units {study['observation_units']['status']}({study['observation_units']['rows']})")
    for note in manifest["notes"]:
        log(f"  note: {note}")


def main(argv: list[str] | None = None, ask: Callable[[str], str] = input, *, client_factory: Callable[[Settings], BrapiClient] | None = None,
         access_notes: Path = ACCESS_NOTES_PATH) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.prepare_snapshot", description="Freeze an evaluation snapshot with verified hashes.")
    parser.add_argument("--fixture", default=None, help="'synthetic': build from the offline synthetic cache")
    parser.add_argument("--live", action="store_true", help="interactive live preparation of ONE study/variable in two typed-approval steps (Step 6.3)")
    parser.add_argument("--question", default=None, help="with --live: the preparation sentence, e.g. 'Prepare fresh-root-yield data for study NAME.'")
    parser.add_argument("--study", default=None, help="with --live: the study's exact ID or name when the sentence does not say it plainly")
    parser.add_argument("--trait", default=None, help="with --live: the variable's exact ID, exact name or a phrase its name contains")
    parser.add_argument("--max-fetches", type=int, default=1, help="with --live: pages of observations (and of the plot roster) for the one resolved pair")
    parser.add_argument("--max-http-attempts", type=int, default=20, help="with --live: one ceiling on HTTP attempts for the whole preparation")
    parser.add_argument("--read-timeout", type=float, default=None,
                        help=f"with --live: seconds of silence tolerated during one reply (client default 30, at most {MAX_READ_TIMEOUT:.0f}); "
                             "raise it on purpose when a big study makes the server slow")
    parser.add_argument("--auto", action="store_true", help="refused with --live: live preparation is interactive by design")
    parser.add_argument("--cache-dir", default=None, help="harness only: cache folder (fixture to read, or where live replies are stored)")
    parser.add_argument("--snapshots-dir", default=None, help="harness only: where snapshots/<ID> goes")
    args = parser.parse_args(argv)
    if args.live:
        if args.auto:
            print("error: --live is interactive and incompatible with --auto", file=sys.stderr)
            return 2
        if not args.question or not args.question.strip():
            print("error: --live needs --question TEXT", file=sys.stderr)
            return 2
        if args.max_fetches < 1 or args.max_http_attempts < 1:
            print("error: --max-fetches and --max-http-attempts must be at least 1", file=sys.stderr)
            return 2
        if args.read_timeout is not None and not (1 <= args.read_timeout <= MAX_READ_TIMEOUT):
            print(f"error: --read-timeout must be between 1 and {MAX_READ_TIMEOUT:.0f} seconds", file=sys.stderr)
            return 2
        outcome = prepare_live_snapshot(question=args.question, study=args.study, trait=args.trait, max_fetches=args.max_fetches,
                                        max_http_attempts=args.max_http_attempts, read_timeout=args.read_timeout, ask=ask, client_factory=client_factory,
                                        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
                                        snapshots_dir=Path(args.snapshots_dir) if args.snapshots_dir else None, access_notes=access_notes)
        return outcome.exit_code
    if args.fixture is None:
        print("error: choose --fixture synthetic or --live --question TEXT", file=sys.stderr)
        return 2
    try:
        snapshot_id, target, manifest = prepare_synthetic_snapshot(cache_dir=Path(args.cache_dir) if args.cache_dir else None,
                                                                   snapshots_dir=Path(args.snapshots_dir) if args.snapshots_dir else None, fixture=args.fixture)
    except SnapshotError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    check = verify_snapshot(target, expected_base_url=manifest["source"]["base_url"])
    _print_summary(snapshot_id, target, manifest, check)
    return 0 if check.ok else 1


if __name__ == "__main__":
    sys.exit(main())
