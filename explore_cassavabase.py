"""
explore_cassavabase.py — a readable proof that the data foundation works, with NO AI
and, by default, NO network.

Two modes:

1. Offline proof (safe, run any time):
       python -m explore_cassavabase --offline --fixture synthetic
   Loads the SYNTHETIC fixtures (tests/fixtures/brapi/*.json) into a cache that
   pretends to be a made-up server (https://synthetic.invalid/brapi/v2), then runs
   the real client in offline mode against that cache. It prints the server
   identity, record counts, completeness, request IDs, cache/live request counts
   and evidence paths, and writes a JSON report under out/. Running it twice gives
   the same data. The address synthetic.invalid can never resolve, and the
   synthetic cache lives in its own folder, so this data can never be confused
   with CassavaBase.

2. Controlled live probe (NOT run by default, refuses without explicit approval):
       python -m explore_cassavabase --live-probe --endpoint /serverinfo --max-attempts 1 --i-reviewed-access-notes
   Refuses unless ACCESS_NOTES.md records a review date, BRAPI_MODE=live is set,
   the endpoint is exactly /serverinfo, an attempt budget is given, and you type
   "yes" at the prompt. It then sends ONE GET /serverinfo through the normal client
   (approval checked, no follow-up calls), saves the reply and the request log, and
   prints the endpoints the server advertises. A successful probe proves reachability
   only — not data-use permission, not that observations work.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from brapi_client import PART2_DIR, BrapiClient, BrapiClientError, Settings, load_settings
from cache_store import CacheEntry, CacheIdentity, CacheStore, sha256_bytes
from contracts import CollectionResult, FetchApproval, RequestRecord

__all__ = [
    "SYNTHETIC_BASE", "FIXTURE_DIR", "NORMAL_FIXTURES", "MALFORMED_FIXTURES", "ACCESS_NOTES_PATH",
    "load_fixture", "seed_fixture", "seed_synthetic", "run_offline_proof", "render_report",
    "access_notes_review_date", "live_probe", "main",
]

SYNTHETIC_BASE = "https://synthetic.invalid/brapi/v2"      # .invalid is reserved: it can never resolve
FIXTURE_DIR = PART2_DIR / "tests" / "fixtures" / "brapi"
ACCESS_NOTES_PATH = PART2_DIR / "ACCESS_NOTES.md"
FIXTURE_FETCHED_AT = datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone.utc)   # a fixed, fake fetch time

NORMAL_FIXTURES = [
    "synthetic_serverinfo.json",
    "synthetic_studies.json",
    "synthetic_study_S1.json",          # GET /studies/S1 — the same record as the S1 row of the list
    "synthetic_variables.json",
    "synthetic_observations_S1_V1.json",
    "synthetic_observationunits_S1.json",
    "synthetic_locations.json",
    "synthetic_programs.json",
    "synthetic_seasons.json",
]
# Separate broken cases for tests. They are never seeded together with the normal set.
MALFORMED_FIXTURES = {
    "null_result_with_count": "malformed_observations_null_result_with_count.json",
    "no_pagination": "malformed_observations_no_pagination.json",
    "duplicate_id": "malformed_observations_duplicate_id.json",
}

# What the synthetic proof must show. These are properties of OUR fixture, not values from any guide.
EXPECTED_SYNTHETIC = {
    "studies": ("complete", 1),
    "variables": ("complete", 2),
    "find_variables[fresh root]": ("complete", 1),
    "observations[S1,V1]": ("complete", 6),
    "observation_units[S1]": ("complete", 6),
    "locations": ("complete", 1),
    "programs": ("complete", 1),
    "seasons": ("complete", 1),
}


# --------------------------------------------------------------------------
# Fixtures -> cache
# --------------------------------------------------------------------------

def load_fixture(name: str, fixture_dir: Path = FIXTURE_DIR) -> tuple[dict[str, Any], bytes]:
    """Return (parsed JSON, exact bytes). Refuses a fixture that is not labelled SYNTHETIC."""
    path = fixture_dir / name
    body = path.read_bytes()
    parsed = json.loads(body.decode("utf-8"))
    if parsed.get("_synthetic") is not True or "SYNTHETIC" not in str(parsed.get("_note", "")):
        raise ValueError(f"{name} is not labelled SYNTHETIC; refusing to use it as a fixture")
    if "_request" not in parsed or "metadata" not in parsed or "result" not in parsed:
        raise ValueError(f"{name} lacks _request/metadata/result")
    return parsed, body


def seed_fixture(cache: CacheStore, base_url: str, name: str, fixture_dir: Path = FIXTURE_DIR) -> CacheEntry:
    """Put one fixture into the cache under the request it stands for. Idempotent: same bytes -> reuse."""
    parsed, body = load_fixture(name, fixture_dir)
    request = parsed["_request"]
    identity = CacheIdentity(base_url=base_url, endpoint=request["endpoint"], params=dict(request.get("params", {})))
    existing = cache.get(identity)
    if existing is not None and existing[0].response_sha256 == sha256_bytes(body):
        return existing[0]
    query = "&".join(f"{k}={identity.params[k]}" for k in sorted(identity.params))
    redacted = f"{base_url}{identity.endpoint}" + (f"?{query}" if query else "")
    return cache.put(identity, body, http_status=200, fetched_at_utc=FIXTURE_FETCHED_AT,
                     source_url_redacted=redacted, content_type="application/json")


def seed_synthetic(cache: CacheStore, base_url: str = SYNTHETIC_BASE,
                   names: list[str] | None = None, fixture_dir: Path = FIXTURE_DIR) -> list[CacheEntry]:
    return [seed_fixture(cache, base_url, name, fixture_dir) for name in (names or NORMAL_FIXTURES)]


# --------------------------------------------------------------------------
# Offline proof
# --------------------------------------------------------------------------

def _collection_summary(result: CollectionResult, id_field: str | None) -> dict[str, Any]:
    ids = [str(r.get(id_field)) for r in result.records] if id_field else []
    return {
        "returned": result.returned_count,
        "reported": result.reported_total,
        "status": result.status,
        "complete": result.complete,
        "ids": ids[:20],
        "warnings": list(result.warnings),
        "requests_used": len(result.request_ids),   # the IDs themselves are listed under report["requests"]
    }


def run_offline_proof(*, cache_dir: Path, out_dir: Path, fixture: str = "synthetic",
                      run_id: str | None = None, fixture_dir: Path = FIXTURE_DIR) -> dict[str, Any]:
    """Seed the synthetic cache, run the client offline, and return the report (also written to out_dir)."""
    if fixture != "synthetic":
        raise ValueError(f"unknown fixture set {fixture!r}; only 'synthetic' exists")
    started = datetime.now(timezone.utc)
    run_id = run_id or f"explore_{started:%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:6]}"
    cache_dir = Path(cache_dir)
    cache = CacheStore(cache_dir)
    seeded = seed_synthetic(cache, SYNTHETIC_BASE, fixture_dir=fixture_dir)

    settings = Settings(base_url=SYNTHETIC_BASE, mode="offline", cache_dir=cache_dir)
    client = BrapiClient(settings, cache=cache, run_id=run_id)

    data: dict[str, Any] = {"collections": {}}
    serverinfo, _ = client.get("/serverinfo")
    info = serverinfo["result"] or {}
    data["serverinfo"] = {
        "serverName": info.get("serverName"),
        "organizationName": info.get("organizationName"),
        "advertised_calls": sorted(c.get("service", "?") for c in info.get("calls", []) if isinstance(c, dict)),
    }

    collections = {
        "studies": (client.studies(), "studyDbId"),
        "variables": (client.variables(), "observationVariableDbId"),
        "find_variables[fresh root]": (client.find_variables("fresh root"), "observationVariableDbId"),
        "observations[S1,V1]": (client.observations("S1", "V1"), "observationDbId"),
        "observation_units[S1]": (client.observation_units("S1"), "observationUnitDbId"),
        "locations": (client.locations(), "locationDbId"),
        "programs": (client.programs(), "programDbId"),
        "seasons": (client.seasons(), "seasonDbId"),
    }
    for name, (result, id_field) in collections.items():
        data["collections"][name] = _collection_summary(result, id_field)

    # One single-record endpoint too: GET /studies/S1 must agree with the S1 row of the list.
    study_record, _ = client.study("S1")
    listing_rows = collections["studies"][0].records
    data["study_S1"] = {"studyDbId": study_record.get("studyDbId"), "studyName": study_record.get("studyName"),
                        "same_as_listing_row": bool(listing_rows) and study_record == listing_rows[0]}

    obs = collections["observations[S1,V1]"][0]
    units = collections["observation_units[S1]"][0]
    data["teaching_fixture"] = {
        "study": "S1", "variable": "V1", "variable_name": "fresh root yield", "unit": "t/ha",
        "plots_in_order": [r.get("observationUnitDbId") for r in obs.records],
        "clones_in_order": [r.get("germplasmName") for r in obs.records],
        "raw_values_in_order": [r.get("value") for r in obs.records],
        "unit_roster": [r.get("observationUnitDbId") for r in units.records],
        "timepoints": sorted({r.get("observationTimeStamp") for r in obs.records}),
    }
    data["response_hashes"] = {r.endpoint + ("?" + "&".join(f"{k}={v}" for k, v in sorted(r.params.items())) if r.params else ""):
                               r.response_sha256 for r in client.request_log}

    origins = Counter(r.origin for r in client.request_log)
    checks = []
    for name, (status, count) in EXPECTED_SYNTHETIC.items():
        actual = data["collections"][name]
        checks.append({"check": name, "expected": f"{status}, {count} records",
                       "actual": f"{actual['status']}, {actual['returned']} records",
                       "pass": actual["status"] == status and actual["returned"] == count})
    checks.append({"check": "study[S1] equals its listing row", "expected": "True",
                   "actual": str(data["study_S1"]["same_as_listing_row"]), "pass": data["study_S1"]["same_as_listing_row"] is True})
    checks.append({"check": "live requests", "expected": "0", "actual": str(origins.get("live", 0)),
                   "pass": origins.get("live", 0) == 0})
    checks.append({"check": "serverinfo is the synthetic server", "expected": "SYNTHETIC fixture server",
                   "actual": str(data["serverinfo"]["serverName"]),
                   "pass": data["serverinfo"]["serverName"] == "SYNTHETIC fixture server"})

    out_dir = Path(out_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "explore_report.json"
    report = {
        "identity": {"base_url": SYNTHETIC_BASE, "mode": "offline", "fixture": fixture, "run_id": run_id,
                     "started_utc": started.isoformat(), "cache_dir": str(cache_dir), "synthetic": True},
        "data": data,
        "requests": {"cache": origins.get("cache", 0), "live": origins.get("live", 0),
                     "request_ids": [r.request_id for r in client.request_log]},
        "evidence": {"cache_paths": sorted(e.relative_path for e in seeded), "report_path": str(report_path)},
        "checks": checks,
        "all_passed": all(c["pass"] for c in checks),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "request_log.json").write_text(
        json.dumps([json.loads(r.model_dump_json()) for r in client.request_log], indent=2), encoding="utf-8")
    return report


def render_report(report: dict[str, Any]) -> str:
    lines = []
    ident = report["identity"]
    lines.append(f"SYNTHETIC offline proof  run_id={ident['run_id']}")
    lines.append(f"  base identity : {ident['base_url']}  mode={ident['mode']}  fixture={ident['fixture']}")
    lines.append(f"  server        : {report['data']['serverinfo']['serverName']}  "
                 f"({len(report['data']['serverinfo']['advertised_calls'])} advertised calls)")
    lines.append("  collections   :")
    for name, c in report["data"]["collections"].items():
        flag = "complete" if c["complete"] else "NOT complete"
        lines.append(f"    {name:<28} {c['returned']:>4} returned / {c['reported']!s:>4} reported  {c['status']:<10} {flag}")
    st = report["data"]["study_S1"]
    lines.append(f"  single study  : {st['studyDbId']} {st['studyName']!r}  same as listing row: {st['same_as_listing_row']}")
    tf = report["data"]["teaching_fixture"]
    lines.append(f"  teaching rows : plots {tf['plots_in_order']}")
    lines.append(f"                  values {tf['raw_values_in_order']}  unit {tf['unit']}")
    req = report["requests"]
    lines.append(f"  requests      : cache={req['cache']}  live={req['live']}  ids={len(req['request_ids'])}")
    lines.append(f"  evidence      : {len(report['evidence']['cache_paths'])} cache entries under {ident['cache_dir']}")
    lines.append(f"                  report {report['evidence']['report_path']}")
    lines.append("  checks        :")
    for c in report["checks"]:
        lines.append(f"    [{'PASS' if c['pass'] else 'FAIL'}] {c['check']}: expected {c['expected']}; got {c['actual']}")
    lines.append("ALL CHECKS PASSED" if report["all_passed"] else "SOME CHECKS FAILED")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Controlled live probe
# --------------------------------------------------------------------------

_REVIEW_RE = re.compile(r"^Reviewed by Bidhan on:\s*(\d{4}-\d{2}-\d{2})\s*$", re.MULTILINE)


def access_notes_review_date(path: Path = ACCESS_NOTES_PATH) -> str | None:
    """The date on the 'Reviewed by Bidhan on: YYYY-MM-DD' line, or None if not reviewed."""
    if not path.is_file():
        return None
    match = _REVIEW_RE.search(path.read_text(encoding="utf-8"))
    return match.group(1) if match else None


def live_probe(
    *,
    endpoint: str | None,
    max_attempts: int | None,
    reviewed_flag: bool,
    out_dir: Path,
    ask: Callable[[str], str] = input,
    client_factory: Callable[[int], BrapiClient] | None = None,
    access_notes: Path = ACCESS_NOTES_PATH,
    log: Callable[[str], None] = print,
) -> int:
    """Send at most ONE approved GET /serverinfo. Returns an exit code; 0 only after a saved reply."""
    if endpoint != "/serverinfo":
        log("REFUSED: the initial probe permits only --endpoint /serverinfo")
        return 2
    if max_attempts is None or not (1 <= max_attempts <= 3):
        log("REFUSED: give --max-attempts N with N between 1 and 3 (the HTTP attempt budget for this one request)")
        return 2
    review_date = access_notes_review_date(access_notes)
    if not reviewed_flag or review_date is None:
        log("REFUSED: read ACCESS_NOTES.md, write the line 'Reviewed by Bidhan on: YYYY-MM-DD', and pass --i-reviewed-access-notes")
        return 2

    if client_factory is None:
        settings = dataclasses.replace(load_settings(), max_attempts=max_attempts)
        client = BrapiClient(settings)
    else:
        client = client_factory(max_attempts)
    if client.settings.mode != "live":
        log("REFUSED: BRAPI_MODE is not 'live'. Set BRAPI_MODE=live for this one run, on purpose.")
        return 2

    now = datetime.now(timezone.utc)
    approval = FetchApproval(
        approval_id=f"appr_probe_{now:%Y%m%dT%H%M%S}", run_id=client.run_id, base_url=client.settings.base_url,
        endpoint_families=["serverinfo"], max_http_attempts=max_attempts, max_observation_studies=0,
        issued_at_utc=now, expires_at_utc=now + timedelta(minutes=10),
    )
    log("PLAN: exactly one GET " + client.settings.base_url + "/serverinfo")
    log(f"      attempt budget {max_attempts}, approval expires in 10 minutes, ACCESS_NOTES reviewed {review_date}")
    log("      no other endpoint will be called; the reply and request log are saved under out/")
    try:
        answer = ask("Type yes to send this one request (anything else cancels): ")
    except EOFError:
        log("CANCELLED: no interactive terminal to receive your approval. Run this command yourself in a terminal; "
            "the helper must not answer for you.")
        return 3
    if answer.strip().lower() != "yes":
        log("CANCELLED: nothing was sent")
        return 3

    out_dir = Path(out_dir) / f"probe_{client.run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        envelope, record = client.get("/serverinfo", approval=approval)
    except BrapiClientError as exc:
        (out_dir / "request_log.json").write_text(
            json.dumps([json.loads(r.model_dump_json()) for r in client.request_log], indent=2), encoding="utf-8")
        log(f"PROBE FAILED: {exc}  (attempts made: {len(client.request_log)}; log saved to {out_dir})")
        return 1
    (out_dir / "serverinfo.json").write_text(json.dumps(envelope, indent=2), encoding="utf-8")
    (out_dir / "request_log.json").write_text(
        json.dumps([json.loads(r.model_dump_json()) for r in client.request_log], indent=2), encoding="utf-8")
    info = envelope.get("result") or {}
    calls = info.get("calls") if isinstance(info, dict) else None
    services = sorted({c.get("service", "?") for c in calls if isinstance(c, dict)}) if isinstance(calls, list) else []
    versions = sorted({v for c in (calls or []) if isinstance(c, dict) for v in (c.get("versions") or [])})
    log(f"PROBE OK: HTTP {record.http_status}, origin={record.origin}, response sha256 {record.response_sha256}")
    log(f"  server: {info.get('serverName')!r}  organization: {info.get('organizationName')!r}")
    log(f"  advertised versions: {versions or 'not stated'}")
    log(f"  advertised calls ({len(services)}): {services}")
    log("  Only the calls listed above are known to exist; nothing else is assumed.")
    log(f"  saved: {out_dir / 'serverinfo.json'} and request_log.json")
    log("  This proves reachability only — not data-use permission, not that observations work.")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m explore_cassavabase", description=__doc__.split("\n\n")[0])
    parser.add_argument("--offline", action="store_true", help="offline proof against the synthetic cache (default mode)")
    parser.add_argument("--fixture", default=None, help="fixture set to use; only 'synthetic' exists")
    parser.add_argument("--cache-dir", default=None, help="cache folder (default: part2/cache/synthetic)")
    parser.add_argument("--out-dir", default=None, help="report folder (default: part2/out)")
    parser.add_argument("--json", action="store_true", help="print the JSON report instead of text")
    parser.add_argument("--live-probe", action="store_true", help="controlled live probe (refuses without approval)")
    parser.add_argument("--endpoint", default=None, help="live probe: must be /serverinfo")
    parser.add_argument("--max-attempts", type=int, default=None, help="live probe: HTTP attempt budget (1-3)")
    parser.add_argument("--i-reviewed-access-notes", action="store_true", help="live probe: you read ACCESS_NOTES.md")
    return parser


def main(argv: list[str] | None = None, *, ask: Callable[[str], str] = input,
         client_factory: Callable[[int], BrapiClient] | None = None, log: Callable[[str], None] = print) -> int:
    args = _build_parser().parse_args(argv)
    out_dir = Path(args.out_dir) if args.out_dir else PART2_DIR / "out"

    if args.live_probe:
        if args.offline or args.fixture:
            log("REFUSED: --live-probe cannot be combined with --offline/--fixture")
            return 2
        return live_probe(endpoint=args.endpoint, max_attempts=args.max_attempts,
                          reviewed_flag=args.i_reviewed_access_notes, out_dir=out_dir, ask=ask,
                          client_factory=client_factory, log=log)

    if args.fixture is None:
        log("Nothing to do: use --offline --fixture synthetic for the offline proof, or --live-probe (read the docstring first).")
        return 2
    cache_dir = Path(args.cache_dir) if args.cache_dir else PART2_DIR / "cache" / "synthetic"
    try:
        report = run_offline_proof(cache_dir=cache_dir, out_dir=out_dir, fixture=args.fixture)
    except (ValueError, BrapiClientError) as exc:
        log(f"FAILED: {exc}")
        return 1
    log(json.dumps(report, indent=2, sort_keys=True) if args.json else render_report(report))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
