"""Human-readable, fingerprinted review of managed tables before analysis."""
from __future__ import annotations

import html
import json
from typing import Any

from artifacts import ArtifactError
from contracts import ArtifactManifest, Plan, RetrievalReport


def _cell(value: Any, limit: int = 180) -> str:
    text = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value if value is not None else "")
    text = " ".join(text.split())
    if len(text) > limit:
        text = text[:limit] + "…"
    # Escape Markdown too: an untrusted cell must not become an image/link or
    # change the human review's formatting. Entities render as ordinary text.
    markdown = "\\|`[]()*_!#~"
    return "".join(f"&#{ord(char)};" if char in markdown else html.escape(char) for char in text)


def _fingerprint(manifest: ArtifactManifest) -> dict[str, Any]:
    return {"artifact_id": manifest.artifact_id, "sha256": manifest.sha256,
            "row_count": manifest.row_count, "complete": manifest.complete}


def review_snapshot(ctx: Any, report: RetrievalReport, plan: Plan, requests: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Load/hash-check each complete table; show bounded samples, never model-generated evidence."""
    if report.status != "completed" or not report.complete or not report.artifacts:
        raise ArtifactError("incomplete_data", "no complete retrieved tables are available for review")
    lines = ["## Review the retrieved data", "", "Analysis has not started. Check that these data match your question.", "",
             f"**Source:** {_cell(plan.scope.base_url, 500)}",
             f"**Studies:** {_cell(', '.join(plan.scope.study_ids) or '(catalog question)', 1500)}",
             f"**Traits / variables:** {_cell(', '.join(plan.scope.variable_ids) or '(none)', 1500)}",
             f"**HTTP attempts:** {sum(r.get('origin') == 'live' for r in requests)} · "
             f"**Cached responses:** {sum(r.get('origin') == 'cache' for r in requests)}", ""]
    fingerprints = []
    preferred = ["studyDbId", "germplasmDbId", "germplasmName", "observationVariableDbId", "observationUnitDbId",
                 "value", "unit", "locationDbId", "locationName", "seasonDbId", "year"]
    for declared in report.artifacts:
        if not declared.complete:
            raise ArtifactError("incomplete_data", "an incomplete table cannot be approved for analysis")
        manifest, rows = ctx.registry.load(declared.artifact_id)
        if not manifest.complete or _fingerprint(manifest) != _fingerprint(declared):
            raise ArtifactError("hash_mismatch", "retrieved table changed before review")
        fingerprints.append(_fingerprint(manifest))
        lines += [f"### {_cell(manifest.kind)} · {manifest.row_count} rows", "",
                  f"Table: `{manifest.artifact_id}` · complete collection", ""]
        columns = list(manifest.columns)
        shown = [key for key in preferred if key in columns]
        shown = (shown + [key for key in columns if key not in shown])[:8]
        if shown:
            lines += ["| " + " | ".join(_cell(key) for key in shown) + " |", "| " + " | ".join("---" for _ in shown) + " |"]
            lines += ["| " + " | ".join(_cell(row.get(key)) for key in shown) + " |" for row in rows[:5]]
        lines += ["", f"Preview: {min(5, len(rows))} of {len(rows)} rows; {len(shown)} of {len(columns)} columns. "
                  "Cells longer than 180 characters are shortened. This sample is not the full table.", ""]
    measurements = getattr(ctx, "variable_measurements", {})
    for variable in plan.scope.variable_ids:
        meta = measurements.get(variable)
        if meta:
            lines.append(f"- {_cell(variable)}: {_cell(meta.trait or '(trait not stated)')} · unit {_cell(meta.unit or '(not stated)')}")
    warnings = [warning for warning in report.warnings if not warning.lower().startswith("model note:")]
    if warnings:
        lines += ["", "**Data limitations:**", *[f"- {_cell(warning, 500)}" for warning in warnings]]
    lines += ["", "Continuing approves these saved tables for analysis only. It does not accept an answer or authorize additional retrieval."]
    return "\n".join(lines), fingerprints


def verify_reviewed_tables(ctx: Any, report: RetrievalReport, expected: list[dict[str, Any]]) -> None:
    """Recheck the exact approved byte set after the person has reviewed it."""
    if report.status != "completed" or not report.complete or not report.artifacts:
        raise ArtifactError("incomplete_data", "retrieved data is no longer complete; analysis is blocked")
    if [a.artifact_id for a in report.artifacts] != [a["artifact_id"] for a in expected]:
        raise ArtifactError("invalid_handle", "the set of retrieved tables changed during review")
    for declared, fingerprint in zip(report.artifacts, expected):
        if not declared.complete:
            raise ArtifactError("incomplete_data", "a retrieved table is no longer complete; analysis is blocked")
        if _fingerprint(declared) != fingerprint:
            raise ArtifactError("hash_mismatch", "the retrieved table declaration changed during review; analysis is blocked")
        manifest, _ = ctx.registry.load(fingerprint["artifact_id"])
        if _fingerprint(manifest) != fingerprint:
            raise ArtifactError("hash_mismatch", "a retrieved table changed during review; analysis is blocked")
