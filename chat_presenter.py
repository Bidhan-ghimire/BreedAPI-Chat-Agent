"""Small, evidence-backed chat views. Display only; full reports and approvals stay in the controller."""
from __future__ import annotations

import html
import json
import re
from collections import defaultdict
from typing import Any

from contracts import Claim
from metadata_answers import render_metadata_facts
from agents.coordinator import person_words
from run import required_fetches, scope_needs_a_check


LIST_LIMIT = 30
DISTINCT = "number of distinct non-empty exact text values"
FREQUENCY = "frequency of exact text values"
NAMES = {
    "studyDbId": ("Studies", (("studyName", "Study"), ("locationName", "Location"), ("seasons", "Season"))),
    "locationDbId": ("Locations", (("locationName", "Location"), ("countryName", "Country"))),
    "germplasmDbId": ("Germplasm", (("germplasmName", "Name"),)),
    "observationVariableDbId": ("Traits", (("observationVariableName", "Trait"),)),
    "observationUnitDbId": ("Observation units", (("observationUnitName", "Name"),)),
    "seasonDbId": ("Seasons", (("seasonName", "Season"),)),
    "programDbId": ("Programs", (("programName", "Program"),)),
}
_INTERNAL_ID = re.compile(r"\b(?:clm|art|req|plan|run)_[A-Za-z0-9_-]+\b")
_WINDOWS_PATH = re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/][^\n;:]*?\.[A-Za-z0-9]{1,8}(?=\s|[\"'`),;:]|$)(?::\d+)?|[A-Za-z]:[\\/][^\s;:]+)")
_LOCAL_PATH = re.compile(r"(?<![:/\w])(?:/(?:home|tmp|mnt|workspace|app)/|(?:out|logs|cache)/)[^\s,;]+")


def _plain(value: Any) -> str:
    text = str(value or "").strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                text = ", ".join(str(item) for item in parsed)
        except ValueError:
            pass
    text = _WINDOWS_PATH.sub("(file details in Full report)", text)
    text = _LOCAL_PATH.sub("(file details in Full report)", text)
    return _INTERNAL_ID.sub("(evidence in Full report)", text)


def _text(value: Any) -> str:
    text = html.escape(_plain(value), quote=False).replace("\n", " ").replace("\r", " ")
    return re.sub(r"([\\`*_{}\[\]()|])", r"\\\1", text)


def _number(value: float | int | None) -> str:
    if value is None:
        return "Not available"
    if isinstance(value, int):
        return str(value)
    return format(value, ".12g")


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |",
            *["| " + " | ".join(_text(cell) for cell in row) + " |" for row in rows]]


class _Sources:
    def __init__(self, registry: Any):
        self.registry = registry
        self.by_hash: dict[str, str] = {}
        self.loaded: dict[str, list[dict[str, str]] | None] = {}
        if registry is not None:
            for aid in registry.artifact_ids():
                try:
                    self.by_hash[registry.manifest(aid).sha256] = aid
                except Exception:
                    continue

    def rows(self, hashes: tuple[str, ...]) -> list[dict[str, str]] | None:
        # Do not arbitrarily choose one table when a claim cites several.
        if len(hashes) != 1 or hashes[0] not in self.by_hash:
            return None
        fingerprint = hashes[0]
        if fingerprint not in self.loaded:
            try:
                self.loaded[fingerprint] = self.registry.load(self.by_hash[fingerprint])[1]
            except Exception:
                self.loaded[fingerprint] = None
        return self.loaded[fingerprint]


def _sort_id(value: str) -> tuple[float | int, str]:
    return (int(value) if value.isdigit() else float("inf"), value)


def _names(rows: list[dict[str, str]], column: str, key: str, name_column: str) -> str:
    return "; ".join(sorted({_plain(row.get(name_column, "")) for row in rows
                              if row.get(column, "") == key and _plain(row.get(name_column, ""))}))


def _catalog_block(claims: list[Claim], sources: _Sources, source_label: str, *, focus_count: bool = False) -> list[str] | None:
    first = claims[0]
    column = first.grouping[0]
    rows = sources.rows(tuple(sorted(first.source_artifact_hashes)))
    if rows is None:
        return None
    distinct = [claim for claim in claims if claim.transformation == DISTINCT]
    frequency = [claim for claim in claims if claim.transformation == FREQUENCY]
    if len({claim.value for claim in distinct}) > 1:
        return None
    counts: dict[str, Any] = {}
    for claim in frequency:
        key = claim.filters[column]
        if key in counts and counts[key] != claim.value:
            return None
        counts[key] = claim.value
    source_ids = {row.get(column, "") for row in rows if row.get(column, "")}
    if distinct and distinct[0].value != len(source_ids):
        return None  # Preserve conflicting statistics separately instead of silently choosing one.
    keys = sorted(source_ids if distinct else set(counts), key=_sort_id)
    keys.extend(sorted(set(counts) - set(keys), key=_sort_id))  # includes the explicit missing-value group
    if frequency:
        keys.sort(key=lambda key: (-(counts.get(key) or 0), *_sort_id(key)))
    title, name_columns = NAMES.get(column, (column, ()))
    total = f"{_number(distinct[0].value)} distinct IDs" if distinct else f"{len(keys)} groups"
    lines = [f"**{_text(title)}: {total}{source_label}**", ""]
    # Only redundant one-per-ID frequencies can be omitted; conflicting/additional counts stay visible.
    if focus_count and distinct and (not frequency or (set(counts) == source_ids and all(value == 1 for value in counts.values()))):
        return [*lines, f"Based on {_text(first.denominator.name)}: {first.denominator.n}."]
    headers = ["ID", *[label for _, label in name_columns]]
    show_frequency = bool(frequency) and (not distinct or any(value != 1 for value in counts.values()))
    if show_frequency:
        headers.append("Studies" if column == "locationDbId" and "studies table" in first.denominator.name else "Records")
    table_rows = []
    for key in keys[:LIST_LIMIT]:
        values = [key, *[_names(rows, column, key, name_column) or "Not stated" for name_column, _ in name_columns]]
        if show_frequency:
            values.append(_number(counts[key]) if key in counts else "Not reported")
        table_rows.append(values)
    lines += _table(headers, table_rows)
    if len(keys) > LIST_LIMIT:
        lines += ["", f"Showing {LIST_LIMIT} of {len(keys)} entries; {len(keys) - LIST_LIMIT} additional IDs are not displayed here. "
                  "Full report records the counts and source evidence."]
    lines += ["", f"Based on {_text(first.denominator.name)}: {first.denominator.n}."]
    return lines


def _claim_method(claim: Claim) -> str:
    return re.sub(r"^(?:mean|median|min|max|sd|count) of valid values;\s*", "", claim.transformation)


def _mean_focus(controller: Any, claims: list[Claim]) -> bool:
    """Only a clearly requested mean may hide the numeric tool's incidental statistics in chat."""
    plan = getattr(controller, "plan", None)
    if getattr(getattr(plan, "statistic", None), "kind", None) != "mean" or sum(claim.kind == "mean" for claim in claims) != 1:
        return False
    words = person_words(getattr(plan, "question", ""))
    if not re.search(r"\b(?:mean|average)\b", words, re.I):
        return False
    # An explicit second metric or a general summary request keeps every supplied metric visible.
    # 'n' means the displayed denominator and does not require a redundant Count row.
    return not re.search(r"\b(?:median|minimum|maximum|min|max|sd|std|standard deviation|variance|range|"
                         r"count|counts|total|totals|ratio|rank|ranks|percentile|quartile|statistics|stats|summary|summarize|describe)\b", words, re.I)


def _count_focus(controller: Any, claims: list[Claim]) -> bool:
    """A single total request may omit redundant IDs; lists, breakdowns and comparisons keep them."""
    plan = getattr(controller, "plan", None)
    if getattr(getattr(plan, "statistic", None), "kind", None) != "count":
        return False
    words = person_words(getattr(plan, "question", ""))
    if not re.search(r"^\s*(?:how many|what (?:is|are) the (?:total(?: number)?|number) of)\b", words, re.I):
        return False
    if re.search(r"\b(?:and|per|each|by|group|groups|grouped|compare|comparison|versus|vs|which|list|show|breakdown|distribution)\b", words, re.I):
        return False
    distinct = [claim for claim in claims if claim.transformation == DISTINCT]
    if len(distinct) != 1 or len(distinct[0].grouping) != 1:
        return False
    target = distinct[0]
    return all(claim.kind == "count" and claim.transformation in (DISTINCT, FREQUENCY)
               and claim.grouping == target.grouping
               and sorted(claim.source_artifact_hashes) == sorted(target.source_artifact_hashes)
               and claim.denominator == target.denominator for claim in claims)


def _numeric_blocks(claims: list[Claim], sources: _Sources, source_labels: dict[tuple[str, ...], str], *, focus_mean: bool = False) -> list[str]:
    groups: dict[tuple[Any, ...], list[Claim]] = defaultdict(list)
    for claim in claims:
        key = (tuple(sorted(claim.source_artifact_hashes)), tuple(sorted(claim.filters.items())), tuple(claim.grouping),
               claim.denominator.name, claim.denominator.n, claim.n_independent_units,
               tuple((item.reason, item.count) for item in claim.exclusions), _claim_method(claim))
        groups[key].append(claim)
    lines: list[str] = []
    labels = {"mean": "Mean", "median": "Median", "min": "Minimum", "max": "Maximum", "sd": "Sample standard deviation", "count": "Count"}
    for key, group in groups.items():
        first = group[0]
        hashes = key[0]
        rows = sources.rows(hashes) or []
        about = []
        for column, value in first.filters.items():
            name_columns = NAMES.get(column, (column, ()))[1]
            names = [_names(rows, column, value, name_column) for name_column, _ in name_columns]
            about.append(" · ".join([value, *[name for name in names if name]]))
        title = "; ".join(about) or "Summary"
        if not focus_mean or about or source_labels.get(hashes):
            lines += [f"**{_text(title)}{source_labels.get(hashes, '')}**", ""]
        rendered = []
        for claim in group:
            value = _number(claim.value)
            if claim.value is None:
                value += ": " + (claim.missing_reason or "reason not provided")
            if claim.kind != "count":
                value += " " + (claim.unit or "(unit not stated)")
            rendered.append([labels.get(claim.kind, claim.kind.title()), value])
        if focus_mean and len(rendered) == 1:
            lines += [f"**Mean: {_text(rendered[0][1])}**"]
        else:
            lines += _table(["Statistic", "Result"], rendered)
        independent = str(first.n_independent_units) if first.n_independent_units is not None else "unknown"
        if focus_mean:
            lines += ["", f"n = {first.denominator.n} ({_text(first.denominator.name)}). Recorded analysis units: {independent}.", ""]
        else:
            lines += ["", f"Based on {_text(first.denominator.name)}: {first.denominator.n}. Recorded analysis units: {independent}.",
                      f"Method: {_text(_claim_method(first))}.", ""]
    return lines


def _human_note(note: str) -> str:
    """Translate specific internal phrases without guessing the cause of a failed run."""
    text = note.strip()
    text = re.sub(r"^(?:(?:numeric_summary|group_stats|count_records|analysis target|model(?: note)?)\s*:\s*)+", "", text, flags=re.I)
    if "no complete artifact is eligible for analysis" in text.lower():
        return re.sub("no complete artifact is eligible for analysis", "No complete data table was available for analysis, so an answer could not be calculated.", text, flags=re.I)
    # Normalize only the two known unit-only messages; other caveat wording is preserved.
    unit_messages = {
        "unit not stated in the artifact's measurement metadata; claims carry unit=null",
        "unit is not stated in the artifact's metadata, so the mean value is reported with unit=null.",
        "unit is not stated in the source metadata.",
    }
    if " ".join(text.split()).casefold() in unit_messages:
        return "The measurement unit is not stated in the source data."
    return re.sub(r"unit=null", "unit not stated", text, flags=re.I)


def _note_key(note: str) -> str:
    return " ".join(note.split()).casefold()


_CODE_AVERAGE_LIMIT = "descriptive means of valid values as recorded; not adjusted for trial design, environment, missingness or clone composition; not evidence of genetic superiority"
_CODE_STAGE_LIMIT = "Growth stage or timepoint is not stated in the source metadata; a shared growth stage cannot be verified."
_ALL_MISSING_DATES = re.compile(r"Measurement dates are not recorded for ([0-9]+) of ([0-9]+) valid source rows; the timing of those measurements is unknown\.")
_MODEL_REPEAT_REQUIRES = {
    "growth stage or timepoint is not stated; shared timing cannot be verified.": _note_key(_CODE_STAGE_LIMIT),
    "descriptive mean is not evidence of genetic superiority.": _note_key(_CODE_AVERAGE_LIMIT),
    "the mean is a simple descriptive average of all valid recorded values in the study; it does not account for trial design, environment, or genetic factors.": _note_key(_CODE_AVERAGE_LIMIT),
}
# Exact administrative wording already covered by the table; never a keyword-based warning filter.
_MODEL_BOOKKEEPING = frozenset({"study names were found in the artifact preview"})
# Observed process summaries only. Additional clauses or unfamiliar warnings are kept in chat.
_MODEL_PROCESS_PATTERNS = (
    re.compile(r"Observations for variable '[^'\r\n]+' \([^()\r\n]+\) in study [A-Za-z0-9_-]+ were fetched\. Artifact: art_[A-Za-z0-9_]+\. Statistical calculations, such as the pooled mean, are not performed\."),
    re.compile(r"Exported complete study metadata for season [0-9]+ in art_[A-Za-z0-9_]+ \([0-9]+ studies, [0-9]+ previewed\)\."),
    re.compile(r"There are [0-9]+ locations in the catalog\. Artifact art_[A-Za-z0-9_]+ contains the complete exported table\."),
)


def _caveat_groups(controller: Any, claims: list[Claim]) -> tuple[list[str], list[str]]:
    """Keep tool/controller limits separate from unverified model prose; preserve unknown warnings."""
    analysis = getattr(controller, "analysis", None)
    retrieval = getattr(controller, "retrieval", None)
    notes = [*(getattr(analysis, "caveats", []) or []), *(getattr(retrieval, "warnings", []) or [])]
    exclusions = [*(getattr(analysis, "exclusions", []) or []), *[item for claim in claims for item in claim.exclusions]]
    notes += [f"Excluded: {item.count} — {item.reason}." for item in exclusions if item.count]
    source_notes, model_notes = [], []
    catalog_only = (bool(claims) and all(claim.kind == "count" and claim.transformation in
                    (DISTINCT, FREQUENCY, "row count of a complete table") for claim in claims)) or (
                    not claims and getattr(getattr(getattr(controller, "plan", None), "statistic", None), "kind", None) == "count")
    for note in notes:
        if note.lower().startswith("key claims (model"):
            continue
        if note.lower().startswith(("model note:", "model:")):
            model_notes.append(_human_note(note))
        elif catalog_only and note.lower().startswith("analysis target: descriptive means"):
            continue
        else:
            source_notes.append(_human_note(note))
    source_keys = {_note_key(note) for note in source_notes}
    all_dates_missing = any(match and int(match[1]) > 0 and match[1] == match[2]
                            for match in (_ALL_MISSING_DATES.fullmatch(note) for note in source_notes))
    unit_in_result = any(claim.kind != "count" and claim.unit is None for claim in claims)
    source_result, model_result = [], []
    seen_source, seen_model = set(), set()
    for note in source_notes:
        key = _note_key(note)
        if note == "The measurement unit is not stated in the source data." and unit_in_result:
            continue
        if key and key not in seen_source:
            seen_source.add(key)
            source_result.append(_text(note))
    for note in model_notes:
        key = _note_key(note)
        if key in source_keys or key in _MODEL_BOOKKEEPING or any(pattern.fullmatch(note) for pattern in _MODEL_PROCESS_PATTERNS):
            continue
        if note == "The measurement unit is not stated in the source data." and unit_in_result:
            continue
        if _MODEL_REPEAT_REQUIRES.get(key) in source_keys:
            continue
        if key == "measurement dates are missing for all rows; timing is unknown." and all_dates_missing:
            continue
        if key and key not in seen_model:
            seen_model.add(key)
            model_result.append(_text(note))
    if any(claim.kind != "count" for claim in claims):
        warning = "Distinct IDs alone do not establish statistical independence."
        if _note_key(warning) not in seen_source:
            source_result.append(warning)
    return source_result, model_result


def _caveats(controller: Any, claims: list[Claim]) -> list[str]:
    limits, model_notes = _caveat_groups(controller, claims)
    return [*limits, *["Unverified model note: " + note for note in model_notes]]


def _limit_blocks(limits: list[str], model_notes: list[str]) -> list[str]:
    lines = []
    if limits:
        lines += ["**Data limits and exclusions**", "", *[f"- {note}" for note in limits], ""]
    if model_notes:
        lines += ["**Additional model notes (unverified)**", "", *[f"- {note}" for note in model_notes], ""]
    return lines


def _provider_failure(controller: Any) -> tuple[str, str, str] | None:
    """Recognize the runtime's explicit provider error record, never model-written warning text."""
    analysis = getattr(controller, "analysis", None)
    retrieval = getattr(controller, "retrieval", None)
    notes = [*(getattr(analysis, "caveats", []) or []), *(getattr(retrieval, "warnings", []) or []),
             *(getattr(controller, "notes", []) or [])]
    for note in notes:
        match = re.fullmatch(r"agent error (provider_[a-z_]+): (.+)", note)
        if match:
            return note, match[1], match[2]
    return None


def answer_summary(controller: Any) -> str:
    """One compact answer from typed claims; no model narrative or internal evidence IDs."""
    status = getattr(controller, "execution_status", "unknown")
    analysis = getattr(controller, "analysis", None)
    retrieval = getattr(controller, "retrieval", None)
    if status != "completed" or getattr(analysis, "status", None) != "completed" or getattr(retrieval, "status", None) != "completed":
        shown = {"incomplete": "Incomplete result", "failed": "Could not complete this question", "blocked": "Question blocked",
                 "canceled": "Question canceled", "needs_clarification": "More detail is needed", "limit_reached": "A request limit was reached"}.get(status, "No verified answer is available")
        reasons = _caveats(controller, [])
        provider = _provider_failure(controller)
        if provider is not None:
            raw_error, code, message = provider
            if code == "provider_http_error" and message == "provider returned HTTP 429":
                code = "provider_rate_limit_unknown"
                message = "The model provider returned HTTP 429; its cause was not reported"
            reasons = [reason for reason in reasons if reason != _text(_human_note(raw_error))]
            evidence = ("Data retrieval completed, but the model service could not finish the analysis."
                        if getattr(retrieval, "status", None) == "completed" else "The model service could not finish this question.")
            next_step = {"provider_quota": "Check the model provider account quota or billing limit before retrying.",
                         "provider_rate_limit": "Wait before retrying this question.",
                         "provider_rate_limit_unknown": "Check the provider quota and rate limits before retrying; the cause of HTTP 429 is unknown."}.get(
                             code, "Resolve the model service issue before retrying.")
            return "\n\n".join(["**Could not complete this question: Model service unavailable.**", _text(message.rstrip(".")) + ".", evidence,
                                 "This failure does not establish that the database has no matching records.",
                                 *reasons, next_step, "Full report keeps the exact failure and any completed retrieval evidence."])
        if not reasons:
            notes = getattr(controller, "notes", [])
            reasons = [_text(_human_note(note)) for note in notes if re.search(r"error|fail|block|missing|not |timeout|cancel|limit|incomplete|clarification|stop|refus", note, re.I)][-3:]
        if not reasons:
            reasons = ["The run did not produce a complete, verified analysis."]
        retry = []
        if any("No complete data table was available" in reason for reason in reasons):
            retry = ["Retry the question, then review the proposed scope and approve its retrieval. "
                     "If the same step stops again, check Full report for the recorded retrieval reason."]
        return "\n\n".join([f"**{shown}.**", *reasons, *retry, "See Full report for the complete reason and evidence."])
    facts = getattr(analysis, "metadata_facts", [])
    if facts:
        lines = render_metadata_facts(facts)
        limits, model_notes = _caveat_groups(controller, [])
        lines += ["", *_limit_blocks(limits, model_notes)]
        lines += ["", "Full report contains the source records and evidence for these fields."]
        return "\n".join(lines)
    claims = list(analysis.claims)
    focus_mean = _mean_focus(controller, claims)
    focus_count = _count_focus(controller, claims)
    displayed_claims = [claim for claim in claims if claim.kind == "mean"] if focus_mean else claims
    sources = _Sources(getattr(getattr(controller, "ctx", None), "registry", None))
    hashes = list(dict.fromkeys(tuple(sorted(claim.source_artifact_hashes)) for claim in claims))
    source_labels = {key: f" · source {index + 1}" for index, key in enumerate(hashes)} if len(hashes) > 1 else {}
    buckets: dict[tuple[Any, ...], list[Claim]] = defaultdict(list)
    other = []
    for claim in displayed_claims:
        eligible = claim.kind == "count" and len(claim.grouping) == 1 and (
            (claim.transformation == DISTINCT and not claim.filters) or
            (claim.transformation == FREQUENCY and set(claim.filters) == set(claim.grouping)))
        if eligible:
            buckets[(tuple(sorted(claim.source_artifact_hashes)), claim.grouping[0], claim.denominator.name, claim.denominator.n)].append(claim)
        else:
            other.append(claim)
    lines: list[str] = []
    for key, group in buckets.items():
        block = _catalog_block(group, sources, source_labels.get(key[0], ""), focus_count=focus_count)
        if block is None:
            other.extend(group)
        else:
            lines += [*block, ""]
    lines += _numeric_blocks(other, sources, source_labels, focus_mean=focus_mean)
    caveats, model_notes = _caveat_groups(controller, claims)
    if focus_mean:
        for claim in claims:
            if claim.kind != "mean" and claim.value is None and claim.missing_reason:
                warning = _text("An additional statistic could not be calculated: " + claim.missing_reason)
                if warning not in caveats:
                    caveats.append(warning)
    lines += _limit_blocks(caveats, model_notes)
    if focus_mean and len(displayed_claims) < len(claims):
        lines.append("Other calculated statistics remain in Full report, together with the complete methods and evidence.")
    else:
        lines.append("Full report contains the complete methods and evidence.")
    return "\n".join(lines)


def approval_summary(controller: Any) -> str:
    """Show the exact permission being requested without duplicating the technical manifest."""
    manifest, plan = controller.manifest, controller.plan
    if manifest is None or plan is None:
        return "No validated fetch plan is ready. See Full report for details."
    metadata = getattr(controller, "metadata", None)
    studies = {study.study_id: study for study in (getattr(metadata, "studies", []) or [])}
    variables = {variable.variable_id: variable for variable in (getattr(metadata, "variables", []) or [])}
    pairs, units = required_fetches(plan, manifest)
    lines = ["**Review this plan before approving**", "", f"Source: {_text(manifest.base_url)}."]
    if manifest.origin == "snapshot":
        lines += ["Uses an offline snapshot; no live requests."]
    else:
        fresh = getattr(controller, "catalog_fetched_at", None) or "unknown date"
        setup = getattr(controller, "catalog_setup", None) or {}
        origin = "prepared during this question" if setup.get("status") == "completed" else "previously cached"
        lines += [f"Catalogs: {origin}; fetched {_text(fresh)}. Cached replies are reused."]
    if manifest.observation_study_ids:
        lines += ["", "**Observation scope**", ""]
        rows = []
        for study_id in manifest.observation_study_ids:
            study = studies.get(study_id)
            rows.append([study_id, getattr(study, "name", "") or "Not stated"])
        lines += _table(["Study ID", "Study"], rows)
        if manifest.observation_variable_ids:
            lines += ["", "Traits: " + "; ".join(
                _text(f"{variable_id} · {getattr(variables.get(variable_id), 'name', '') or 'name not stated'} "
                      f"({getattr(variables.get(variable_id), 'unit', None) or 'unit not stated'})")
                for variable_id in manifest.observation_variable_ids) + "."]
        lines += [f"At least {len(pairs) + len(units)} data fetches: {len(pairs)} study/trait requests and {len(units)} plot rosters.",
                  "Approval covers only these observation study IDs and trait IDs; other observations are excluded."]
    else:
        lines += ["", "**Catalog question: no observation requests.**"]
        scoped = list(plan.scope.study_ids)
        if scoped:
            shown = ", ".join(_text(study_id) for study_id in scoped[:10])
            tail = f"; {len(scoped) - 10} additional matched IDs are not shown here" if len(scoped) > 10 else ""
            lines += [f"Matched studies: {len(scoped)}. IDs: {shown}{tail}."]
    filters = {key: value for key, value in plan.scope.filters.model_dump().items()
               if key != "contract_version" and value not in (None, "", [])}
    if filters:
        lines += ["Filters: " + "; ".join(f"{_text(key)} = {_text(value)}" for key, value in filters.items()) + "."]
    lines += ["Allowed data: " + ", ".join(_text(family) for family in manifest.endpoint_families) + ".",
              f"Limits: {manifest.max_http_attempts} HTTP attempts, {manifest.max_observation_studies} observation studies, "
              f"{controller.config.max_fetches} data fetch calls, and {controller.budget.max_model_calls} model calls."]
    if scope_needs_a_check(controller.config.question, list(plan.scope.study_ids), metadata):
        lines += ["**Check the scope:** the planner interpreted your question as several studies. Cancel if you meant one."]
    lines += ["", "Approve authorizes this read-only plan. It does not accept the eventual answer. Cancel stops here; "
              "edit can only narrow the study list. Full report contains the exact plan and approval details."]
    return "\n".join(lines)
