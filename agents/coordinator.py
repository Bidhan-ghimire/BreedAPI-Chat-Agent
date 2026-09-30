"""
agents/coordinator.py — the Coordinator: drafts a Plan from validated records, renders the final
numeric table from typed claims, and polices optional prose. It has NO execution authority.

What the Coordinator is allowed to do:
1. Draft a plan. The model receives the question plus the VALIDATED metadata the controller holds
   (cached studies/variables/locations/seasons exports) and answers with a PlanDraft: interpretation,
   clarifications, a scope of exact IDs, a statistic with a named denominator, and ordered steps for the
   retriever and analyst. It has no tools at all — no HTTP, no shell, no filesystem, no budget knobs.
2. Have that draft checked and assembled by CODE into a contracts.Plan:
   - every study/variable/location/season ID must exist in the metadata (or the plan must ask for
     discovery when there is no metadata); IDs the model invents are refused;
   - matching_study_count is computed here or left None ("unknown") — never taken from the model;
   - clarifications the question does not need are dropped (a uniquely named study already supplies
     its location and season); clarifications the question DOES need are added (two matching traits,
     no matching trait for a numeric question, "best / recommend" wording, which is a breeding
     judgement this assistant never makes);
   - a rejected draft is sent back with the reasons, at most max_replans times.
3. Render the answer table DETERMINISTICALLY from typed claims (same claims -> same text), keeping
   every caveat and every no-data study on the ledger.
4. Sanitise optional prose: a sentence that states a number must cite a claim ID in brackets whose
   value/n/count it repeats; numbers inside identifiers (P4, S1, art_..._0001) are not numbers;
   sentences with unsupported numbers or breeding-decision words are REMOVED and flagged.

Everyday example: a project manager who writes the plan and the summary but holds no keys, no
budget card and no lab access — and whose summary is checked line by line against the lab's
printed results before it goes out.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from artifacts import ArtifactRegistry
from contracts import (
    AgentResult,
    AnalysisReport,
    Claim,
    Clarification,
    Plan,
    PlanStep,
    ResolvedScope,
    RetrievalReport,
    Statistic,
    StudyFilters,
)
from llm import Budget, ModelClient, run_agent_loop

__all__ = [
    "KnownStudy", "KnownVariable", "KnownMetadata", "Candidates", "find_candidates", "PlanDraft", "CoordinatorResult",
    "COORDINATOR_SYSTEM_PROMPT", "NoTools", "draft_plan", "assemble_plan", "render_numeric_table", "sanitize_prose",
    "RETRIEVER_ACTIONS", "ANALYST_ACTIONS", "DECISION_WORDS", "WHOLE_SET_WORDS", "WHOLE_SET_REJECTION", "ONE_OR_SEVERAL", "SCOPE_JUDGED_NOTE",
    "PERSON_REVIEWS_NOTE",
    "add_clarification", "clarifications_given", "person_words", "latest_words",
]

RETRIEVER_ACTIONS = frozenset({"server_info", "search_studies", "study_types", "get_study", "list_variables", "get_observations",
                               "list_locations", "list_programs", "list_seasons", "get_observation_units", "export_metadata"})
ANALYST_ACTIONS = frozenset({"table_info", "numeric_summary", "group_stats", "filter_rows", "concat_tables", "missing_report", "count_records"})
DISCOVERY_FAMILIES: frozenset[str] = frozenset({"serverinfo", "commoncropnames", "studies", "observationvariables", "locations", "programs", "seasons"})
# Wording that makes a question about SEVERAL studies (so "all known studies" may be its scope). A question that names one study
# by an unknown name has none of these, and then no study may be chosen for the user. Seen 2026-09-28: a strong model answered
# "In study 19ayt18highYldIB ..." with S1, the only study it knew; the scope was valid metadata but not what the question said.
WHOLE_SET_WORDS = re.compile(r"\b(studies|trials|experiments)\b|\b(which|every|each|any|all) (study|trial|experiment)\b", re.IGNORECASE)
WHOLE_SET_REJECTION = "all_known_studies is only for a question about several studies"
# Asked by code when the whole-set rule refused every attempt (seen 2026-09-29: "list ids of sweetpotato study done in 2026" was
# planned as "all studies" three times and refused three times: "study", not "studies"). Only the person knows which is meant,
# so the person is asked; the rule itself is not widened, because "How many clones are in study X?" has the same list-like
# wording and must never become a question about all studies.
ONE_OR_SEVERAL = ("Is this question about one study or several? For one study, give its ID or exact name. For several, answer "
                  "with the words 'all studies' and any filter, for example: all studies from 2026.")
# Written into the notes when a person-reviewed plan covers several studies although the question has no several-studies
# wording (2026-09-29: "study", "stdy", "list ids of ... study"); the approval screen then asks the person to check the scope.
SCOPE_JUDGED_NOTE = ("the planner judged the question to be about several studies although it has no several-studies wording; "
                     "the approval screen shows that scope for the person to check")
# Told to the planner only when a person will see the approval screen (the chat; the command line without --auto). Run 203842
# (2026-09-29): the planner offered 'studies from the 2026 season?', the person said yes, and it asked again three times.
PERSON_REVIEWS_NOTE = ("A person sees the approval screen, with the plan's scope, before anything is fetched, and can cancel there. "
                       "If the question most likely means several studies (for example 'the study from 2026' or 'list study IDs'), "
                       "plan it with all_known_studies and exact filters instead of asking again which study is meant.")

# A person's answer to the planner's question is ADDED to the question (run.clarify_phase calls add_clarification), after a
# label that code writes. The rules that read a question use only the person's words: person_words() drops the labels, so
# "Clarification 1" can never name a study whose ID is 1, and latest_words() is what the person said last.
_CLARIFIED = re.compile(r"\nClarification (\d+) from the person: ")


def clarifications_given(question: str) -> int:
    return len(_CLARIFIED.findall(question))


def add_clarification(question: str, answer: str) -> str:
    """The question so far plus the person's newest answer, labelled and numbered. Nothing is replaced: the first words stay."""
    return f"{question}\nClarification {clarifications_given(question) + 1} from the person: {' '.join(answer.split())}"


def person_words(question: str) -> str:
    """Only what the person typed: the question with the code-written labels removed."""
    return _CLARIFIED.sub("\n", question)


def latest_words(question: str) -> str:
    """What the person said last: the question itself, or their newest answer once they have answered."""
    return _CLARIFIED.split(question)[-1]


DECISION_WORDS = re.compile(r"\b(best|worst|superior|inferior|recommend\w*|release|outperform\w*|top[- ]performing|winner|highest[- ]yielding|should (?:we|i) (?:grow|plant|select))\b", re.IGNORECASE)
_STOPWORDS = frozenset({"the", "and", "for", "with", "per", "mean", "means", "average", "study", "studies", "trial", "trials", "clone", "clones",
                        "synthetic", "give", "show", "compare", "what", "which", "how", "many", "each", "all", "from", "that", "this", "are", "was",
                        "were", "have", "has", "does", "list", "count", "number", "yields",
                        # generic measurement words that appear in many variable names and point at none in particular
                        "content", "percent", "percentage", "index", "score", "value", "values", "weight", "ratio", "total"})
_WORD = re.compile(r"[A-Za-z0-9_.-]+")
_NUMBER = re.compile(r"(?<![\w.:-])[-+]?\d+(?:\.\d+)?(?![\w:])(?!\.\d)(?!-\d)")   # a number NOT glued to letters, underscores, dates or times (P4, S1, art_..._0001, 2026-06-01T00:00)
_CITATION = re.compile(r"\[(clm_[A-Za-z0-9_-]+)\]")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


# --------------------------------------------------------------------------
# Validated records the Coordinator may see (built by code from cached exports)
# --------------------------------------------------------------------------

class _Strict(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class KnownStudy(_Strict):
    study_id: str
    name: str = ""
    location_id: str | None = None
    location_name: str | None = None
    seasons: list[str] = Field(default_factory=list)
    study_type: str | None = None
    program_id: str | None = None
    program_name: str | None = None


class KnownVariable(_Strict):
    variable_id: str
    name: str = ""
    trait: str | None = None
    unit: str | None = None


class KnownMetadata(_Strict):
    """What the controller has actually fetched and validated. Nothing here comes from a model."""

    base_url: str
    complete: bool
    studies: list[KnownStudy] = Field(default_factory=list)
    variables: list[KnownVariable] = Field(default_factory=list)
    location_ids: list[str] = Field(default_factory=list)
    season_ids: list[str] = Field(default_factory=list)

    @classmethod
    def from_registry(cls, registry: ArtifactRegistry, *, base_url: str, studies_artifact: str, variables_artifact: str,
                      locations_artifact: str | None = None, seasons_artifact: str | None = None) -> "KnownMetadata":
        """Build from export_metadata artifacts (text tables). complete is the AND of the exports' completeness."""
        s_manifest, s_rows = registry.load(studies_artifact)
        v_manifest, v_rows = registry.load(variables_artifact)
        complete = s_manifest.complete and v_manifest.complete
        studies = [KnownStudy(study_id=r["studyDbId"], name=r.get("studyName", ""), location_id=r.get("locationDbId") or None,
                              location_name=r.get("locationName") or None, seasons=_seasons_from_text(r.get("seasons", "")),
                              study_type=r.get("studyType") or None, program_id=r.get("programDbId") or None,
                               program_name=r.get("programName") or None) for r in s_rows if r.get("studyDbId")]
        variables = [KnownVariable(variable_id=r["observationVariableDbId"], name=r.get("observationVariableName", ""),
                                   trait=r.get("traitName") or None, unit=r.get("units") or None) for r in v_rows if r.get("observationVariableDbId")]
        location_ids = sorted({s.location_id for s in studies if s.location_id})
        season_ids = sorted({season for s in studies for season in s.seasons})
        if locations_artifact:
            l_manifest, l_rows = registry.load(locations_artifact)
            complete = complete and l_manifest.complete
            location_ids = sorted(set(location_ids) | {r["locationDbId"] for r in l_rows if r.get("locationDbId")})
        if seasons_artifact:
            se_manifest, se_rows = registry.load(seasons_artifact)
            complete = complete and se_manifest.complete
            season_ids = sorted(set(season_ids) | {r["seasonDbId"] for r in se_rows if r.get("seasonDbId")})
        return cls(base_url=base_url, complete=complete, studies=studies, variables=variables, location_ids=location_ids, season_ids=season_ids)

    def summary_for_model(self, question: str | None = None) -> str:
        """The catalog as the planning model sees it.

        A small catalog (the synthetic server: one study, two variables) is listed whole. A real catalog is not — SweetPotatoBase
        has 2,029 studies and 486 variables, some 40,000 tokens if listed, more than a local model can read at all. With a
        question given, only what the question's own words point at is listed: the studies it names by ID or exact name, the
        variables whose names share the most words with it (at most SUMMARY_MAX_LISTED), the locations whose names it mentions,
        and the totals. Everything else is unnamed on purpose: the code refuses IDs the question did not name anyway (assemble_plan),
        so the model gains nothing from seeing them, and the prompt stays small enough for every model.
        """
        totals = f"{len(self.studies)} studies, {len(self.variables)} variables, {len(self.location_ids)} locations, {len(self.season_ids)} seasons"
        lines = [f"Validated metadata from {self.base_url} (complete={self.complete}): {totals}."]
        known_programs = sum(bool(s.program_id or s.program_name) for s in self.studies)
        lines.append(f"Program membership metadata: {known_programs} of {len(self.studies)} studies have a program ID or name; "
                     "missing program fields mean unknown membership, not no membership. Never infer it from study names.")
        small = len(self.studies) <= SUMMARY_MAX_LISTED and len(self.variables) <= SUMMARY_MAX_LISTED
        if question is None or small:
            lines.append("Studies: " + "; ".join(_study_line(s) for s in self.studies))
            lines.append("Variables: " + "; ".join(_variable_line(v) for v in self.variables))
            lines.append(f"Locations: {self.location_ids}  Seasons: {self.season_ids}")
            return "\n".join(lines)
        question = person_words(question)                                # only the person's words point at anything
        words = _tokens(question)
        cands = find_candidates(question, self)
        studies = cands.studies[:SUMMARY_MAX_LISTED]
        scored = sorted(((len(_tokens(f'{v.name} {v.trait or ""}') & words), v) for v in cands.variables), key=lambda t: (-t[0], t[1].variable_id))
        variables = [v for score, v in scored if score > 0][:SUMMARY_MAX_LISTED]
        locations = sorted({(s.location_id, s.location_name) for s in self.studies
                            if s.location_id and s.location_name and _tokens(s.location_name) & words})[:SUMMARY_MAX_LISTED]
        lines.append(f"The catalog is large, so only what the question names is listed. Studies the question names ({len(studies)} of {len(self.studies)}): "
                     + ("; ".join(_study_line(s) for s in studies) or "none — for a question about all studies set all_known_studies=true and use filters; "
                        "for a named study missing here, ask which study is meant"))
        lines.append(f"Variables whose names share words with the question ({len(variables)} of {len(self.variables)}, best matches first): "
                     + ("; ".join(_variable_line(v) for v in variables) or "none — ask which trait is meant"))
        lines.append(f"Locations the question mentions ({len(locations)} of {len(self.location_ids)}): "
                     + ("; ".join(f"{lid} '{name}'" for lid, name in locations) or "none") + f"  Seasons: {self.season_ids}")
        return "\n".join(lines)


SUMMARY_MAX_LISTED = 15          # a catalog with at most this many studies AND variables is listed whole; beyond it, only what the question names


def _study_line(s: KnownStudy) -> str:
    return f"{s.study_id} '{s.name}' location={s.location_id or '?'} seasons={s.seasons or '?'}" + (f" type={s.study_type}" if s.study_type else "")


def _variable_line(v: KnownVariable) -> str:
    return f"{v.variable_id} '{v.name}' unit={v.unit or '?'}"


def _seasons_from_text(text: str) -> list[str]:
    """The export stores a list as JSON text; anything else is one season name."""
    text = (text or "").strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            return [str(x) for x in parsed if str(x)]
        except ValueError:
            pass
    return [text]


@dataclass
class Candidates:
    """What the QUESTION itself points at, found by literal word matching against the metadata."""

    studies: list[KnownStudy]
    variables: list[KnownVariable]
    decision_words: list[str]


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _WORD.findall(text) if len(t) >= 3 and t.lower() not in _STOPWORDS}


def _variable_named_in(v: KnownVariable, lowered: str, raw_words: set[str]) -> bool:
    """The question names this variable outright: its ID as a token, or its full name, its name before the ontology suffix
    ('fresh root yield|CO_334:0000013' -> 'fresh root yield') or its trait name written out in the question."""
    if v.variable_id.lower() in raw_words:
        return True
    names = {v.name, v.name.split("|", 1)[0], v.trait or ""}
    return any(len(n.strip()) >= 4 and n.strip().lower() in lowered for n in names)


def find_candidates(question: str, metadata: KnownMetadata | None) -> Candidates:
    """Exact naming wins: a variable the question names outright (ID, full name or trait) is the only candidate; only when the
    question names none does word overlap decide. On a real catalog a common word such as 'estimating' occurs in hundreds of
    variable names, so overlap alone turned a question that quoted one variable into 'more than one trait matches'.

    Only the person's words count (person_words): our own clarification labels never name anything. Decision words are read in
    what the person said LAST (latest_words): once they have answered the 'breeding judgement' question, the 'best' of their
    first question is not asked about again, while every word they typed still counts for studies and traits."""
    latest = latest_words(question)
    question = person_words(question)
    words = _tokens(question)                                          # meaningful words, for names
    raw_words = {t.lower().strip(".,;:!?") for t in _WORD.findall(question)}   # every token, for exact IDs such as S1 or V1 ("S1." at a sentence end is S1)
    lowered = question.lower()
    studies: list[KnownStudy] = []
    variables: list[KnownVariable] = []
    if metadata is not None:
        for s in metadata.studies:
            if s.study_id.lower() in raw_words or (s.name and s.name.lower() in lowered):
                studies.append(s)
        named = [v for v in metadata.variables if _variable_named_in(v, lowered, raw_words)]
        if named:
            variables = named
        else:
            for v in metadata.variables:
                name_tokens = _tokens(f"{v.name} {v.trait or ''}")
                if name_tokens & words:
                    variables.append(v)
    return Candidates(studies=studies, variables=variables, decision_words=sorted({m.group(0).lower() for m in DECISION_WORDS.finditer(latest)}))


# --------------------------------------------------------------------------
# What the model hands back (strict) and how code turns it into a Plan
# --------------------------------------------------------------------------

class DraftClarification(_Strict):
    question: str = Field(min_length=1)
    required: bool = True


class DraftScope(_Strict):
    study_ids: list[str] = Field(default_factory=list)
    all_known_studies: bool = False           # a question about all studies: code fills study_ids with every known study
    variable_ids: list[str] = Field(default_factory=list)
    location_ids: list[str] = Field(default_factory=list)
    season_ids: list[str] = Field(default_factory=list)
    discovery_needed: bool = False
    discovery_families: list[str] = Field(default_factory=list)
    filters: dict[str, str] = Field(default_factory=dict)


class DraftStep(_Strict):
    step_id: str
    agent: Literal["retriever", "analyst"]
    action: str
    inputs: dict[str, str | int | bool | list[str]] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    max_tool_calls: int = 10


class PlanDraft(_Strict):
    """The model's proposal. No counts, no budgets, no approvals: those fields do not exist here."""

    interpretation: str = Field(min_length=1)
    clarifications: list[DraftClarification] = Field(default_factory=list)
    scope: DraftScope
    statistic: Statistic | None = None
    steps: list[DraftStep] = Field(min_length=1)


COORDINATOR_SYSTEM_PROMPT = """You are the Coordinator for a read-only plant-breeding database assistant. You have NO tools.
You write a plan; other agents (retriever, analyst) execute it later, and a human approves every fetch.
Rules:
1. Use ONLY study, variable, location and season IDs that appear in the validated metadata you are given. Never invent an ID.
   A large catalog is shown in part: only the studies, variables and locations the question's words point at, with the totals;
   an ID that is not shown may not be used. A study goes into scope.study_ids only when the question names it by its ID or its exact name. If the question names a study
   that is not in the metadata, do not pick another one: ask which study is meant. For a question about all studies (wording such
   as "studies", "trials", "which study"), set scope.all_known_studies=true and leave study_ids empty; the controller fills them,
   narrowed by scope.filters. A catalog question that needs no study at all (how many seasons, variables or locations exist)
   leaves study_ids empty with all_known_studies=false and uses export_metadata steps.
   If no metadata is given, set scope.discovery_needed=true and list the metadata endpoint families and filters to discover with
   (studies, observationvariables, locations, seasons). Discovery never includes observations.
2. Never state how many studies match; the controller counts. Do not include counts, estimates, budgets or approvals anywhere.
3. Ask a clarification ONLY when the question cannot be answered without it: two traits could match, no trait matches,
   or the question asks for a breeding judgement (best, superior, recommend) that this assistant never makes.
   A uniquely named study already tells us its location and season; do not ask for them.
4. Steps: retriever actions are search_studies, list_variables, get_observations, get_observation_units, export_metadata,
   list_locations, list_seasons, get_study; analyst actions are table_info, numeric_summary, group_stats, count_records,
   missing_report, filter_rows, concat_tables. Each step depends only on earlier steps; every analyst step traces back to a
   retriever step, directly or through the earlier analyst steps it depends on (export -> filter -> count is fine; a step that
   depends on nothing is not). The list_* and search_* actions return
   previews only, never a table the analyst can use: a count over catalog rows (variables, studies, seasons, locations) needs an
   export_metadata step first, then an analyst filter_rows or count_records step that depends on it.
   Step inputs are SIMPLE values only — a string, an integer, a boolean or a list of strings — never an object or a nested list —
   and they use the tool's OWN argument names: get_observations(study_db_id="S1", variable_db_id="V1") is written
   "inputs": {"study_db_id": "S1", "variable_db_id": "V1"}; likewise export_metadata(entity="studies"),
   group_stats(by="germplasmDbId", min_independent_n=2), filter_rows(column="observationVariableName", op="contains", value="dry matter").
5. The statistic names its denominator (for example "valid plot-level values"); a catalog question uses kind "count".
6. scope has exactly these keys: study_ids, all_known_studies, variable_ids, location_ids, season_ids, discovery_needed,
   discovery_families, filters. scope.filters holds only location_id, season_id, study_type, name_contains, program_id — each ONE string
   (an exact known ID or the exact study type); lists of IDs go in scope.location_ids / scope.season_ids. Nothing else anywhere.
7. What the LAST analyst step must state, so the answer is tied to rows and not a bare number (argument names as in rule 4):
   - how many / which studies: count_records(group_by="studyDbId", distinct_by="studyDbId") on the FILTERED studies table — one
     call states the count and every counted study.
   - how many / which variables: count_records(group_by="observationVariableDbId", distinct_by="observationVariableDbId") on the
     FILTERED variables table — one call states the count and every counted variable.
   - which locations have the most studies: count_records(group_by="locationDbId") over the complete studies table; the
     controller ranks, the plan does not pick a top few.
   - how many clones / plots per clone / independent units: get_observations AND get_observation_units, then
     group_stats(by="germplasmDbId", min_independent_n=1, top_n=1000) — count_records counts rows, never plots.
   - narrowing a table: filter_rows(column="studyType", op="eq", value="Advanced Yield Trial"); op is one of eq, ne, contains,
     gt, ge, lt, le, is_missing, is_valid_number. A catalog table comes from export_metadata(entity="variables"); entity is one
     of studies, variables, locations, programs, seasons.
8. For a question about several studies (all_known_studies=true), leave study_db_id OUT of the get_observations and
   get_observation_units inputs — never a placeholder such as "*each*": the controller gives the retriever the resolved study
   list, and the retriever fetches each study in it.
Finish with ONLY this JSON object:
{"status": "completed", "payload": {"interpretation": "<one sentence>", "clarifications": [{"question": "<text>", "required": true}],
 "scope": {"study_ids": [], "all_known_studies": false, "variable_ids": [], "location_ids": [], "season_ids": [], "discovery_needed": false, "discovery_families": [], "filters": {}},
 "statistic": {"kind": "mean", "denominator": "<name>", "grouping": []} or null,
 "steps": [{"step_id": "step_1", "agent": "retriever", "action": "get_observations", "inputs": {"study_db_id": "S1", "variable_db_id": "V1"}, "depends_on": []}]}}
or {"status": "needs_clarification", "question": "<one precise question>"}"""


class NoTools:
    """The Coordinator's tool interface: nothing is advertised and nothing can be called."""

    async def schemas(self) -> list[dict[str, Any]]:
        return []

    async def call(self, name: str, args: Any) -> Any:  # pragma: no cover - the loop refuses unknown tools before reaching here
        raise RuntimeError(f"the Coordinator has no tools; {name!r} was requested")


@dataclass
class CoordinatorResult:
    status: str                                   # completed | needs_clarification | failed | limit_reached | blocked | incomplete
    plan: Plan | None
    questions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    agents: list[AgentResult] = field(default_factory=list)
    attempts: int = 0


def study_program_metadata_question(question: str, draft: PlanDraft, metadata: KnownMetadata | None) -> str | None:
    """Program membership needs recorded program fields; study-name matches cannot supply that fact."""
    words = person_words(question)
    latest = latest_words(question)
    if (clarifications_given(question) and not re.search(r"\bprogram(?:me)?s?\b", latest, re.IGNORECASE)
            and re.search(r"\b(?:use|instead|forget|ignore)\b", latest, re.IGNORECASE)
            and re.search(r"\b(?:location|season|year|study type)\b", latest, re.IGNORECASE)
            and (any(draft.scope.filters.get(k) for k in ("location_id", "season_id", "study_type"))
                 or draft.scope.location_ids or draft.scope.season_ids)):
        return None                     # the person explicitly replaced the unsupported scope
    if (metadata is None or not metadata.studies
            or not re.search(r"\bprogram(?:me)?s?\b", words, re.IGNORECASE)
            or not re.search(r"\b(?:stud(?:y|ies)|trials?)\b", words, re.IGNORECASE)):
        return None
    if re.search(r"\b(?:study|trial)\s+names?\b.*\b(?:contain\w*|include\w*|match\w*)\b",
                 latest_words(question), re.IGNORECASE):
        return None
    columns = {str(step.inputs.get(key, "")) for step in draft.steps if step.agent == "analyst"
               for key in ("column", "group_by", "distinct_by")}
    requested = columns & {"programDbId", "programName"}
    if draft.scope.filters.get("program_id"):
        requested.add("programDbId")
    attrs = {"programDbId": "program_id", "programName": "program_name"}
    selected = set(draft.scope.study_ids)
    relevant = (metadata.studies if draft.scope.all_known_studies or not selected else
                [study for study in metadata.studies if study.study_id in selected])
    unknown = sum(1 for study in relevant
                  if (any(not (getattr(study, attrs[column]) or "").strip() for column in requested) if requested
                      else not ((study.program_id or "").strip() or (study.program_name or "").strip())))
    if unknown:
        known = len(relevant) - unknown
        return (f"Program membership is unknown for {unknown} of {len(relevant)} scoped studies "
                f"({known} have the needed program metadata). Missing metadata does not mean zero matching studies. "
                "This catalog cannot give a complete program-membership answer. Would you like to use a recorded field, "
                "such as location or season, instead?")
    if not requested:
        return ("Program membership must use programDbId or programName; searching study names does not establish it. "
                "Which recorded program ID or program name should be used?")
    return None


def assemble_plan(question: str, draft: PlanDraft, metadata: KnownMetadata | None, *, base_url: str, now: datetime,
                  plan_id: str, person_reviews_scope: bool = False) -> tuple[Plan | None, list[str], list[str]]:
    """Turn a draft into a validated Plan. Returns (plan, hard_problems, notes); hard problems mean 'send it back'.

    person_reviews_scope: a person will see the approval screen before anything is fetched (the chat, or the command line
    without --auto). Then the planner's judgment that a question is about several studies is accepted even without the
    wording 'studies' ('study', 'stdy', a list question): the approval screen shows that scope, and the person can cancel.
    Without a person (--auto, the evaluation) the wording rule stays strict."""
    problems: list[str] = []
    notes: list[str] = []
    cands = find_candidates(question, metadata)
    scope = draft.scope
    study_ids = list(scope.study_ids)                                  # may be filled by code below; the model never widens it

    if metadata is None:
        if not scope.discovery_needed:
            problems.append("no validated metadata is available; the plan must set discovery_needed=true and propose metadata families and filters")
        if scope.study_ids or scope.variable_ids or scope.location_ids or scope.season_ids:
            problems.append("without validated metadata no exact IDs may be named; use discovery")
    else:
        if scope.discovery_needed and metadata.complete:
            problems.append("validated metadata is complete; discovery is not needed and no discovery step may be planned")
        known_studies = {s.study_id for s in metadata.studies}
        known_vars = {v.variable_id for v in metadata.variables}
        for label, wanted, known in (("study_ids", scope.study_ids, known_studies), ("variable_ids", scope.variable_ids, known_vars),
                                     ("location_ids", scope.location_ids, set(metadata.location_ids)), ("season_ids", scope.season_ids, set(metadata.season_ids))):
            unknown = [x for x in wanted if x not in known]
            if unknown:
                problems.append(f"scope.{label} names IDs that are not in the validated metadata: {unknown}")
        candidate_vars = {v.variable_id for v in cands.variables}
        off_question = [v for v in scope.variable_ids if v not in candidate_vars]
        if off_question:
            problems.append(f"scope.variable_ids {off_question} are not what the question names; matching variables are {sorted(candidate_vars) or 'none'} "
                            "- ask a clarification instead of choosing for the user")
        # a study enters the scope only when the QUESTION names it; "all known studies" only when the question is about several
        candidate_studies = {s.study_id for s in cands.studies}
        if scope.all_known_studies:
            if candidate_studies:
                problems.append(f"the question names study {sorted(candidate_studies)}; all_known_studies must be false when a study is named")
            elif not person_reviews_scope and not WHOLE_SET_WORDS.search(person_words(question)):
                problems.append(WHOLE_SET_REJECTION + " (wording such as 'studies', 'trials', 'which study'); "
                                "this question names one study that is not in the validated metadata, or none. If it names a study, ask which study "
                                "is meant; if it is a catalog question that needs no study at all (how many seasons, variables or locations exist), "
                                "leave study_ids empty with all_known_studies=false and plan export_metadata steps")
            else:
                study_ids = sorted(s.study_id for s in metadata.studies if _study_matches(s, scope))
                notes.append(f"scope.study_ids set by code to the {len(study_ids)} of {len(known_studies)} known studies that match the plan's "
                             "filters, because the question is about all studies; on a live server this list can be long, and every fetch still "
                             "needs the approval screen and stays under the fetch ceiling")
                if not WHOLE_SET_WORDS.search(person_words(question)):
                    notes.append(SCOPE_JUDGED_NOTE)
        else:
            unnamed = [s for s in scope.study_ids if s not in candidate_studies]
            whole_set = sorted(s.study_id for s in metadata.studies if _study_matches(s, scope))
            if unnamed and not candidate_studies and WHOLE_SET_WORDS.search(person_words(question)) and whole_set and sorted(scope.study_ids) == whole_set:
                study_ids = whole_set                         # the model said aloud exactly what code would fill: nothing added, nothing dropped
                notes.append(f"scope.study_ids {whole_set} accepted for a question about all studies: the list equals the {len(whole_set)} known "
                             "studies that match the plan's filters (checked by code)")
            elif unnamed:
                problems.append(f"scope.study_ids {unnamed} are not named in the question by ID or exact name; it names {sorted(candidate_studies) or 'no known study'} "
                                "- ask which study is meant instead of choosing for the user (a question about all studies sets all_known_studies)")
    bad_families = [f for f in scope.discovery_families if f not in DISCOVERY_FAMILIES]
    if bad_families:
        problems.append(f"discovery may use metadata families only, not {bad_families}")

    in_scope_ids = set(study_ids) | set(scope.variable_ids)
    for step in draft.steps:
        allowed = RETRIEVER_ACTIONS if step.agent == "retriever" else ANALYST_ACTIONS
        if step.action not in allowed:
            problems.append(f"{step.step_id}: {step.action!r} is not a {step.agent} action")
        for key in ("study_db_id", "variable_db_id"):
            value = step.inputs.get(key)
            if isinstance(value, str) and value not in in_scope_ids:
                hint = (" - for a question about several studies leave study_db_id out: the controller gives the retriever the resolved "
                        "study list" if key == "study_db_id" and scope.all_known_studies else "")
                problems.append(f"{step.step_id}: input {key}={value!r} is outside the plan's scope{hint}")

    # clarifications: drop the irrelevant, add the required
    clarifications = [Clarification(question=c.question, required=c.required) for c in draft.clarifications]
    if len(cands.studies) == 1:
        kept = []
        for c in clarifications:
            if re.search(r"\b(location|season|site|year)\b", c.question, re.IGNORECASE):
                notes.append(f"dropped clarification {c.question!r}: study {cands.studies[0].study_id} is uniquely named and its metadata supplies location and season")
            else:
                kept.append(c)
        clarifications = kept
    numeric = draft.statistic is not None and draft.statistic.kind != "count"
    if cands.decision_words:
        text = (f"The question uses {', '.join(repr(w) for w in cands.decision_words)}, which asks for a breeding judgement this assistant does not make. "
                "Should it report descriptive means per clone or study with named n instead, and for which trait?")
        clarifications = _ensure_clarification(clarifications, text, r"breeding judgement", notes)
    if metadata is not None and numeric and not scope.discovery_needed:
        if len(cands.variables) == 0:
            names = ", ".join(f"{v.variable_id} ({v.name})" for v in metadata.variables) or "none"
            clarifications = _ensure_clarification(clarifications, f"The question names no known trait. Which variable is meant: {names}?", r"which variable", notes)
        elif len(cands.variables) > 1:
            names = ", ".join(f"{v.variable_id} ({v.name})" for v in cands.variables)
            clarifications = _ensure_clarification(clarifications, f"More than one trait matches the question: {names}. Which one?", r"more than one trait|which one", notes)

    metadata_question = study_program_metadata_question(question, draft, metadata)
    if metadata_question is not None:
        # A model-written optional or answered question cannot satisfy a code-required evidence guard.
        clarifications = [c for c in clarifications if not re.search(r"program membership", c.question, re.IGNORECASE)]
        clarifications.append(Clarification(question=metadata_question, required=True, answer=None))
        notes.append("program membership is not established; a fresh required clarification keeps matching_study_count unknown")
    if problems:
        return None, problems, notes

    status = "needs_clarification" if any(c.required and c.answer is None for c in clarifications) else "ready"
    matching = len(study_ids) if (metadata is not None and metadata.complete and not scope.discovery_needed
                                 and metadata_question is None) else None
    try:
        resolved = ResolvedScope(base_url=base_url, study_ids=study_ids, variable_ids=list(scope.variable_ids),
                                 location_ids=list(scope.location_ids), season_ids=list(scope.season_ids), filters=StudyFilters(**scope.filters),
                                 discovery_needed=scope.discovery_needed, discovery_families=[f for f in scope.discovery_families],   # type: ignore[misc]
                                 matching_study_count=matching)
        steps = [PlanStep(step_id=s.step_id, agent=s.agent, action=s.action, inputs=dict(s.inputs), depends_on=list(s.depends_on),
                          max_tool_calls=s.max_tool_calls) for s in draft.steps]
        plan = Plan(plan_id=plan_id, question=question, interpretation=draft.interpretation, clarifications=clarifications, status=status,
                    scope=resolved, statistic=draft.statistic, steps=steps, created_at_utc=now)
    except ValidationError as exc:
        return None, [f"plan contract rejected the draft at {_explain_rejection(exc)}"], notes
    if matching is not None:
        notes.append(f"matching_study_count={matching} was counted by code from the scope, not taken from the model")
    return plan, [], notes


def _study_matches(study: KnownStudy, scope: "DraftScope") -> bool:
    """Does this known study pass the plan's exact filters? Filters narrow a whole-set question (Ibadan, 2019, Advanced Yield Trial)
    BEFORE any study is put in scope; with no filters every known study matches."""
    filters = scope.filters
    if filters.get("program_id") is not None and study.program_id != filters["program_id"]:
        return False
    location = filters.get("location_id") or (scope.location_ids[0] if len(scope.location_ids) == 1 else None)
    if scope.location_ids and study.location_id not in scope.location_ids:
        return False
    if location is not None and study.location_id != location:
        return False
    season = filters.get("season_id")
    wanted_seasons = set(scope.season_ids) | ({season} if season else set())
    if wanted_seasons and not (wanted_seasons & set(study.seasons)):
        return False
    study_type = filters.get("study_type")
    if study_type is not None and (study.study_type or "").strip().casefold() != study_type.strip().casefold():
        return False
    name_contains = filters.get("name_contains")
    if name_contains is not None and name_contains.strip().casefold() not in (study.name or "").casefold():
        return False
    return True


_SCOPE_KEYS = "study_ids, all_known_studies, variable_ids, location_ids, season_ids, discovery_needed, discovery_families, filters"
_FILTER_KEYS = "location_id, season_id, study_type, name_contains, program_id (each one string)"


def _explain_rejection(exc: ValidationError) -> str:
    """The first contract error as an instruction the model can act on, not a bare 'Input should be a valid string'.

    On the real catalog gpt-4.1 spent all three attempts putting objects into step inputs and lists into scope.filters; each
    time the raw pydantic message told it where, never what would be accepted. The location is kept; the remedy is added."""
    first = exc.errors()[0]
    loc = ".".join(str(p) for p in first["loc"]) or "(root)"
    msg = first["msg"]
    if "inputs" in first["loc"] and ("valid string" in msg or "valid integer" in msg or "valid boolean" in msg or "valid list" in msg):
        remedy = ("step inputs must be simple values - a string, an integer, a boolean or a list of strings - never an object or a nested "
                  "list, and only the tool's own argument names, e.g. group_stats(by=\"germplasmDbId\", min_independent_n=2) or "
                  "filter_rows(column=\"observationVariableName\", op=\"contains\", value=\"dry matter\"), written as JSON inputs")
    elif loc.startswith("scope.filters") or loc.startswith("filters"):
        remedy = f"scope.filters may hold only {_FILTER_KEYS}; lists of known IDs go in scope.location_ids / scope.season_ids"
    elif "Extra inputs are not permitted" in msg:
        remedy = f"'{first['loc'][-1]}' is not a field here; scope has only {_SCOPE_KEYS}, and filters only {_FILTER_KEYS}"
    elif "must depend on at least one retriever step" in msg:
        remedy = ("give that analyst step a depends_on that leads back to a retriever step: the step_id of the export_metadata or "
                  "get_observations step, or of an earlier analyst step that itself depends on one")
    else:
        remedy = None
    return f"{loc}: {msg}" + (f" - {remedy}" if remedy else "")


def _ensure_clarification(existing: list[Clarification], text: str, already_pattern: str, notes: list[str]) -> list[Clarification]:
    if any(re.search(already_pattern, c.question, re.IGNORECASE) for c in existing):
        return existing
    notes.append(f"added required clarification: {text}")
    return [*existing, Clarification(question=text, required=True)]


def _user_message(question: str, metadata: KnownMetadata | None, problems: list[str], earlier_questions: list[str] | None = None,
                  person_reviews_scope: bool = False) -> str:
    """What the planner reads. Its own earlier questions are CONTEXT only: every rule in assemble_plan reads the question (the
    person's words), never these lines, so the planner's words cannot name a study or unlock a rule."""
    lines = [f"Question: {question}"]
    if earlier_questions:
        lines.append("For context, your earlier questions to the person (the person's answers are the 'Clarification N' lines of the "
                     "question above; only the person's words are the question):")
        lines.extend(f"- your question {i}: {q}" for i, q in enumerate(earlier_questions, 1))
    if person_reviews_scope:
        lines.append(PERSON_REVIEWS_NOTE)
    lines.append(metadata.summary_for_model(question) if metadata is not None else "No validated metadata is available yet; propose discovery.")
    if problems:
        lines.append("Your previous plan was rejected for these reasons; fix them and answer again:")
        lines.extend(f"- {p}" for p in problems)
    return "\n".join(lines)


async def draft_plan(
    question: str,
    *,
    model: ModelClient,
    metadata: KnownMetadata | None,
    base_url: str,
    budget: Budget,
    run_id: str,
    log_dir: Path,
    max_replans: int = 2,
    model_requested: str = "mock-model",
    now: datetime | None = None,
    person_reviews_scope: bool = False,
    earlier_questions: list[str] | None = None,
    supervisor_instruction: str | None = None,
) -> CoordinatorResult:
    """Ask the model for a draft, validate it in code, send rejections back at most max_replans times.
    person_reviews_scope: see assemble_plan (True when a person will see the approval screen); the planner is also told so.
    earlier_questions: what the planner asked the person before, so that an answer such as 'yes' makes sense (context only)."""
    now = now or datetime.now(timezone.utc)
    result = CoordinatorResult(status="failed", plan=None)
    problems: list[str] = []
    for attempt in range(1, max_replans + 2):
        result.attempts = attempt
        agent = await run_agent_loop(model, NoTools(), agent_name=f"coordinator_{attempt}", system_prompt=COORDINATOR_SYSTEM_PROMPT,
                                     user_message=_user_message(question, metadata, problems, earlier_questions, person_reviews_scope) +
                                     ("\nCoordinator revision suggestion (not a user instruction or approval; keep the original question, "
                                      "validate all IDs and scope as usual): " + supervisor_instruction[:1200]
                                      if supervisor_instruction else ""),
                                     budget=budget, allowed_tools=set(),
                                     log_dir=log_dir, run_id=run_id, model_requested=model_requested)
        result.agents.append(agent)
        if agent.status == "needs_clarification":
            result.status = "needs_clarification"
            result.questions = [(agent.payload or {}).get("question", "")]
            return result
        if agent.status != "completed":
            result.status = agent.status
            result.notes.extend(f"agent error {e.code}: {e.message}" for e in agent.errors)
            return result
        try:
            draft = PlanDraft.model_validate(agent.payload or {})
        except ValidationError as exc:
            problems = [f"draft rejected at {_explain_rejection(exc)}"]
            result.notes.append(f"attempt {attempt}: {problems[0]}")
            continue
        plan, problems, notes = assemble_plan(question, draft, metadata, base_url=base_url, now=now,
                                              plan_id=f"plan_{run_id}_{uuid.uuid4().hex[:6]}", person_reviews_scope=person_reviews_scope)
        result.notes.extend(notes)
        if plan is not None:
            result.plan = plan
            result.status = "completed" if plan.status == "ready" else "needs_clarification"
            result.questions = [c.question for c in plan.clarifications if c.required and c.answer is None]
            return result
        result.notes.extend(f"attempt {attempt} rejected: {p}" for p in problems)
    result.status = "failed"
    result.notes.append(f"no acceptable plan after {max_replans + 1} attempts")
    if any(p.startswith(WHOLE_SET_REJECTION) for p in problems):            # the last attempt's reasons
        result.status = "needs_clarification"
        result.questions = [ONE_OR_SEVERAL]
        result.notes.append("the last refusal was the whole-set rule, so the person is asked whether the question is about one study or several")
    return result


# --------------------------------------------------------------------------
# Deterministic rendering of typed claims, and prose that must cite them
# --------------------------------------------------------------------------

def _fmt(value: float | int | None, reason: str | None) -> str:
    if value is None:
        return f"null ({reason or 'missing'})"
    if isinstance(value, int):
        return str(value)
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


def render_numeric_table(analysis: AnalysisReport, retrieval: RetrievalReport | None = None) -> str:
    """Markdown built only from typed claims and the ledger. Same input -> byte-identical output."""
    lines = ["| claim | kind | value | unit | denominator (n) | independent units | scope | exclusions |", "|---|---|---|---|---|---|---|---|"]
    for c in sorted(analysis.claims, key=lambda x: x.claim_id):
        scope = ", ".join(f"{k}={v}" for k, v in sorted(c.filters.items())) or "all rows"
        exclusions = "; ".join(f"{e.reason} x{e.count}" for e in c.exclusions) or "none"
        units = "unknown" if c.n_independent_units is None else str(c.n_independent_units)
        lines.append(f"| {c.claim_id} | {c.kind} | {_fmt(c.value, c.missing_reason)} | {c.unit or 'not stated'} | {c.denominator.name} ({c.denominator.n}) | {units} | {scope} | {exclusions} |")
    if not analysis.claims:
        lines.append("| (no claims) | | | | | | | |")
    if retrieval is not None:
        lines.append("")
        lines.append("| study | fetch status | rows | reason |")
        lines.append("|---|---|---|---|")
        rows_by_artifact = {a.artifact_id: a.row_count for a in retrieval.artifacts}
        for e in sorted(retrieval.study_ledger, key=lambda x: x.study_id):
            rows = rows_by_artifact.get(e.artifact_id or "", "-")
            lines.append(f"| {e.study_id} | {e.status} | {rows} | {e.reason or ''} |")
        if not retrieval.study_ledger:
            lines.append("| (no studies fetched) | | | |")
    caveats = list(analysis.caveats) + ([w for w in retrieval.warnings] if retrieval is not None else [])
    if caveats:
        lines.append("")
        lines.append("Caveats:")
        lines.extend(f"- {c}" for c in caveats)
    return "\n".join(lines)


def _number_supported(text: str, claims: list[Claim]) -> bool:
    """A number in prose is supported when a cited claim carries it: counts exactly, values to the stated rounding."""
    text = text.lstrip("+")
    for claim in claims:
        counts = {claim.denominator.n, claim.n_independent_units, *(e.count for e in claim.exclusions)}
        if claim.kind == "count" and claim.value is not None:
            counts.add(claim.value)
        if "." not in text and text.lstrip("-").isdigit() and int(text) in {c for c in counts if c is not None}:
            return True
        if claim.value is not None:
            decimals = len(text.split(".")[1]) if "." in text else 0
            if abs(float(text) - float(claim.value)) <= 0.5 * 10 ** (-decimals) + 1e-12:
                return True
    return False


def sanitize_prose(prose: str, claims: list[Claim]) -> tuple[str, list[str]]:
    """Keep only sentences whose numbers are backed by the claims they cite; flag what was removed.

    Numbers glued to letters or underscores (P4, S1, art_2a9751_0001, 2026-06-01T...) are identifiers,
    not numeric claims, and are ignored. A sentence with a number and no citation is removed. A sentence
    with a breeding-decision word is removed whatever it cites.
    """
    by_id = {c.claim_id: c for c in claims}
    kept: list[str] = []
    flags: list[str] = []
    for sentence in (s.strip() for s in _SENTENCE_END.split(prose.strip()) if s.strip()):
        if DECISION_WORDS.search(sentence):
            flags.append(f"removed (breeding judgement): {sentence[:80]}")
            continue
        cited = _CITATION.findall(sentence)
        unknown = [cid for cid in cited if cid not in by_id]
        if unknown:
            flags.append(f"removed (cites unknown claim {unknown[0]}): {sentence[:80]}")
            continue
        without_citations = _CITATION.sub(" ", sentence)
        numbers = [n.lstrip("+") for n in _NUMBER.findall(without_citations)]
        if numbers and not cited:
            flags.append(f"removed (numbers without a claim citation): {sentence[:80]}")
            continue
        cited_claims = [by_id[cid] for cid in cited]
        unsupported = [n for n in numbers if not _number_supported(n, cited_claims)]
        if unsupported:
            flags.append(f"removed (number {unsupported[0]} not in cited claims): {sentence[:80]}")
            continue
        kept.append(sentence)
    return " ".join(kept), flags
