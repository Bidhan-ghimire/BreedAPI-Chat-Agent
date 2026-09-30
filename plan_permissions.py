"""Derive the database families a human is asked to approve for a typed plan.

This describes proposed access, never grants it. The controller still binds a
human's approval to the complete manifest; observation IDs and all request
budgets remain enforced by the existing controller/client guards.
"""
from __future__ import annotations

from typing import get_args

from contracts import EndpointFamily, Plan


_ACTION_FAMILIES: dict[str, EndpointFamily | None] = {
    "server_info": "serverinfo",
    "search_studies": "studies",
    "study_types": "studies",
    "get_study": "studies",
    "list_variables": "observationvariables",
    "get_observations": "observations",
    "get_observation_units": "observationunits",
    "list_locations": "locations",
    "list_programs": "programs",
    "list_seasons": "seasons",
    "request_log": None,  # Reads this run's local evidence, never the database.
}
_EXPORT_FAMILIES: dict[str, EndpointFamily] = {
    "studies": "studies",
    "variables": "observationvariables",  # BrAPI family differs from artifact name.
    "locations": "locations",
    "programs": "programs",
    "seasons": "seasons",
}
_INPUT_LOOKUPS: dict[str, EndpointFamily] = {
    "study_db_id": "studies",
    "variable_db_id": "observationvariables",
    "location_id": "locations",
    "season_id": "seasons",
    "program_id": "programs",
}


def endpoint_families_for_plan(plan: Plan) -> list[EndpointFamily]:
    """Return only planned read families and their required ID/name lookups.

    Catalog-only questions do not inherit permission for studies or traits.
    Observation plans still allow their study/trait lookups: the Retriever must
    verify IDs from tool results in this run. The same requirement applies to
    a plan's already-resolved location/season IDs, including study filters.
    Unknown actions/entities fail closed instead of receiving broad access.
    """
    families: set[EndpointFamily] = set(plan.scope.discovery_families)
    for step in plan.steps:
        if step.agent != "retriever":
            continue
        if step.action == "export_metadata":
            entity = step.inputs.get("entity")
            if not isinstance(entity, str) or entity not in _EXPORT_FAMILIES:
                raise ValueError(f"unsupported export_metadata entity in {step.step_id}: {entity!r}")
            families.add(_EXPORT_FAMILIES[entity])
        elif step.action in _ACTION_FAMILIES:
            family = _ACTION_FAMILIES[step.action]
            if family is not None:
                families.add(family)
        else:
            raise ValueError(f"unsupported retriever action in {step.step_id}: {step.action!r}")
        for key, family in _INPUT_LOOKUPS.items():
            if step.inputs.get(key):
                families.add(family)

    scope = plan.scope
    if scope.study_ids:
        families.add("studies")
    if scope.variable_ids:
        families.add("observationvariables")
    if scope.location_ids or scope.filters.location_id or scope.filters.location_name:
        families.add("locations")
    if scope.season_ids or scope.filters.season_id or scope.filters.season_name:
        families.add("seasons")
    if scope.filters.program_id:
        families.add("programs")
    # Stable ordering keeps the approval display/hash deterministic. No fallback
    # family: a plan with only local evidence reads grants no database access.
    return [family for family in get_args(EndpointFamily) if family in families]
