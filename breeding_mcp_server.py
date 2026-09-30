"""One local MCP service with code-owned retrieval and analysis stages.

Only the trusted workflow controller can change a stage. Tools cannot grant
approval, select a server, reopen retrieval, or supply paths. The service
reuses the existing typed tool registries and guarded dispatchers.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any, Literal

import anyio
from pydantic import ValidationError

from agents.analyst import GuardedAnalystTools
from agents.retriever import ApprovedTools
from analyst_tools import ANALYST_TOOLS, AnalystTools, analyst_tool_schemas
from brapi_mcp_server import TOOLS, ToolContext, dispatch, tool_schemas
from contracts import ToolError, ToolResult, dump_json

SERVER_NAME = "breeding-assistant"
SERVER_VERSION = "0.1"
Phase = Literal["catalog", "retrieval", "suspended", "analysis", "closed"]
_CATALOG_TOOLS = frozenset({"export_metadata", "request_log"})

_FAMILIES = {
    "server_info": "serverinfo", "search_studies": "studies", "study_types": "studies",
    "get_study": "studies", "list_variables": "observationvariables", "list_locations": "locations",
    "list_programs": "programs", "list_seasons": "seasons", "get_observations": "observations",
    "get_observation_units": "observationunits", "request_log": None,
}


def _refuse(code: str, message: str) -> ToolResult:
    return ToolResult(ok=False, error=ToolError(code=code, message=message))


class _RetrievalDispatcher:
    """One dispatch, one client: no extra HTTP calls or budget accounting."""

    def __init__(self, ctx: ToolContext):
        self.ctx = ctx

    async def schemas(self):
        return tool_schemas()

    async def call(self, name, args):
        # Keep the worker joined on cancellation, just like the existing server.
        return await anyio.to_thread.run_sync(dispatch, self.ctx, name, dict(args))


class BreedingMcpService:
    """A per-run server; retrieval -> suspended -> analysis -> closed.

    suspend/enable_analysis/close are Python methods, never MCP tools. All
    stage checks, transitions, inspection state and calls share one lock.
    Suspending waits for an already running tool; when it returns, no tool
    can run until trusted code enables analysis. Retrieval cannot reopen.
    """

    def __init__(self, ctx: ToolContext, *, now_utc: Callable[[], datetime] | None = None,
                 catalog_only: bool = False):
        from mcp import types
        from mcp.server.lowlevel import Server

        self._ctx = ctx
        self._now_utc = now_utc or (lambda: datetime.now(timezone.utc))
        self._phase: Phase = "catalog" if catalog_only else "retrieval"
        self._lock = anyio.Lock()
        self._retrieval = ApprovedTools(_RetrievalDispatcher(ctx), ctx, ctx.approval, now_utc=self._now_utc)
        self._analyst: GuardedAnalystTools | None = None
        self.server = Server(SERVER_NAME, version=SERVER_VERSION, instructions=(
            "Read-only breeding tools. Available tools depend on the controller's approved stage. "
            "The controller alone changes stages. Tables use managed handles; analysis must inspect every input."))

        @self.server.list_tools()
        async def _list_tools() -> list[types.Tool]:
            async with self._lock:
                schemas = tool_schemas() if self._phase in ("catalog", "retrieval") else (
                    analyst_tool_schemas() if self._phase == "analysis" else [])
                if self._phase == "catalog":
                    schemas = [s for s in schemas if s["name"] in _CATALOG_TOOLS]
                return [types.Tool(name=s["name"], description=s["description"], inputSchema=s["inputSchema"])
                        for s in schemas]

        # We validate with the registries below after checking the current stage.
        # A cached MCP discovery result is never treated as authorization.
        @self.server.call_tool(validate_input=False)
        async def _call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
            async with self._lock:
                result = await self._dispatch(name, arguments)
            payload = json.loads(dump_json(result))
            return types.CallToolResult(content=[types.TextContent(type="text", text=json.dumps(payload))],
                                        structuredContent=payload, isError=not result.ok)

    @property
    def phase(self) -> Phase:
        return self._phase

    async def suspend(self) -> None:
        async with self._lock:
            if self._phase not in ("retrieval", "suspended"):
                raise RuntimeError("only retrieval can be suspended for data review")
            self._phase = "suspended"

    async def enable_analysis(self, artifact_ids: list[str]) -> None:
        async with self._lock:
            if self._phase != "suspended":
                raise RuntimeError("analysis can only begin after retrieval is suspended")
            if not isinstance(artifact_ids, list) or any(not isinstance(a, str) for a in artifact_ids):
                raise ValueError("analysis inputs must be a list of registered artifact IDs")
            if len(set(artifact_ids)) != len(artifact_ids):
                raise ValueError("analysis inputs must not contain duplicate artifact IDs")
            registered = set(self._ctx.registry.artifact_ids())
            if any(a not in registered for a in artifact_ids):
                raise ValueError("analysis inputs must belong to this run's artifact registry")
            self._analyst = GuardedAnalystTools(AnalystTools(self._ctx.registry), list(artifact_ids),
                                               model_claim_refs=False)
            self._phase = "analysis"

    async def close(self) -> None:
        async with self._lock:
            self._phase = "closed"
            self._analyst = None

    def _approval_error(self, name: str, args: dict[str, Any]) -> str | None:
        approval = self._ctx.approval
        if approval is None:
            if self._phase == "catalog" and self._ctx.client.settings.mode == "offline":
                return None  # trusted cache-only catalog; the client cannot open a network connection
            return "no FetchApproval for this run"
        if approval.run_id != self._ctx.client.run_id or approval.run_id != self._ctx.registry.run_id:
            return "FetchApproval was issued for a different run"
        if approval.base_url != self._ctx.client.settings.base_url:
            return "FetchApproval was issued for a different server"
        if not approval.is_valid_at(self._now_utc()):
            return "FetchApproval expired or not yet valid"
        family = _FAMILIES.get(name)
        if name == "export_metadata":
            family = "observationvariables" if args["entity"] == "variables" else args["entity"]
        if family is not None and not approval.allows_family(family):
            return f"FetchApproval does not cover endpoint family {family!r}"
        return None

    async def _dispatch(self, name: str, arguments: Mapping[str, Any]) -> ToolResult:
        if self._phase not in ("catalog", "retrieval", "analysis"):
            return _refuse("not_authorized", f"tools are unavailable while the service is {self._phase}")
        registry = TOOLS if self._phase in ("catalog", "retrieval") else ANALYST_TOOLS
        spec = registry.get(name)
        if spec is None or (self._phase == "catalog" and name not in _CATALOG_TOOLS):
            code = "not_authorized" if name in TOOLS or name in ANALYST_TOOLS else "invalid_argument"
            return _refuse(code, f"tool {name!r} is not available in the {self._phase} stage")
        if not isinstance(arguments, Mapping):
            return _refuse("invalid_argument", "arguments must be a JSON object")
        try:
            args = spec.args_model.model_validate(dict(arguments)).model_dump()
        except ValidationError as exc:
            first = exc.errors()[0]
            loc = ".".join(str(p) for p in first.get("loc", ())) or "(root)"
            return _refuse("invalid_argument", f"{loc}: {first.get('msg')}")
        if self._phase == "catalog" and name == "export_metadata" and args["entity"] not in ("studies", "variables"):
            return _refuse("not_authorized", "catalog setup only permits study and variable metadata")
        if self._phase in ("catalog", "retrieval"):
            reason = self._approval_error(name, args)
            if reason:
                return _refuse("not_authorized", reason)
            self._retrieval.approval = self._ctx.approval
            return await self._retrieval.call(name, args)
        assert self._analyst is not None
        return await self._analyst.call(name, args)
