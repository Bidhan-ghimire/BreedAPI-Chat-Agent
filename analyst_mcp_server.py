"""The eight fixed Analyst tools over MCP, restricted to one analysis's artifacts.

This server receives no database client and cannot fetch additional data. Schemas
and typed dispatch stay in analyst_tools; existing runtime guards also run here
so a raw MCP client cannot bypass artifact scope or inspection requirements.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from agents.analyst import GuardedAnalystTools
from analyst_tools import ANALYST_TOOLS, AnalystTools, analyst_tool_schemas
from artifacts import ArtifactRegistry
from contracts import ToolError, ToolResult, dump_json

SERVER_NAME = "brapi-artifact-analyst"
SERVER_VERSION = "0.1"


def build_analyst_server(registry: ArtifactRegistry, artifact_ids: list[str]):
    """Create one isolated MCP tool session with canonical, validated results."""
    import anyio
    from mcp import types
    from mcp.server.lowlevel import Server

    guarded = GuardedAnalystTools(AnalystTools(registry), list(artifact_ids), model_claim_refs=False)
    call_lock = anyio.Lock()
    server = Server(SERVER_NAME, version=SERVER_VERSION,
                    instructions="Fixed calculations on supplied artifact handles only. Inspect every supplied "
                                 "table before calculation. No database access or arbitrary file paths.")

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        return [types.Tool(name=s["name"], description=s["description"], inputSchema=s["inputSchema"])
                for s in analyst_tool_schemas()]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        # MCP's tool listing is not an authorization check: its SDK may pass an
        # unlisted name to this handler. Deny it before touching the registry.
        spec = ANALYST_TOOLS.get(name)
        if spec is None:
            result = ToolResult(ok=False, error=ToolError(code="invalid_argument",
                               message=f"unknown Analyst tool {name!r}"))
        elif not isinstance(arguments, Mapping):
            result = ToolResult(ok=False, error=ToolError(code="invalid_argument",
                               message="arguments must be a JSON object"))
        else:
            try:
                # Validate before the guard iterates handles, including when an
                # SDK version changes or the handler is called without its wrapper.
                validated = spec.args_model.model_validate(dict(arguments))
            except ValidationError as exc:
                first = exc.errors()[0]
                loc = ".".join(str(p) for p in first.get("loc", ())) or "(root)"
                result = ToolResult(ok=False, error=ToolError(code="invalid_argument",
                                   message=f"{loc}: {first.get('msg')}"))
            else:
                # AnalystTools joins any worker on cancellation; derived writes
                # never continue in an abandoned background thread.
                # A concurrent MCP client must not race inspection state or
                # artifact registry writes within this analysis session.
                async with call_lock:
                    result = await guarded.call(name, validated.model_dump())
        payload = json.loads(dump_json(result))
        return types.CallToolResult(content=[types.TextContent(type="text", text=json.dumps(payload))],
                                    structuredContent=payload, isError=not result.ok)

    return server
