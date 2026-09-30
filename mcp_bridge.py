"""MCP sessions for the hosted breeding service.

The controller supplies a scoped server factory. Initialization, tool discovery,
and calls use real MCP over SDK memory streams. Direct dispatch remains an
explicit adapter for focused offline tests; the hosted controller cannot select
it. There is no standalone child-process launcher in this package.
"""
from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import Any, Literal

import anyio
from pydantic import ValidationError

from contracts import ToolError, ToolResult

__all__ = ["BridgeError", "BridgeTimeout", "BridgeClosed", "McpTools", "to_model_tools"]



class BridgeError(Exception):
    """The bridge could not obtain a valid tool reply (malformed content, closed session, ...)."""


class BridgeTimeout(BridgeError):
    """A tool call exceeded the call timeout. The session is marked unusable afterwards."""


class BridgeClosed(BridgeError):
    """The session is not open (never entered, already exited, or broken by a timeout)."""


def to_model_tools(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Wrap the server's schemas in the OpenAI-style function-tool shape. The parameters ARE the inputSchema."""
    tools = []
    for schema in schemas:
        if not isinstance(schema, dict) or not schema.get("name") or not isinstance(schema.get("inputSchema"), dict):
            raise BridgeError(f"malformed tool schema: {schema!r}")
        tools.append({
            "type": "function",
            "function": {
                "name": schema["name"],
                "description": schema.get("description") or "",
                "parameters": schema["inputSchema"],
            },
        })
    return tools


class McpTools:
    """One sequential tool session per controller run; do not issue concurrent calls.

    Memory mode shares one approved context and operation deadline. Use `async with`.
    """

    def __init__(
        self,
        *,
        mode: Literal["direct", "memory"],
        ctx: Any = None,
        call_timeout: float = 60.0,
        remaining_time: Callable[[], float] | None = None,
        server_factory: Callable[[Any], Any] | None = None,
    ) -> None:
        if mode not in ("direct", "memory"):
            raise BridgeError(f"unknown mode {mode!r}")
        if mode in ("direct", "memory") and ctx is None:
            raise BridgeError(f"{mode} mode needs a ToolContext (brapi_mcp_server.build_context)")
        if mode == "memory" and server_factory is None:
            raise BridgeError("memory MCP requires a controller-owned server factory")
        if server_factory is not None and mode != "memory":
            raise BridgeError("a custom server factory is only supported in memory mode")
        if not math.isfinite(call_timeout) or call_timeout <= 0:
            raise BridgeError("call_timeout must be positive")
        self.mode = mode
        self.ctx = ctx
        self.call_timeout = call_timeout
        self.remaining_time = remaining_time
        self.server_factory = server_factory
        self.protocol_events: list[dict[str, str]] = []
        self._memory_group: Any = None
        self._operation_deadline: float | None = None
        self._previous_remaining_time: Callable[[], float] | None = None
        self._stack: AsyncExitStack | None = None
        self._session: Any = None
        self._open = False
        self._broken: str | None = None
        self.calls_made = 0

    # -- constructors ------------------------------------------------------------


    @classmethod
    def direct(cls, ctx: Any, *, call_timeout: float = 60.0) -> "McpTools":
        return cls(mode="direct", ctx=ctx, call_timeout=call_timeout)

    @classmethod
    def memory(cls, ctx: Any, *, call_timeout: float = 60.0,
               remaining_time: Callable[[], float] | None = None,
               server_factory: Callable[[Any], Any] | None = None) -> "McpTools":
        """Real MCP over SDK memory streams. A trusted factory can select a restricted tool server."""
        return cls(mode="memory", ctx=ctx, call_timeout=call_timeout, remaining_time=remaining_time,
                   server_factory=server_factory)

    def _allowance(self) -> float:
        seconds = self.call_timeout
        if self.remaining_time is not None:
            seconds = min(seconds, float(self.remaining_time()))
        if not math.isfinite(seconds) or seconds <= 0:
            raise BridgeTimeout("no remaining work time for MCP")
        return seconds

    def _client_remaining_time(self) -> float:
        limits = [self.call_timeout]
        if self._operation_deadline is not None:
            limits.append(self._operation_deadline - time.monotonic())
        if self.remaining_time is not None:
            limits.append(float(self.remaining_time()))
        if self._previous_remaining_time is not None:
            limits.append(float(self._previous_remaining_time()))
        return max(0.0, min(limits))

    def _stop(self, reason: str) -> None:
        self._broken = reason
        if self.mode == "memory":
            # Cooperative stop: a blocking HTTP call is allowed to finish and be recorded.
            # The server never abandons its worker; later pages/retries are refused.
            self.ctx.client.cancel_pending_requests()

    def _event(self, method: str, outcome: str, tool: str | None = None) -> None:
        event = {"method": method, "outcome": outcome}
        if tool is not None:
            event["tool"] = tool
        self.protocol_events.append(event)

    # -- lifetime ------------------------------------------------------------------

    async def __aenter__(self) -> "McpTools":
        if self._open:
            raise BridgeError("session already open")
        if self._broken:
            raise BridgeClosed(f"session unusable after: {self._broken}")
        if self.mode == "memory":
            from mcp import ClientSession
            from mcp.shared.memory import create_client_server_memory_streams

            self._stack = AsyncExitStack()
            try:
                server = self.server_factory(self.ctx)
                streams = await self._stack.enter_async_context(create_client_server_memory_streams())
                (read, write), (server_read, server_write) = streams
                self._memory_group = await self._stack.enter_async_context(anyio.create_task_group())
                self._memory_group.start_soon(server.run, server_read, server_write,
                                              server.create_initialization_options())
                self._previous_remaining_time = self.ctx.client.remaining_time
                self.ctx.client.remaining_time = self._client_remaining_time
                self._session = await self._stack.enter_async_context(ClientSession(read, write))
                with anyio.fail_after(self._allowance()):
                    await self._session.initialize()
                self._event("initialize", "ok")
            except BaseException as exc:
                cancelled = isinstance(exc, anyio.get_cancelled_exc_class())
                timeout = isinstance(exc, (TimeoutError, BridgeTimeout))
                outcome = "cancelled" if cancelled else "timeout" if timeout else "error"
                self._event("initialize", outcome)
                self._stop(f"{outcome} during MCP initialization")
                await self.__aexit__(type(exc), exc, exc.__traceback__)
                if timeout:
                    raise BridgeTimeout(self._broken) from exc
                raise
        self._open = True
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._open = False
        self._session = None
        if self.mode == "memory" and exc_type is not None:
            self._stop(self._broken or "interrupted MCP session")
        group, self._memory_group = self._memory_group, None
        if group is not None:
            group.cancel_scope.cancel()
        stack, self._stack = self._stack, None
        if stack is not None:
            # Close ClientSession, server task group, and streams in reverse entry order.
            try:
                await stack.aclose()
            except BaseExceptionGroup as group:
                # A late reply can hit a closing stream after a timeout; ignore only stream noise.
                # split(noise) -> (the noise, everything else); a callable predicate would be applied to
                # the nested TaskGroup itself and never match, so the plain type tuple is the right tool.
                noise = (anyio.BrokenResourceError, anyio.ClosedResourceError)
                if group.split(noise)[1] is not None:
                    raise
            finally:
                if self.mode == "memory":
                    self.ctx.client.remaining_time = self._previous_remaining_time

    def _require_open(self) -> None:
        if not self._open:
            raise BridgeClosed("McpTools is not open; use `async with McpTools...`")
        if self._broken:
            raise BridgeClosed(f"session unusable after: {self._broken}")

    # -- the two operations ------------------------------------------------------------

    async def schemas(self) -> list[dict[str, Any]]:
        """[{name, description, inputSchema}] exactly as the server advertises them."""
        self._require_open()
        if self.mode == "direct":
            from brapi_mcp_server import tool_schemas

            return await anyio.to_thread.run_sync(tool_schemas)
        try:
            with anyio.fail_after(self._allowance()):
                listed = await self._session.list_tools()
        except (TimeoutError, BridgeTimeout) as exc:
            self._stop("timeout while listing tools")
            self._event("tools/list", "timeout")
            raise BridgeTimeout(self._broken) from exc
        except anyio.get_cancelled_exc_class():
            self._stop("cancelled while listing tools")
            self._event("tools/list", "cancelled")
            raise
        except Exception as exc:
            self._stop("transport failure while listing tools")
            self._event("tools/list", "error")
            raise BridgeError(self._broken) from exc
        self._event("tools/list", "ok")
        schemas = []
        for tool in listed.tools:
            if not tool.name or not isinstance(tool.inputSchema, dict):
                raise BridgeError(f"server advertised a malformed tool: {tool!r}")
            schemas.append({"name": tool.name, "description": tool.description or "", "inputSchema": tool.inputSchema})
        return schemas

    async def model_tools(self) -> list[dict[str, Any]]:
        """schemas() in the model's function-tool shape."""
        return to_model_tools(await self.schemas())

    async def call(self, name: str, args: Mapping[str, Any] | None) -> ToolResult:
        """Run one tool and return its ToolResult. Raises BridgeTimeout / BridgeError; never fakes a result."""
        self._require_open()
        if not isinstance(name, str) or not name:
            raise BridgeError("tool name must be a non-empty string")
        arguments = dict(args or {})
        self.calls_made += 1
        if self.mode == "direct":
            from brapi_mcp_server import dispatch

            return await anyio.to_thread.run_sync(dispatch, self.ctx, name, arguments)
        try:
            allowance = self._allowance()
            self._operation_deadline = time.monotonic() + allowance
            with anyio.fail_after(allowance):
                reply = await self._session.call_tool(name, arguments,
                                                      read_timeout_seconds=timedelta(seconds=allowance))
        except anyio.get_cancelled_exc_class():
            self._stop(f"cancelled calling {name!r}")
            self._event("tools/call", "cancelled", name)
            raise
        except Exception as exc:
            code = getattr(getattr(exc, "error", None), "code", None)
            if isinstance(exc, (TimeoutError, BridgeTimeout)) or code == 408 or "timed out" in str(exc).lower() or "timeout" in type(exc).__name__.lower():
                self._stop(f"timeout calling {name!r}")
                self._event("tools/call", "timeout", name)
                raise BridgeTimeout(self._broken) from exc
            self._stop(f"transport failure calling {name!r}: {type(exc).__name__}")
            self._event("tools/call", "error", name)
            raise BridgeError(self._broken) from exc
        try:
            result = decode_call_result(reply, name)
        except BridgeError:
            self._stop(f"malformed reply calling {name!r}")
            self._event("tools/call", "error", name)
            raise
        self._operation_deadline = None
        self._event("tools/call", "ok" if result.ok else "rejected", name)
        return result



# --------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------

def decode_call_result(reply: Any, name: str) -> ToolResult:
    """Turn an MCP CallToolResult into a ToolResult, or raise BridgeError for malformed content."""
    structured = getattr(reply, "structuredContent", None)
    content = list(getattr(reply, "content", None) or [])
    is_error = bool(getattr(reply, "isError", False))

    if isinstance(structured, dict) and "ok" in structured:
        try:
            return ToolResult.model_validate_json(json.dumps(structured))
        except (ValidationError, ValueError) as exc:
            raise BridgeError(f"{name}: structured content is not a ToolResult: {exc.__class__.__name__}") from exc

    texts = [getattr(block, "text", None) for block in content if getattr(block, "type", None) == "text"]
    unsupported = [getattr(block, "type", type(block).__name__) for block in content if getattr(block, "type", None) != "text"]
    if not texts:
        if unsupported:
            raise BridgeError(f"{name}: unsupported content types {unsupported}; only text/structured are accepted")
        raise BridgeError(f"{name}: empty reply")
    text = texts[0] if isinstance(texts[0], str) else ""
    try:
        return ToolResult.model_validate_json(text)
    except (ValidationError, ValueError):
        pass
    if is_error:
        # the MCP layer itself refused (for example the server's inputSchema check); keep it as a coded error
        return ToolResult(ok=False, error=ToolError(code="invalid_argument", message=f"{name}: {text[:500] or 'rejected by the server'}"))
    raise BridgeError(f"{name}: reply text is neither a ToolResult nor an error: {text[:120]!r}")
