"""
mcp_bridge.py — one session to the twelve tools, for the agents.

An agent loop (Task 3B) asks two things: "what tools exist, with what arguments?"
and "run this tool with these arguments". McpTools answers both through ONE
async, context-managed session that lives for the whole run:

    async with McpTools.mcp(server_args=[...]) as tools:      # real MCP: a stdio child process
        schemas = await tools.schemas()                        # the server's advertised tools
        result = await tools.call("get_study", {"study_db_id": "S1"})   # -> ToolResult

    async with McpTools.memory(ctx) as tools:                  # real MCP, local SDK memory streams
        ...

    async with McpTools.direct(ctx) as tools:                  # same tools, same validators, no transport
        ...

Design rules:
* The server child is started with sys.executable and the server file's resolved
  absolute path; it inherits an explicit environment (offline by default).
* One event-loop lifetime per run. The caller does exactly one asyncio.run (or
  anyio.run); this module never starts loops of its own.
* Tool schemas are converted to the model's function-tool shape by wrapping the
  server's own inputSchema — no second schema definition exists anywhere.
* Replies are decoded deliberately: structured content or text that is a valid
  ToolResult -> ToolResult; an MCP-level error with plain text (for example the
  server's schema check) -> ToolResult(ok=False, invalid_argument); anything
  malformed or unsupported (images, empty content, invalid JSON) -> BridgeError,
  never silently turned into an "ok" result.
* A timeout raises BridgeTimeout and marks the session unusable; leaving the
  context closes the child, also after errors and cancellation.

Everyday example: a phone line to the library desk. You dial once (the session),
ask "what forms do you have?" (schemas) and "please process form 4" (call). If the
line drops or the desk stalls, you hear a clear message — not a fabricated answer.
"""
from __future__ import annotations

import json
import math
import time
import os
import sys
from collections.abc import Callable, Mapping
from contextlib import AsyncExitStack
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

import anyio
from pydantic import ValidationError

from brapi_client import PART2_DIR
from contracts import ToolError, ToolResult

__all__ = ["SERVER_PATH", "BridgeError", "BridgeTimeout", "BridgeClosed", "McpTools", "to_model_tools"]

SERVER_PATH = (PART2_DIR / "brapi_mcp_server.py").resolve()
OFFLINE_ENV = {"BRAPI_MODE": "offline", "LLM_ALLOW_REMOTE": "false"}


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
        mode: Literal["mcp", "direct", "memory"],
        server_args: list[str] | None = None,
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
        ctx: Any = None,
        call_timeout: float = 60.0,
        server_path: Path = SERVER_PATH,
        python: str = sys.executable,
        remaining_time: Callable[[], float] | None = None,
        server_factory: Callable[[Any], Any] | None = None,
    ) -> None:
        if mode not in ("mcp", "direct", "memory"):
            raise BridgeError(f"unknown mode {mode!r}")
        if mode in ("direct", "memory") and ctx is None:
            raise BridgeError(f"{mode} mode needs a ToolContext (brapi_mcp_server.build_context)")
        if server_factory is not None and mode != "memory":
            raise BridgeError("a custom server factory is only supported in memory mode")
        if not math.isfinite(call_timeout) or call_timeout <= 0:
            raise BridgeError("call_timeout must be positive")
        self.mode = mode
        self.server_args = list(server_args or ["--offline", "--fixture", "synthetic"])
        self.env = {**os.environ, **OFFLINE_ENV, **(dict(env) if env else {})}
        self.cwd = Path(cwd) if cwd else PART2_DIR
        self.ctx = ctx
        self.call_timeout = call_timeout
        self.remaining_time = remaining_time
        self.server_factory = server_factory
        self.protocol_events: list[dict[str, str]] = []
        self._memory_group: Any = None
        self._operation_deadline: float | None = None
        self._previous_remaining_time: Callable[[], float] | None = None
        self.server_path = Path(server_path).resolve()
        self.python = python
        self._stack: AsyncExitStack | None = None
        self._session: Any = None
        self._open = False
        self._broken: str | None = None
        self.calls_made = 0

    # -- constructors ------------------------------------------------------------

    @classmethod
    def mcp(cls, *, server_args: list[str] | None = None, env: Mapping[str, str] | None = None,
            cwd: Path | None = None, call_timeout: float = 60.0, server_path: Path = SERVER_PATH) -> "McpTools":
        return cls(mode="mcp", server_args=server_args, env=env, cwd=cwd, call_timeout=call_timeout, server_path=server_path)

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
        if self.mode in ("mcp", "memory"):
            from mcp import ClientSession, StdioServerParameters

            self._stack = AsyncExitStack()
            try:
                if self.mode == "memory":
                    from mcp.shared.memory import create_client_server_memory_streams
                    from brapi_mcp_server import build_server

                    server = self.server_factory(self.ctx) if self.server_factory is not None else build_server(self.ctx)
                    streams = await self._stack.enter_async_context(create_client_server_memory_streams())
                    (read, write), (server_read, server_write) = streams
                    self._memory_group = await self._stack.enter_async_context(anyio.create_task_group())
                    self._memory_group.start_soon(server.run, server_read, server_write,
                                                  server.create_initialization_options())
                    self._previous_remaining_time = self.ctx.client.remaining_time
                    self.ctx.client.remaining_time = self._client_remaining_time
                else:
                    from mcp.client.stdio import stdio_client

                    if not self.server_path.is_file():
                        raise BridgeError(f"server file not found: {self.server_path}")
                    params = StdioServerParameters(command=self.python, args=[str(self.server_path), *self.server_args],
                                                   env=self.env, cwd=str(self.cwd))
                    read, write = await self._stack.enter_async_context(stdio_client(params))
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
            # Exits stdio_client and ClientSession in the order they were entered (anyio requires this);
            # stdio_client's own teardown closes stdin, waits, then terminates the child if needed.
            try:
                await stack.aclose()
            except BaseExceptionGroup as group:
                # After a timeout the child's LATE reply can hit a stream that is already closing.
                # stdio_client has already terminated the child; swallow only that stream noise.
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


# --------------------------------------------------------------------------
# Small self-check CLI: one asyncio.run for the whole run
# --------------------------------------------------------------------------

async def _selfcheck(server_args: list[str], call_timeout: float) -> int:
    async with McpTools.mcp(server_args=server_args, call_timeout=call_timeout) as tools:
        schemas = await tools.schemas()
        print(f"{len(schemas)} tools advertised: {[s['name'] for s in schemas]}")
        info = await tools.call("server_info", {})
        print(f"server_info ok={info.ok} server={info.data.get('server_name') if info.data else None} live_requests=0 (offline)")
        return 0 if len(schemas) == 12 and info.ok else 1


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m mcp_bridge", description="Bridge self-check over a real stdio child (offline).")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--fixture", default="synthetic")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--call-timeout", type=float, default=60.0)
    args = parser.parse_args(argv)
    server_args = ["--offline", "--fixture", args.fixture]
    if args.cache_dir:
        server_args += ["--cache-dir", args.cache_dir]
    if args.out_dir:
        server_args += ["--out-dir", args.out_dir]
    return anyio.run(_selfcheck, server_args, args.call_timeout)     # ONE loop for the whole run


if __name__ == "__main__":
    sys.exit(main())
