"""
llm.py — the model adapter and the bounded tool loop.

The one idea that matters: TEXT FROM A MODEL IS NOT EXECUTION AUTHORITY.
The model may *request* a tool call. This loop validates the tool name and the
argument object against the tool's own schema, checks authorization and every
budget, runs the tool through the bridge, and hands the result back. The model
never touches a file, a socket or an approval.

What is here:
* LlmSettings / load_llm_settings — LLM_BASE_URL, LLM_API_KEY, LLM_MODEL,
  LLM_ALLOW_REMOTE from .env, real environment winning. A non-loopback endpoint
  is refused unless LLM_ALLOW_REMOTE=true. There is no cloud fallback.
* ModelClient — the async interface the loop talks to. OpenAICompatibleClient
  implements it on the installed `openai` async SDK (local Ollama by default),
  with an explicit lifetime (`async with`) and no SDK auto-retries.
  FakeModelClient is a scripted stand-in for tests and the mock smoke test.
* Budget — model calls, tool calls, elapsed seconds, fetch tools, repairs.
  One Budget object is SHARED across agents and retries; it is never reset per
  model turn. A model turn containing 8 tool calls is 8 tool calls.
* run_agent_loop — the loop itself. Returns a typed AgentResult. Statuses:
  completed, needs_clarification, incomplete, blocked, failed, limit_reached.
  Length truncation, refusal, empty reply, invalid JSON, oversized output and
  provider errors are all different, and a partial completion is not success.
  One reply may use DEFAULT_MAX_TOKENS tokens, thinking included; more room
  does not change the rule that a reply which is still cut off is no answer.
* compact_tool_result — the summary a tool result becomes for the model: always
  valid JSON, with explicit "..._omitted" counts and the artifact/request
  handles kept. JSON text is never sliced at a character limit.
* --smoke --mock-model: a fake model asks add(2,3), receives 5, answers.
  --smoke --local: the same round trip with the configured local model (never
  BrAPI). Not run automatically.

Everyday example: a student (the model) may fill in request forms; the school
office (this loop) checks each form against its template, checks the student's
quota, files the request, and reports back. The student cannot open the filing
cabinet.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import uuid
from collections.abc import Awaitable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Literal, Protocol
from urllib.parse import urlsplit

import anyio
import jsonschema

from brapi_client import PART2_DIR
from contracts import (
    AgentResult,
    NormalizedMessage,
    StructuredError,
    TerminalStatus,
    ToolError,
    ToolResult,
    Usage,
    dump_json,
)
from mcp_bridge import BridgeError, BridgeTimeout

__all__ = [
    "LlmSettings", "LlmConfigError", "load_llm_settings", "is_loopback", "REPLY_LIMIT_PARAMS", "REASONING_EFFORTS",
    "ToolCallRequest", "ModelReply", "ModelProviderError", "ModelClient", "OpenAICompatibleClient", "FakeModelClient",
    "reply_text", "reply_tools", "normalize_openai_response",
    "Budget", "compact_tool_result", "run_agent_loop", "FETCH_TOOLS", "DEFAULT_MAX_TOKENS", "SmokeTools", "mock_smoke_model", "smoke", "main",
]

FETCH_TOOLS = frozenset({"get_observations", "get_observation_units"})   # the expensive, approval-guarded tools
# The most tokens ONE reply may use, thinking included. A reasoning model writes its thinking as tokens that count here although
# nobody sees them: on 2026-09-28, at 2048, a planner reply was cut off before its first visible word. This is a limit on what the
# model may WRITE; how much it can READ is the context window, a setting of the model server (16384 for the evaluation model).
# Reply limit plus the longest request must stay inside that window.
DEFAULT_MAX_TOKENS = 8192
# The request field that carries that limit. Ollama and older OpenAI-style servers read "max_tokens"; newer OpenAI models reject it
# ("Unsupported parameter ... Use 'max_completion_tokens' instead", HTTP 400, seen 2026-09-28). Chosen in .env, never guessed.
REPLY_LIMIT_PARAMS = ("max_tokens", "max_completion_tokens")
# Optional "reasoning_effort" sent with every request when LLM_REASONING_EFFORT is set (blank = not sent, the provider's default).
# Some OpenAI reasoning models refuse tools on the chat-completions endpoint unless it is "none" (HTTP 400, seen 2026-09-28).
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high")
_LOOPBACK = {"localhost", "127.0.0.1", "::1", "[::1]"}


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------

class LlmConfigError(Exception):
    """A settings problem. The message never contains the key."""


@dataclass(frozen=True)
class LlmSettings:
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"                 # a local placeholder Ollama accepts; never printed
    model: str | None = None
    allow_remote: bool = False
    request_timeout: float = 120.0
    reply_limit_param: str = "max_tokens"      # LLM_REPLY_LIMIT_PARAM: which request field carries the reply limit
    reasoning_effort: str | None = None        # LLM_REASONING_EFFORT: sent as "reasoning_effort" when set; None = not sent

    def __repr__(self) -> str:                 # keep the key out of logs and tracebacks
        return (f"LlmSettings(base_url={self.base_url!r}, api_key='***', model={self.model!r}, "
                f"allow_remote={self.allow_remote}, request_timeout={self.request_timeout}, reply_limit_param={self.reply_limit_param!r}, "
                f"reasoning_effort={self.reasoning_effort!r})")


def is_loopback(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return host in _LOOPBACK or host.startswith("127.")


def load_llm_settings(env_file: Path | None = None, environ: Mapping[str, str] | None = None) -> LlmSettings:
    """Read part2/.env then the real environment (which wins). Refuse a remote endpoint unless allowed."""
    from dotenv import dotenv_values

    # Offline tests and hosted deployments may explicitly ignore the local secrets file.
    # Explicit fixture paths still work for configuration-parser tests.
    environment = dict(os.environ if environ is None else environ)
    ignore_default = env_file is None and environment.get("PART2_IGNORE_DOTENV", "").strip().lower() == "true"
    env_file = PART2_DIR / ".env" if env_file is None else env_file
    values: dict[str, str] = {}
    if not ignore_default and env_file.is_file():
        values.update({k: v for k, v in dotenv_values(env_file).items() if v is not None})
    values.update(environment)
    base_url = values.get("LLM_BASE_URL", "http://localhost:11434/v1").strip().rstrip("/")
    allow_remote = values.get("LLM_ALLOW_REMOTE", "false").strip().lower() in ("1", "true", "yes")
    reply_limit_param = values.get("LLM_REPLY_LIMIT_PARAM", "max_tokens").strip() or "max_tokens"
    if reply_limit_param not in REPLY_LIMIT_PARAMS:
        raise LlmConfigError(f"LLM_REPLY_LIMIT_PARAM must be one of {list(REPLY_LIMIT_PARAMS)}, not {reply_limit_param!r}")
    reasoning_effort = values.get("LLM_REASONING_EFFORT", "").strip().lower() or None
    if reasoning_effort is not None and reasoning_effort not in REASONING_EFFORTS:
        raise LlmConfigError(f"LLM_REASONING_EFFORT must be blank or one of {list(REASONING_EFFORTS)}, not {reasoning_effort!r}")
    settings = LlmSettings(base_url=base_url, api_key=values.get("LLM_API_KEY", "ollama"),
                           model=(values.get("LLM_MODEL", "").strip() or None), allow_remote=allow_remote, reply_limit_param=reply_limit_param,
                           reasoning_effort=reasoning_effort)
    check_endpoint(settings)
    return settings


def check_endpoint(settings: LlmSettings) -> None:
    parts = urlsplit(settings.base_url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise LlmConfigError(f"LLM_BASE_URL is not a valid http(s) URL: {settings.base_url!r}")
    if not is_loopback(settings.base_url) and not settings.allow_remote:
        raise LlmConfigError(f"remote model endpoint {settings.base_url!r} refused: set LLM_ALLOW_REMOTE=true on purpose, "
                             "or point LLM_BASE_URL at the local server")


# --------------------------------------------------------------------------
# The model interface (async), normalized replies
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolCallRequest:
    id: str
    name: str
    arguments_json: str          # exactly what the model produced; parsed and validated by the loop


@dataclass(frozen=True)
class ModelReply:
    content: str | None
    tool_calls: list[ToolCallRequest]
    finish_reason: str | None    # stop | length | tool_calls | content_filter | ...
    model: str | None            # the model the SERVER says answered
    usage: Usage | None          # None = the provider reported nothing (unknown, not zero)
    refusal: str | None = None


class ModelProviderError(Exception):
    def __init__(self, code: str, message: str, *, status_code: int | None = None,
                 retryable: bool = False, retry_after_seconds: float | None = None) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable
        self.retry_after_seconds = retry_after_seconds


def _retry_after_seconds(headers: Mapping[str, str], *, now: datetime | None = None) -> float | None:
    """Only finite, nonnegative durations; never log raw headers. Keep long waits so callers can refuse them."""
    delays: list[float] = []
    for key, divisor in (("retry-after-ms", 1000.0), ("retry-after", 1.0)):
        raw = headers.get(key)
        try:
            seconds = float(raw) / divisor
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(seconds) and seconds >= 0:
            delays.append(seconds)
    try:
        target = parsedate_to_datetime(headers.get("retry-after", ""))
        if target.tzinfo is not None:
            seconds = (target - (now or datetime.now(timezone.utc))).total_seconds()
            if math.isfinite(seconds):
                delays.append(max(0.0, seconds))
    except (TypeError, ValueError, OverflowError):
        pass
    # Conflicting valid headers must not cause a retry before either requested wait.
    return max(delays) if delays else None


def _status_error(exc: Any) -> ModelProviderError:
    """Classify allowlisted structured fields, not provider prose (which can contain secrets or input)."""
    status = exc.status_code
    if status != 429:
        return ModelProviderError("http_error", f"provider returned HTTP {status}", status_code=status)
    body = getattr(exc, "body", None)
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        body = body["error"]
    fields = [getattr(exc, "code", None), getattr(exc, "type", None)]
    if isinstance(body, dict):
        fields += [body.get("code"), body.get("type")]
    codes = {value.strip().lower() for value in fields if isinstance(value, str)}
    if codes & {"insufficient_quota", "billing_hard_limit_reached", "billing_not_active"}:
        return ModelProviderError("quota", "The model provider reports an account quota or billing limit (HTTP 429). "
                                  "Check the provider account's quota and billing before trying again; no automatic retry was made.",
                                  status_code=429)
    headers = getattr(getattr(exc, "response", None), "headers", {})
    if codes & {"rate_limit_exceeded", "rate_limit_error", "slow_down"}:
        may_retry = headers.get("x-should-retry", "").lower() != "false"
        return ModelProviderError("rate_limit", "The model provider reports a temporary rate limit (HTTP 429). "
                                  "Wait before trying the question again.", status_code=429, retryable=may_retry,
                                  retry_after_seconds=_retry_after_seconds(headers))
    return ModelProviderError("rate_limit_unknown", "The model provider returned HTTP 429 without a recognized quota or rate-limit code. "
                              "The cause is unknown; check the provider account's quota and rate limits before trying again. "
                              "No automatic retry was made.", status_code=429)


class ModelClient(Protocol):
    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *, max_tokens: int) -> ModelReply: ...


class OpenAICompatibleClient:
    """The real adapter on the installed openai async SDK. Use `async with`. No auto-retries, explicit timeout."""

    def __init__(self, settings: LlmSettings) -> None:
        check_endpoint(settings)
        if not settings.model:
            raise LlmConfigError("LLM_MODEL is blank: inspect the installed model first (Stage 3.3), then set it in .env")
        self.settings = settings
        self._client: Any = None

    async def __aenter__(self) -> "OpenAICompatibleClient":
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(base_url=self.settings.base_url, api_key=self.settings.api_key,
                                   timeout=self.settings.request_timeout, max_retries=0)
        return self

    async def __aexit__(self, *exc) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.close()

    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *, max_tokens: int) -> ModelReply:
        if self._client is None:
            raise ModelProviderError("closed", "client is not open; use `async with`")
        import openai

        kwargs: dict[str, Any] = {"model": self.settings.model, "messages": messages, self.settings.reply_limit_param: max_tokens}
        if self.settings.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.settings.reasoning_effort
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        try:
            response = await self._client.chat.completions.create(**kwargs)
        except openai.APITimeoutError as exc:
            raise ModelProviderError("timeout", "the model did not answer in time") from exc
        except openai.APIConnectionError as exc:
            raise ModelProviderError("connection", f"could not reach {self.settings.base_url}") from exc
        except openai.APIStatusError as exc:
            raise _status_error(exc) from exc
        except openai.OpenAIError as exc:
            raise ModelProviderError("provider_error", type(exc).__name__) from exc
        return normalize_openai_response(response)


def normalize_openai_response(response: Any) -> ModelReply:
    """Turn the SDK object into our plain ModelReply. Missing usage stays None."""
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise ModelProviderError("empty_response", "provider returned no choices")
    choice = choices[0]
    message = getattr(choice, "message", None)
    calls: list[ToolCallRequest] = []
    for raw in getattr(message, "tool_calls", None) or []:
        function = getattr(raw, "function", None)
        calls.append(ToolCallRequest(
            id=str(getattr(raw, "id", "") or f"call_{uuid.uuid4().hex[:8]}"),
            name=str(getattr(function, "name", "") or ""),
            arguments_json=getattr(function, "arguments", "") or "",
        ))
    usage_obj = getattr(response, "usage", None)
    usage = None
    if usage_obj is not None and all(isinstance(getattr(usage_obj, k, None), int) for k in ("prompt_tokens", "completion_tokens", "total_tokens")):
        try:
            usage = Usage(prompt_tokens=usage_obj.prompt_tokens, completion_tokens=usage_obj.completion_tokens,
                          total_tokens=usage_obj.total_tokens)
        except Exception:  # noqa: BLE001 - inconsistent usage is "unknown", not a crash
            usage = None
    return ModelReply(content=getattr(message, "content", None), tool_calls=calls,
                      finish_reason=getattr(choice, "finish_reason", None), model=getattr(response, "model", None),
                      usage=usage, refusal=getattr(message, "refusal", None))


class FakeModelClient:
    """Scripted replies for tests and the mock smoke. Records every request it receives."""

    def __init__(self, replies: list[ModelReply], *, reported_model: str = "fake-model", gate: Any = None) -> None:
        self._replies = list(replies)
        self.reported_model = reported_model
        self.requests: list[dict[str, Any]] = []
        self.gate = gate                     # an anyio.Event: when set, complete() waits on it (for cancellation tests)

    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *, max_tokens: int) -> ModelReply:
        self.requests.append({"messages": [dict(m) for m in messages], "tools": [t["function"]["name"] for t in tools], "max_tokens": max_tokens})
        if self.gate is not None:
            await self.gate.wait()
        if not self._replies:
            raise ModelProviderError("script_exhausted", "the fake model has no more scripted replies")
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def reply_text(text: str, *, finish_reason: str = "stop", usage: Usage | None = None, model: str = "fake-model") -> ModelReply:
    return ModelReply(content=text, tool_calls=[], finish_reason=finish_reason, model=model, usage=usage)


def reply_tools(calls: list[tuple[str, str]], *, usage: Usage | None = None, model: str = "fake-model") -> ModelReply:
    """calls = [(tool_name, arguments_json), ...] exactly as a model would emit them."""
    return ModelReply(content=None, finish_reason="tool_calls", model=model, usage=usage,
                      tool_calls=[ToolCallRequest(id=f"call_{i}", name=n, arguments_json=a) for i, (n, a) in enumerate(calls)])


# --------------------------------------------------------------------------
# Budgets — shared, never reset per turn
# --------------------------------------------------------------------------

@dataclass
class Budget:
    max_model_calls: int = 12
    max_tool_calls: int = 30
    max_elapsed_seconds: float = 300.0
    max_fetches: int = 5              # calls to FETCH_TOOLS (the approval still governs each HTTP attempt)
    max_repairs: int = 2              # bounded repair of malformed model output
    model_calls: int = 0
    tool_calls: int = 0               # every REQUESTED call counts, dispatched or not
    fetches: int = 0
    repairs: int = 0
    started: float | None = None
    max_provider_retries: int = 2     # shared across agents; SDK retries stay disabled
    provider_retries: int = 0

    def start(self, clock: Callable[[], float]) -> None:
        if self.started is None:
            self.started = clock()

    def elapsed(self, clock: Callable[[], float]) -> float:
        return 0.0 if self.started is None else max(0.0, clock() - self.started)

    def exceeded(self, clock: Callable[[], float]) -> str | None:
        """Name of the first exhausted budget that blocks ANOTHER MODEL CALL, or None.

        The tool-call budget is checked per requested tool call, not here: a model that has used
        its tool budget may still give its final answer without tools.
        """
        if self.model_calls >= self.max_model_calls:
            return f"model calls ({self.max_model_calls})"
        if self.elapsed(clock) >= self.max_elapsed_seconds:
            return f"elapsed time ({self.max_elapsed_seconds:.0f}s)"
        return None


# --------------------------------------------------------------------------
# Compact, always-valid summaries of tool results for the model
# --------------------------------------------------------------------------

def compact_tool_result(result: ToolResult, *, max_items: int = 5, max_string: int = 400, max_depth: int = 4,
                        max_chars: int = 6000) -> dict[str, Any]:
    """A bounded JSON object for the model. Lists are cut with explicit omitted counts; long strings are
    replaced by a marker; handles (artifact_ids, request_ids) are always kept. Never a sliced JSON string."""

    def shrink(value: Any, depth: int) -> Any:
        if isinstance(value, str):
            return value if len(value) <= max_string else {"omitted_string_chars": len(value), "starts_with": value[:80]}
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        if depth >= max_depth:
            return {"omitted_nested": True}
        if isinstance(value, list):
            shown = [shrink(v, depth + 1) for v in value[:max_items]]
            if len(value) > max_items:
                shown.append({"omitted_items": len(value) - max_items})
            return shown
        if isinstance(value, dict):
            return {str(k): shrink(v, depth + 1) for k, v in value.items()}
        return str(value)

    summary: dict[str, Any] = {"ok": result.ok, "complete": result.complete,
                               "artifact_ids": list(result.artifact_ids),
                               "request_ids": list(result.request_ids[:max_items]),
                               "request_ids_omitted": max(0, len(result.request_ids) - max_items),
                               "warnings": [w[:max_string] for w in result.warnings[:max_items]],
                               "warnings_omitted": max(0, len(result.warnings) - max_items)}
    if result.error is not None:
        summary["error"] = {"code": result.error.code, "message": result.error.message[:max_string]}
    data = result.data
    page_key = None
    if (isinstance(data, dict) and isinstance(data.get("offset"), int)
            and "next_offset" in data and "total_matches" in data):
        page_key = next((key for key in ("candidates", "variables", "seasons", "programs")
                         if isinstance(data.get(key), list)), None)
    page_limit = max_items

    def model_data() -> Any:
        compact = shrink(data, 0)
        if page_key is not None:
            rows = data[page_key]
            shown = min(page_limit, len(rows))
            compact[page_key] = [shrink(row, 1) for row in rows[:shown]]
            compact["returned"] = shown
            compact["next_offset"] = data["offset"] + shown if shown < len(rows) else data["next_offset"]
            compact["display_complete"] = result.complete is True and compact["next_offset"] is None
            compact["model_view_note"] = ("Only this page is shown. Follow next_offset with the same filters; "
                                           "complete describes the source collection, not this page.")
        return compact

    if data is not None:
        summary["data"] = model_data()
    text = json.dumps(summary, ensure_ascii=True, allow_nan=False)
    # Wide candidate records can exceed the character cap. Reduce the displayed page
    # and move its continuation back, so no hidden records are silently skipped.
    while len(text) > max_chars and page_key is not None and page_limit > 1:
        page_limit -= 1
        summary["data"] = model_data()
        text = json.dumps(summary, ensure_ascii=True, allow_nan=False)
    if len(text) > max_chars and "data" in summary:
        keys = sorted(result.data.keys()) if isinstance(result.data, dict) else []
        summary["data"] = {"omitted_data": True, "keys": keys[:max_items], "keys_omitted": max(0, len(keys) - max_items)}
        summary["note"] = "data too large for a message; read the artifact by handle"
    return summary


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------

class ToolInterface(Protocol):
    async def schemas(self) -> list[dict[str, Any]]: ...
    async def call(self, name: str, args: Mapping[str, Any] | None) -> ToolResult: ...


FINAL_ANSWER_INSTRUCTIONS = (
    'When you are done, reply with ONLY a JSON object: {"status": "completed", "payload": {...}} or '
    '{"status": "needs_clarification", "question": "..."}. No prose around it.'
)


@dataclass
class _LoopState:
    messages: list[dict[str, Any]]
    normalized: list[NormalizedMessage] = field(default_factory=list)
    errors: list[StructuredError] = field(default_factory=list)
    model_reported: str | None = None
    usage_total: Usage | None = None
    usage_known: bool = True
    events: list[dict[str, Any]] = field(default_factory=list)


def _to_model_tools(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from mcp_bridge import to_model_tools

    return to_model_tools(schemas)


async def run_agent_loop(
    model: ModelClient,
    tools: ToolInterface,
    *,
    agent_name: str,
    system_prompt: str,
    user_message: str,
    budget: Budget,
    allowed_tools: set[str] | None = None,
    log_dir: Path,
    run_id: str,
    model_requested: str,
    clock: Callable[[], float] = time.monotonic,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_output_chars: int = 20000,
    sleep: Callable[[float], Awaitable[None]] = anyio.sleep,
    jitter: Callable[[], float] = random.random,
) -> AgentResult:
    """Drive one agent to a terminal AgentResult. See the module docstring for the rules."""
    budget.start(clock)
    log_dir = Path(log_dir)
    log_file = log_dir / run_id / f"llm_{agent_name}.jsonl"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_rel = f"{run_id}/llm_{agent_name}.jsonl"
    state = _LoopState(messages=[{"role": "system", "content": system_prompt + "\n\n" + FINAL_ANSWER_INSTRUCTIONS},
                                 {"role": "user", "content": user_message}])
    state.normalized.append(NormalizedMessage(role="system", content=system_prompt, correlation_id="setup"))
    state.normalized.append(NormalizedMessage(role="user", content=user_message, correlation_id="setup"))

    def emit(event: str, **fields: Any) -> None:
        record = {"ts_utc": datetime.now(timezone.utc).isoformat(), "agent": agent_name, "event": event, **fields}
        state.events.append(record)
        with open(log_file, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True, allow_nan=False, default=str) + "\n")

    def add_error(code: str, message: str) -> None:
        state.errors.append(StructuredError(code=code, message=message[:500], stage=agent_name))

    def finish(status: TerminalStatus, payload: dict[str, Any] | None) -> AgentResult:
        emit("terminal", status=status, correlation_id="terminal", model_calls=budget.model_calls, tool_calls=budget.tool_calls,
             errors=[e.code for e in state.errors])
        state.normalized.append(NormalizedMessage(role="system", content=f"terminal: {status}", correlation_id="terminal"))
        return AgentResult(status=status, payload=payload, messages=state.normalized, model_requested=model_requested,
                           model_reported=state.model_reported, model_calls=budget.model_calls, tool_calls=budget.tool_calls,
                           usage=state.usage_total if state.usage_known else None, elapsed_seconds=budget.elapsed(clock),
                           log_path=log_rel, errors=state.errors)

    schemas = await tools.schemas()
    schema_by_name = {s["name"]: s["inputSchema"] for s in schemas}
    model_tools = _to_model_tools([s for s in schemas if allowed_tools is None or s["name"] in allowed_tools])
    emit("start", run_id=run_id, tools=[t["function"]["name"] for t in model_tools], budget=dataclass_public(budget),
         max_tokens=max_tokens,                                      # the reply limit this run used, so a log says what actually ran
         reply_limit_param=getattr(getattr(model, "settings", None), "reply_limit_param", "max_tokens"),   # and the field that carried it
         reasoning_effort=getattr(getattr(model, "settings", None), "reasoning_effort", None))             # and the effort asked for, if any

    try:
        while True:
            exhausted = budget.exceeded(clock)
            if exhausted:
                add_error("budget_exhausted", f"stopped before a model call: {exhausted} budget used up")
                return finish("limit_reached", None)

            correlation = f"m{budget.model_calls + 1}"
            budget.model_calls += 1
            try:
                remaining = max(0.0, budget.max_elapsed_seconds - budget.elapsed(clock))
                with anyio.fail_after(remaining):
                    reply = await model.complete(state.messages, model_tools, max_tokens=max_tokens)
            except TimeoutError:
                state.usage_known = False
                add_error("budget_exhausted", "The run's elapsed-time budget expired while waiting for the model; no further request was sent.")
                return finish("limit_reached", None)
            except ModelProviderError as exc:
                state.usage_known = False  # failed-attempt token use is not known
                emit("provider_error", correlation_id=correlation, code=exc.code, status_code=exc.status_code,
                     retryable=exc.retryable, retry_after_seconds=exc.retry_after_seconds)
                if exc.code == "rate_limit" and exc.retryable and budget.provider_retries < budget.max_provider_retries:
                    delay = exc.retry_after_seconds
                    if delay is None or not math.isfinite(delay) or delay < 0:
                        noise = float(jitter())
                        noise = min(1.0, max(0.0, noise)) if math.isfinite(noise) else 0.0
                        delay = 2.0 ** min(budget.provider_retries + 1, 5) + noise
                    remaining = budget.max_elapsed_seconds - budget.elapsed(clock)
                    if budget.model_calls >= budget.max_model_calls or delay >= remaining:
                        add_error(f"provider_{exc.code}", exc.message)
                        add_error("budget_exhausted", "The remaining model-call or elapsed-time budget cannot fit a rate-limit retry.")
                        return finish("limit_reached", None)
                    if delay <= 60.0:
                        budget.provider_retries += 1
                        emit("provider_retry", correlation_id=correlation, delay_seconds=delay,
                             retry_number=budget.provider_retries, max_retries=budget.max_provider_retries)
                        await sleep(delay)  # cancellable; the next iteration rechecks every shared budget
                        continue           # same conversation; no prior tool call is repeated
                    add_error(f"provider_{exc.code}", exc.message + " The provider requested a wait longer than the 60-second automatic-retry limit; the wait was not shortened.")
                    return finish("failed", None)
                message = exc.message
                if exc.code == "rate_limit":
                    message += (" The shared automatic-retry limit has been reached." if exc.retryable else
                                " The provider disabled automatic retry for this response.")
                add_error(f"provider_{exc.code}", message)
                return finish("failed", None)
            state.model_reported = reply.model or state.model_reported
            if reply.usage is None:
                state.usage_known = False
            elif state.usage_known:
                prev = state.usage_total
                state.usage_total = reply.usage if prev is None else Usage(
                    prompt_tokens=prev.prompt_tokens + reply.usage.prompt_tokens,
                    completion_tokens=prev.completion_tokens + reply.usage.completion_tokens,
                    total_tokens=prev.total_tokens + reply.usage.total_tokens)
            content = reply.content or ""
            emit("assistant", correlation_id=correlation, finish_reason=reply.finish_reason, content_chars=len(content),
                 tool_calls=[c.name for c in reply.tool_calls], model=reply.model, usage=None if reply.usage is None else reply.usage.model_dump())
            state.normalized.append(NormalizedMessage(role="assistant", content=content, correlation_id=correlation,
                                                      name=",".join(c.name for c in reply.tool_calls) or None))

            if len(content) > max_output_chars:
                add_error("oversized_output", f"model produced {len(content)} characters; limit {max_output_chars}")
                return finish("failed", None)
            if reply.refusal or reply.finish_reason == "content_filter":
                add_error("model_refusal", (reply.refusal or "content filtered")[:200])
                return finish("blocked", None)
            if reply.finish_reason == "length":
                add_error("truncated", "the model hit its output length limit; a partial reply is not an answer")
                return finish("incomplete", None)

            if reply.tool_calls:
                # the assistant turn goes into the transcript as the SDK expects it
                state.messages.append({"role": "assistant", "content": reply.content,
                                       "tool_calls": [{"id": c.id, "type": "function",
                                                       "function": {"name": c.name, "arguments": c.arguments_json}}
                                                      for c in reply.tool_calls]})
                for call in reply.tool_calls:                      # sequential: no parallel fetching
                    budget.tool_calls += 1
                    tcorr = f"t{budget.tool_calls}"
                    emit("tool_request", correlation_id=tcorr, model_correlation=correlation, name=call.name,
                         arguments_chars=len(call.arguments_json))
                    if budget.tool_calls > budget.max_tool_calls:
                        add_error("budget_exhausted", f"tool call budget ({budget.max_tool_calls}) exceeded at {call.name!r}; "
                                                      f"{len(reply.tool_calls)} calls were requested in one turn")
                        return finish("limit_reached", None)
                    outcome = _validate_call(call, schema_by_name, allowed_tools)
                    if isinstance(outcome, ToolResult):
                        if outcome.error is not None and outcome.error.code == "invalid_argument":
                            budget.repairs += 1
                            if budget.repairs > budget.max_repairs:
                                add_error("malformed_arguments", f"{call.name!r}: {outcome.error.message}; repair budget used up")
                                return finish("failed", None)
                        _append_tool_message(state, call, outcome, tcorr, emit)
                        continue
                    args = outcome
                    if call.name in FETCH_TOOLS:
                        if budget.fetches >= budget.max_fetches:
                            add_error("budget_exhausted", f"fetch budget ({budget.max_fetches}) reached at {call.name!r}")
                            refused = ToolResult(ok=False, error=ToolError(code="budget_exhausted", message="fetch budget used up"))
                            _append_tool_message(state, call, refused, tcorr, emit)
                            return finish("limit_reached", None)
                        budget.fetches += 1
                    if budget.elapsed(clock) >= budget.max_elapsed_seconds:
                        add_error("budget_exhausted", f"elapsed time ({budget.max_elapsed_seconds:.0f}s) used up before dispatching {call.name!r}")
                        return finish("limit_reached", None)
                    try:
                        result = await tools.call(call.name, args)
                    except BridgeTimeout as exc:
                        add_error("tool_timeout", str(exc))
                        return finish("failed", None)
                    except BridgeError as exc:
                        add_error("tool_transport", str(exc))
                        return finish("failed", None)
                    _append_tool_message(state, call, result, tcorr, emit)
                continue

            # no tool calls: this must be the final JSON answer
            if not content.strip():
                add_error("empty_reply", "the model returned no content and no tool call")
                return finish("failed", None)
            parsed = _parse_final(content)
            if parsed is None:
                budget.repairs += 1
                if budget.repairs > budget.max_repairs:
                    add_error("invalid_json", "final reply was not the required JSON object; repair budget used up")
                    return finish("failed", None)
                state.messages.append({"role": "assistant", "content": content})
                state.messages.append({"role": "user", "content": "Your reply was not a valid JSON object. " + FINAL_ANSWER_INSTRUCTIONS})
                emit("repair", correlation_id=correlation, reason="final reply not JSON object")
                continue
            status = parsed.get("status")
            if status == "completed":
                payload = parsed.get("payload")
                if not isinstance(payload, dict):
                    add_error("invalid_json", "status completed without an object payload")
                    return finish("failed", None)
                try:   # the contract rejects authority keys and NaN inside a payload; check before declaring success
                    AgentResult(status="completed", payload=payload, model_requested=model_requested, model_calls=0,
                                tool_calls=0, elapsed_seconds=0.0)
                except Exception as exc:  # noqa: BLE001 - pydantic ValidationError
                    first = exc.errors()[0]["msg"] if hasattr(exc, "errors") else str(exc)
                    add_error("payload_rejected", str(first)[:300])
                    return finish("failed", None)
                return finish("completed", payload)
            if status == "needs_clarification":
                question = parsed.get("question")
                if not isinstance(question, str) or not question.strip():
                    add_error("invalid_json", "needs_clarification without a question")
                    return finish("failed", None)
                return finish("needs_clarification", {"question": question})
            add_error("invalid_json", f"final reply has unknown status {status!r}")
            return finish("failed", None)
    except anyio.get_cancelled_exc_class():
        emit("cancelled", correlation_id="terminal", model_calls=budget.model_calls, tool_calls=budget.tool_calls)
        raise


def _validate_call(call: ToolCallRequest, schema_by_name: dict[str, dict[str, Any]], allowed_tools: set[str] | None) -> ToolResult | dict[str, Any]:
    """Return validated args, or a ToolResult error to hand back to the model. Nothing is dispatched here."""
    if call.name not in schema_by_name:
        return ToolResult(ok=False, error=ToolError(code="invalid_argument", message=f"unknown tool {call.name!r}"))
    if allowed_tools is not None and call.name not in allowed_tools:
        return ToolResult(ok=False, error=ToolError(code="not_authorized", message=f"tool {call.name!r} is not available to this agent"))
    try:
        parsed = json.loads(call.arguments_json) if call.arguments_json.strip() else {}
    except ValueError:
        return ToolResult(ok=False, error=ToolError(code="invalid_argument", message="arguments are not valid JSON"))
    if not isinstance(parsed, dict):
        return ToolResult(ok=False, error=ToolError(code="invalid_argument", message="arguments must be a JSON object, not a list or scalar"))
    try:
        jsonschema.Draft202012Validator(schema_by_name[call.name]).validate(parsed)
    except jsonschema.ValidationError as exc:
        where = ".".join(str(p) for p in exc.absolute_path) or "(root)"
        return ToolResult(ok=False, error=ToolError(code="invalid_argument", message=f"{where}: {exc.message}"[:400]))
    return parsed


def _append_tool_message(state: _LoopState, call: ToolCallRequest, result: ToolResult, tcorr: str, emit: Callable[..., None]) -> None:
    summary = compact_tool_result(result)
    text = json.dumps(summary, ensure_ascii=True, allow_nan=False)
    state.messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": text})
    state.normalized.append(NormalizedMessage(role="tool", content=text, name=call.name, tool_call_id=call.id, correlation_id=tcorr))
    emit("tool_result", correlation_id=tcorr, name=call.name, ok=result.ok, error=None if result.error is None else result.error.code,
         artifact_ids=list(result.artifact_ids), summary_chars=len(text))


def _parse_final(content: str) -> dict[str, Any] | None:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        parsed = json.loads(text)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def dataclass_public(obj: Any) -> dict[str, Any]:
    return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}


# --------------------------------------------------------------------------
# Smoke harness: the only place an `add` tool exists
# --------------------------------------------------------------------------

class SmokeTools:
    """One tool, add(a, b). Exists ONLY here; it is never part of the BrAPI server."""

    SCHEMA = {"name": "add", "description": "Add two integers.",
              "inputSchema": {"type": "object", "additionalProperties": False, "required": ["a", "b"],
                              "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}}}

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def schemas(self) -> list[dict[str, Any]]:
        return [self.SCHEMA]

    async def call(self, name: str, args: Mapping[str, Any] | None) -> ToolResult:
        args = dict(args or {})
        self.calls.append({"name": name, "args": args})
        if name != "add":
            return ToolResult(ok=False, error=ToolError(code="invalid_argument", message=f"unknown tool {name!r}"))
        return ToolResult(ok=True, data={"sum": int(args["a"]) + int(args["b"])}, complete=True)


SMOKE_SYSTEM = ("You are a calculator assistant. Use the add tool for any addition. Never guess. "
                'Your final payload must be {"answer": <the number returned by the tool>}.')
SMOKE_QUESTION = "What is 2 + 3? Use the tool."


def _smoke_answer_found(payload: dict[str, Any] | None, expected: int) -> bool:
    """True if the expected number appears as a numeric value anywhere at the top level of the payload.
    A model may label it 'answer', 'sum' or 'result'; the label is not what the smoke test checks."""
    if not isinstance(payload, dict):
        return False
    return any(isinstance(v, (int, float)) and not isinstance(v, bool) and v == expected for v in payload.values())


def mock_smoke_model() -> FakeModelClient:
    return FakeModelClient([
        reply_tools([("add", json.dumps({"a": 2, "b": 3}))], model="mock-model"),
        reply_text(json.dumps({"status": "completed", "payload": {"answer": 5, "explanation": "add(2,3) returned 5"}}), model="mock-model"),
    ], reported_model="mock-model")


async def smoke(model_kind: Literal["mock", "local"], out_dir: Path, log_fn: Callable[[str], None] = print) -> int:
    tools = SmokeTools()
    budget = Budget(max_model_calls=4, max_tool_calls=4, max_elapsed_seconds=180.0)
    run_id = f"smoke_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:6]}"
    if model_kind == "mock":
        model: Any = mock_smoke_model()
        result = await run_agent_loop(model, tools, agent_name="smoke", system_prompt=SMOKE_SYSTEM, user_message=SMOKE_QUESTION,
                                      budget=budget, log_dir=out_dir, run_id=run_id, model_requested="mock-model")
        real_calls = 0
    else:
        settings = load_llm_settings()
        async with OpenAICompatibleClient(settings) as client:
            result = await run_agent_loop(client, tools, agent_name="smoke", system_prompt=SMOKE_SYSTEM, user_message=SMOKE_QUESTION,
                                          budget=budget, log_dir=out_dir, run_id=run_id, model_requested=settings.model or "")
        real_calls = result.model_calls
    ok = result.status == "completed" and _smoke_answer_found(result.payload, 5) and tools.calls == [{"name": "add", "args": {"a": 2, "b": 3}}]
    log_fn(f"SMOKE ({model_kind} model): status={result.status}  model_calls={result.model_calls}  tool_calls={result.tool_calls}  "
           f"real_model_requests={real_calls}  usage={'unknown' if result.usage is None else result.usage.model_dump()}")
    log_fn(f"  tool round trip: {tools.calls} -> payload {result.payload}")
    log_fn(f"  model requested={result.model_requested!r} reported={result.model_reported!r}  elapsed={result.elapsed_seconds:.2f}s")
    log_fn(f"  log: {out_dir / result.log_path}")
    log_fn("  BrAPI calls: none (the smoke harness has only the add tool)")
    if ok:
        log_fn("SMOKE PASSED")
    else:
        why = [e.model_dump() for e in result.errors] or (
            f"status={result.status}, payload={result.payload!r}, tool calls={tools.calls} (expected the number 5 in the payload and exactly one add(2,3))")
        log_fn(f"SMOKE FAILED: {why}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m llm", description="Model adapter and bounded tool loop — smoke tests.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--mock-model", action="store_true", help="fake model: zero real model requests")
    parser.add_argument("--local", action="store_true", help="the configured LOCAL model (needs LLM_MODEL); never BrAPI")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)
    if not args.smoke or args.mock_model == args.local:
        print("use: python -m llm --smoke --mock-model   or   python -m llm --smoke --local", file=sys.stderr)
        return 2
    out_dir = Path(args.out_dir) if args.out_dir else PART2_DIR / "out"
    try:
        return anyio.run(smoke, "mock" if args.mock_model else "local", out_dir)
    except LlmConfigError as exc:
        print(f"CONFIG: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
