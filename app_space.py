"""Hugging Face entrypoint for the agent-led, human-reviewed MCP application.

Imports build the queued Gradio demo for Spaces hot reload; only main launches
a server. The hosted app is self-contained and uses the shared MCP service.
"""
from __future__ import annotations

from dataclasses import replace
import os
from typing import Any, Mapping

# A Space uses its configured variables/secrets, never a packaged dotenv file.
# Set this before importing modules that can construct settings for the UI.
if os.environ.get("SPACE_ID"):
    os.environ["PART2_IGNORE_DOTENV"] = "true"
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "false")

import app
from run import PART2_DIR, RunConfig

__all__ = ["space_config", "space_preflight", "space_launch_options", "build_demo", "demo", "main"]

_SDK_ERROR = ("The hosted MCP app requires the MCP SDK memory transport. "
              "Check the Space dependencies; the app will not fall back to direct calls.")


def require_memory_transport() -> None:
    """Refuse an SDK that cannot support the hosted MCP connection."""
    try:
        from mcp.shared.memory import create_client_server_memory_streams
    except ImportError:
        raise RuntimeError(_SDK_ERROR) from None
    if not callable(create_client_server_memory_streams):
        raise RuntimeError(_SDK_ERROR)


def space_config(question: str) -> RunConfig:
    """The hosted workflow, with separate run and cache folders."""
    return replace(app.live_config(question), direct=False, mcp_transport="memory",
                   review_retrieved_data=True, agent_led=True, unified_mcp=True,
                   out_dir=PART2_DIR / "out" / "space_mcp", cache_dir=PART2_DIR / "cache" / "space_mcp")


def space_preflight(question: str) -> list[str]:
    """Keep live/model/access checks, and fail closed if memory MCP is missing."""
    try:
        require_memory_transport()
    except RuntimeError:
        return [_SDK_ERROR]
    return app.live_preflight(question, make_config=space_config)


def space_launch_options(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Preserve app login/open settings and let Gradio own the runtime port.

Gradio defaults to port 7860 and honors GRADIO_SERVER_PORT, including the
internal port arrangements used by Spaces SSR. Do not hard-code that port here.
"""
    values = os.environ if environ is None else environ
    options = app.launch_options(values)
    options.update(server_name="0.0.0.0" if values.get("SPACE_ID") else "127.0.0.1",
                   share=False, inbrowser=False)
    return options


def build_demo():
    """Build and queue the interface without starting a server or an agent run."""
    try:
        require_memory_transport()
    except RuntimeError:
        raise SystemExit(_SDK_ERROR) from None
    return app.build_ui(make_config=space_config, preflight=space_preflight).queue(
        max_size=app.CHAT_QUEUE_SIZE, default_concurrency_limit=1
    )


# Spaces looks for this exact object when it reloads the application module.
demo = build_demo()


def main() -> None:
    demo.launch(**space_launch_options(), theme=app.build_theme(), css=app.CSS)


if __name__ == "__main__":
    main()
