"""Shared chat interface for BreedAPI Chat Agent on Hugging Face Spaces.

app_space.py selects agent-led coordination, the shared MCP service, and human
review before retrieval and analysis. Each browser session has its own run;
the Full report retains sources, calculations, answers and review decisions.
"""
from __future__ import annotations

import asyncio
import os
import math
import re
import queue
import threading
from typing import Any, Callable, Iterator, Mapping
from urllib.parse import urlsplit

from app_style import CSS, HERO_HTML, build_theme
from app_help import QUICK_START, QUESTION_GUIDE, faq_items
from chat_presenter import answer_summary, approval_summary
from agents.coordinator import clarifications_given
from brapi_client import load_settings
from run import (ANALYST_PROMPT, CLARIFY_PROMPT, CATALOG_PROMPT, DATA_REVIEW_PROMPT, RETRIEVER_PROMPT, SUPERVISOR_PROMPT,
                 Controller, Models, RunConfig, analyst_questions)

__all__ = ["Job", "live_config", "live_preflight", "respond", "render", "build_ui", "REPLY_TIMEOUT_S", "locked", "after_step",
           "reply_buttons", "start_over"]

REPLY_TIMEOUT_S = 15 * 60          # how long a run waits for your reply in the chat before it cancels itself
STEP_TIMEOUT_S = 15 * 60           # how long the page waits for the run's next step (model calls, fetches)
WORKING = "Working on your question…"
PLACEHOLDER = ("**Start with a question.**\n\nFind a study, count locations, or explore a trait in one trial.")
CHAT_CLARIFICATION_ROUNDS = 3      # how many times a person may answer the planner in the chat (the command line allows 1)
SHOWN_QUESTION_CHARS = 600         # a planner question can list hundreds of variables; the chat shows the start, the notes keep all
CHAT_CONCURRENCY_ID = "chat_steps"
CHAT_QUEUE_SIZE = 20
TABLE_ROWS_SHOWN = 15              # rows of a long table shown in the chat; answer.md keeps every row
EXAMPLES = (                       # the first was answered live on 2026-09-29 (run_20260929T183240_3aec99); the others are catalog questions
    "In study 05DISE178HFLNO7, what is the pooled mean of the variable 'Reaction to Fusarium Wilt estimating 0-5'?",
    "Which studies have a 2026 season?",
    "How many locations are in the catalog?",
)
Message = dict[str, str]           # {"role": "user" | "assistant", "content": markdown}


class Job:
    """One question: its controller runs in a background thread and talks to the chat through two queues.

    inbox  : the person's replies, typed in the chat ("approve", "cancel", "accept", ...)
    outbox : what the run needs next — ("prompt", text), ("done", RunResult) or ("error", message)
    """

    def __init__(self, question: str, *, make_config: Callable[[str], RunConfig], make_models: Callable[[], Models],
                 reply_timeout_s: float = REPLY_TIMEOUT_S) -> None:
        self.inbox: queue.Queue[str] = queue.Queue()
        self.outbox: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.reply_timeout_s = reply_timeout_s
        self.answer_shown = False
        self.waiting_on = ""                     # the prompt the run is waiting on; set by the run's thread before it is shown
        self.controller = Controller(make_config(question), models=make_models(), answer_fn=self._ask, answers_from_human=True)
        self.thread = threading.Thread(target=self._run, name=f"job-{self.controller.run_id}", daemon=True)

    def _ask(self, prompt: str) -> str:
        self.waiting_on = prompt
        self.outbox.put(("prompt", prompt))
        try:
            return self.inbox.get(timeout=self.reply_timeout_s)
        except queue.Empty:
            raise EOFError("no reply in the chat within the time limit") from None   # the controller treats it as no answer

    def _run(self) -> None:
        try:
            self.outbox.put(("done", asyncio.run(self.controller.run())))
        except BaseException as exc:  # noqa: BLE001 - every ending reaches the page, none is lost
            self.outbox.put(("error", f"{type(exc).__name__}: {exc}"))

    def start(self) -> "Job":
        self.thread.start()
        return self

    def reply(self, text: str) -> None:
        if self.waiting_on == CATALOG_PROMPT and text.strip().lower() == "approve":
            text = "approve catalog"
        if self.waiting_on == DATA_REVIEW_PROMPT and text.strip().lower() == "approve":
            text = "continue"               # the first quick-reply button is labelled Continue to analysis here
        if text.strip().lower() == "cancel" and self.waiting_on in (CLARIFY_PROMPT, ANALYST_PROMPT, RETRIEVER_PROMPT, SUPERVISOR_PROMPT):
            text = ""                            # there the controller's rule is 'empty = cancel'; a chat box cannot send an empty line
        self.inbox.put(text)

    def next_event(self, timeout: float = STEP_TIMEOUT_S) -> tuple[str, Any]:
        try:
            return self.outbox.get(timeout=timeout)
        except queue.Empty:
            return ("error", f"no progress within {timeout:.0f} s; the run keeps its evidence under out/{self.controller.run_id}")

    def close(self) -> None:
        """The tab went away: release a waiting prompt so the run ends (as canceled or pending) instead of waiting."""
        self.reply("cancel")


# --------------------------------------------------------------------------
# What a live question needs, and how its steps are shown
# --------------------------------------------------------------------------

def _app_number(name: str, default: str, low: float, high: float, *, integer: bool = False):
    """Refuse bad deployment settings without echoing their values."""
    try:
        value = int(os.environ.get(name, default)) if integer else float(os.environ.get(name, default))
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a {'whole number' if integer else 'number'} from {low:g} to {high:g}") from None
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be a {'whole number' if integer else 'number'} from {low:g} to {high:g}")
    return value


def live_config(question: str) -> RunConfig:
    return RunConfig(question=question, offline=False, mock_model=False, direct=False, auto=False,
                     max_fetches=5, max_http_attempts=_app_number("APP_MAX_HTTP_ATTEMPTS", "10", 1, 100, integer=True),
                     max_clarification_rounds=CHAT_CLARIFICATION_ROUNDS,
                     read_timeout=_app_number("APP_READ_TIMEOUT", "90", 1, 600), max_elapsed_seconds=600,
                     bootstrap_catalogs=True)


def default_models() -> Models:
    return Models(requested="configured model")


def live_preflight(question: str, make_config: Callable[[str], RunConfig] = live_config) -> list[str]:
    """The command line's refusals, said in the chat before any model call or request."""
    try:
        problems = make_config(question).validate()
    except (TypeError, ValueError, OverflowError):
        return ["App settings are invalid. Check APP_MAX_HTTP_ATTEMPTS, APP_READ_TIMEOUT and the BrAPI settings."]
    try:
        from llm import is_loopback, load_llm_settings

        settings = load_llm_settings()
        if not settings.model:
            problems.append("LLM_MODEL is missing; set the model name before starting a question")
        if not is_loopback(settings.base_url) and settings.api_key.strip().lower() in ("", "ollama"):
            problems.append("LLM_API_KEY must be set to a real key for this remote-model chat")
    except Exception:  # a model endpoint that cannot be used is a stated reason, not a crash
        problems.append("the model endpoint cannot be used: check LLM_BASE_URL, LLM_MODEL, LLM_API_KEY and LLM_ALLOW_REMOTE")
    return problems


def clarification_choices(questions: list[str], metadata: Any) -> str:
    """Hints from validated catalogs, only displayed; never added to the person's words."""
    text = " ".join(questions).lower()
    lines = []
    if re.search(r"\blocations?\b", text) and metadata is not None:
        locations = {}
        for study in metadata.studies:
            if not study.location_id:
                continue
            item = locations.setdefault(study.location_id, {"names": set(), "studies": set()})
            item["studies"].add(study.study_id)
            if study.location_name:
                item["names"].add(study.location_name)
        ranked = sorted(locations.items(), key=lambda item: (-len(item[1]["studies"]), item[0]))[:10]
        if ranked:
            lines += ["", "Locations with the most studies: These are popular database locations, not confirmed matches "
                      "for the place you requested. Choosing one changes the location scope of your question."]
            for location_id, item in ranked:
                count = len(item["studies"])
                name = " / ".join(sorted(item["names"])) or "(name not stated)"
                lines.append(f"- {location_id} · {name} — {count} {'study' if count == 1 else 'studies'}")
    if re.search(r"\b(traits?|variables?)\b", text):
        lines += ["", "You can ask 'list all traits' to see their names."]
    return "\n".join(lines)


def render(job: Job, event: tuple[str, Any]) -> tuple[str, bool]:
    """(markdown for the chat, finished?) for one step of the run."""
    kind, payload = event
    controller = job.controller
    if kind == "prompt" and payload == CATALOG_PROMPT:
        return (controller.catalog_approval_screen() + "\n\nReply **approve** to prepare these catalogs, or **cancel**. "
                "Measurements require a separate approval afterward."), False
    if kind == "prompt" and str(payload).startswith("approve / edit / cancel"):
        return (approval_summary(controller) + "\n\nReply **approve** to use this scope, or **cancel**. "
                "You can also type **edit** to change the study IDs, or describe a change in your own words. "
                "A change produces a new plan and asks for approval again before fetching."), False
    if kind == "prompt" and str(payload).startswith("accept / reject"):
        job.answer_shown = True
        return (answer_summary(controller) + "\n\n---\nReply **accept** if this answers your question, or **reject**. "
                "The Full report below keeps the complete evidence."), False
    if kind == "prompt" and payload == DATA_REVIEW_PROMPT:
        return (controller.data_review_screen() + "\n\nReply **continue** to analyze these data, or **cancel**. "
                "Continuing allows analysis; you will review and accept the final answer separately."), False
    if kind == "prompt" and payload == SUPERVISOR_PROMPT:
        asked = "\n".join("> " + _shorten(" ".join(q.split())) for q in controller.supervisor_questions)
        return ("The coordinator needs a detail before choosing the next step:\n\n" + asked +
                "\n\nReply in the chat, or cancel. Any revised data request still needs your approval."), False
    if kind == "prompt" and payload == RETRIEVER_PROMPT:
        asked = [_shorten(" ".join(q.split())) for q in controller.retriever_questions if q and q.strip()]
        shown = "\n".join(f"> {q}" for q in asked) or "> (no question was given; reply cancel to stop)"
        number = controller.retriever_answers + 1
        return (f"The retriever needs one more detail while preparing your data (answer {number} of "
                f"{controller.config.max_clarification_rounds}):\n\n{shown}\n\n"
                "Your answer resumes retrieval within the approved scope. To change that scope, "
                "reply **cancel** and start a new question."), False
    if kind == "prompt" and payload in (CLARIFY_PROMPT, ANALYST_PROMPT):
        planner = payload == CLARIFY_PROMPT
        questions = (controller.coordination.questions if controller.coordination else []) if planner else analyst_questions(controller.analysis)
        asked = [_shorten(" ".join(q.split())) for q in questions if q and q.strip()]
        shown = "\n".join(f"> {q}" for q in asked) or "> (no question was given; say in a few more words what you mean)"
        number = (max(0, clarifications_given(controller.config.question) - getattr(controller, "approval_revisions", 0))
                  if planner else controller.analyst_answers) + 1
        if planner:
            who = "The planner needs more detail before it can make a plan"
            hint = "Your answer is added to your question, so a few words are enough, for example **study 4501** or **all studies from 2026**."
            scope = getattr(getattr(controller, "plan", None), "scope", None)
            if not getattr(scope, "study_ids", []):
                hint += (" Each new question starts a separate analysis. If you mean an earlier study or summary, "
                         "name its study and trait, or paste the relevant summary. Replies here clarify this question.")
        else:
            who = "The analyst has the data and needs one more detail before it can answer"
            hint = "Your answer goes to the analyst; nothing new is fetched."
        if planner:
            shown += clarification_choices(questions, controller.metadata)
        return f"{who} (answer {number} of {controller.config.max_clarification_rounds}):\n\n{shown}\n\n{hint} Or reply **cancel**.", False
    if kind == "prompt":
        return str(payload).strip(), False
    if kind == "done":
        result = payload
        if not job.answer_shown:
            return answer_summary(controller), True
        review = {"accepted": "**Accepted.** Your review is recorded.",
                  "rejected": "**Rejected.** Your review is recorded; you can start a new question.",
                  "pending": "**Review pending.** This answer has not been accepted."}
        return review[result.human_acceptance] + " The Full report is available below.", True
    return f"The run stopped: {payload}", True


def _shorten_tables(markdown: str, keep: int | None = None) -> str:
    """The chat shows the first rows of a long table (136 location rows were one screen after another, 2026-09-29); answer.md
    keeps every row. A table here is a run of lines that start with '|': a header, a separator, then data rows."""
    keep = TABLE_ROWS_SHOWN if keep is None else keep
    out: list[str] = []
    rows = hidden = 0
    in_table = False
    for line in markdown.splitlines() + [""]:
        if line.startswith("|"):
            if not in_table:
                in_table, rows, hidden = True, -2, 0                      # the header and the separator are not data rows
            rows += 1
            if rows <= keep:
                out.append(line)
            else:
                hidden += 1
            continue
        if in_table and hidden:
            out += ["", f"*… and {hidden} more rows: all of them are in answer.md and the Full report panel below.*"]
        in_table = False
        out.append(line)
    return "\n".join(out[:-1])


def _shorten(text: str) -> str:
    if len(text) <= SHOWN_QUESTION_CHARS:
        return text
    return text[:SHOWN_QUESTION_CHARS].rstrip() + " … (shortened here; the run's notes keep the whole question)"


def respond(message: str, history: list[Message], state: dict[str, Any], *, make_config: Callable[[str], RunConfig] = live_config,
            make_models: Callable[[], Models] = default_models,
            preflight: Callable[[str], list[str]] | None = None) -> Iterator[tuple[list[Message], dict[str, Any], str]]:
    """One chat turn. A new question starts a run; while a run waits, the text is its reply. Yields (history, state, box)."""
    text = (message or "").strip()
    if not text:
        yield history, state, ""
        return
    history = list(history) + [{"role": "user", "content": text}]
    job: Job | None = state.get("job")
    if job is None:
        try:
            problems = (preflight or (lambda q: live_preflight(q, make_config)))(text)
        except Exception:  # a bad setting must not leave the chat locked or echo its value
            problems = ["App configuration could not be checked. Ask the operator to review the app and model settings."]
        if problems:
            history.append({"role": "assistant", "content": "This question cannot run yet:\n" + "\n".join(f"- {p}" for p in problems)})
            yield history, state, ""
            return
        try:
            job = Job(text, make_config=make_config, make_models=make_models).start()
        except Exception:
            history.append({"role": "assistant", "content": "This question could not start. Ask the operator to check the app settings and output-folder access."})
            yield history, state, ""
            return
        state = {"job": job}                 # a new question also clears the preceding report
    else:
        job.reply(text)
    history.append({"role": "assistant", "content": WORKING})
    yield history, state, ""
    content, finished = render(job, job.next_event())
    history[-1] = {"role": "assistant", "content": content}
    if job.answer_shown or finished:
        state = {**state, "full_report": job.controller.answer_markdown()}
    elif job.waiting_on.startswith("approve / edit / cancel"):
        scope_ids = ", ".join(job.controller.plan.scope.study_ids)
        state = {**state, "full_report": job.controller.approval_screen() + "\n\nAll matched study IDs: " + (scope_ids or "none")}
    elif job.waiting_on == CATALOG_PROMPT:
        state = {**state, "full_report": job.controller.catalog_approval_screen()}
    elif job.waiting_on == DATA_REVIEW_PROMPT:
        state = {**state, "full_report": job.controller.data_review_screen()}
    if finished:
        state = {k: v for k, v in state.items() if k != "job"}
    yield history, state, ""


# --------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------

def source_markdown() -> str:
    """Data-source details belong in the FAQ, not above the conversation."""
    settings = load_settings()
    return ("The assistant reads public records from the configured breeding database through **BrAPI**, "
            "a standard way to access breeding data. It does not edit the database.\n\n"
            f"Current data source: `{settings.base_url}`. Results depend on the records available there.")


def hosted_notice(environ: Mapping[str, str] | None = None) -> str:
    """Data-sharing policy for hosted demos, without displaying credentials or endpoint URLs."""
    values = os.environ if environ is None else environ
    if not values.get("SPACE_ID"):
        return ""
    try:
        host = (urlsplit(values.get("LLM_BASE_URL", "")).hostname or "").lower()
    except ValueError:
        host = ""
    provider = "OpenAI" if host == "api.openai.com" else "the configured model provider"
    return (f"Your questions and relevant catalog/data excerpts are sent to **{provider}**. "
            "The planner uses the model before database-fetch approval. Run records are saved on this demo's server; "
            "copy your full report before closing the page. Approval allows database requests; acceptance records your review.")


def full_report(state: dict[str, Any] | None) -> str:
    """Only text from this browser session, never a client-supplied path or another run's directory."""
    return str((state or {}).get("full_report", ""))


def _close_job(state: dict[str, Any]) -> None:
    job = (state or {}).get("job")
    if job is not None:
        job.close()


def reply_buttons(state: dict[str, Any] | None) -> tuple[bool, bool, bool, bool]:
    """Which quick replies fit what the run waits for: (Approve, Cancel, Accept, Reject). On 2026-09-29 '1032', 'done' and
    'accept' were typed where the app waited for approve / cancel; buttons show the valid replies (typing still works)."""
    job = (state or {}).get("job")
    waiting = job.waiting_on if job is not None else ""
    approval = waiting.startswith("approve / edit / cancel") or waiting in (CATALOG_PROMPT, DATA_REVIEW_PROMPT)
    answer = waiting.startswith("accept / reject")
    question = waiting in (CLARIFY_PROMPT, ANALYST_PROMPT, RETRIEVER_PROMPT, SUPERVISOR_PROMPT)
    return approval, approval or question, answer, answer


def locked():
    """While the app works: the input box (with its arrow), New question and the reply buttons are dimmed, so a message
    cannot overtake another. Order: box, New question, Approve, Cancel, Accept, Reject."""
    import gradio as gr

    return tuple(gr.update(interactive=False) for _ in range(6))


def after_step(state: dict[str, Any] | None):
    """The step is done: the box and New question work again, and only the reply buttons that fit are shown."""
    import gradio as gr

    job = (state or {}).get("job")
    review = job is not None and job.waiting_on == DATA_REVIEW_PROMPT
    buttons = reply_buttons(state)
    return (gr.update(interactive=True), gr.update(interactive=True),
            gr.update(interactive=True, visible=buttons[0], value="Continue to analysis" if review else "Approve"),
            *(gr.update(interactive=True, visible=show) for show in buttons[1:]))


def start_over(state: dict[str, Any] | None) -> tuple[list[Message], dict[str, Any], str]:
    """New question: a run that waits for a reply is ended (cancel; it is still recorded) and the chat is cleared."""
    _close_job(state or {})
    return [], {}, ""


def build_ui(*, make_config: Callable[[str], RunConfig] = live_config, make_models: Callable[[], Models] = default_models,
             preflight: Callable[[str], list[str]] | None = None):
    import gradio as gr

    with gr.Blocks(title="Breeding data assistant", fill_height=False) as demo:
        gr.HTML(HERO_HTML, elem_id="brapi-hero")
        with gr.Tabs(selected="assistant", elem_id="brapi-tabs"):
            with gr.Tab("Assistant", id="assistant", elem_id="brapi-assistant"):
                chat = gr.Chatbot(height="min(62vh, 620px)", min_height=350, show_label=False, layout="bubble",
                                  buttons=["copy"], placeholder=PLACEHOLDER, elem_id="brapi-chat")
                box = gr.Textbox(placeholder="Ask about a study, trait, year or location…", show_label=False,
                                 autofocus=False, lines=1, max_lines=5, scale=0, submit_btn=True, elem_id="brapi-composer")
                with gr.Row(elem_id="brapi-actions"):
                    approve = gr.Button("Approve", variant="primary", size="sm", scale=0, min_width=110, visible=False)
                    cancel = gr.Button("Cancel", variant="secondary", size="sm", scale=0, min_width=110, visible=False)
                    accept = gr.Button("Accept", variant="primary", size="sm", scale=0, min_width=110, visible=False)
                    reject = gr.Button("Reject", variant="secondary", size="sm", scale=0, min_width=110, visible=False)
                    new = gr.Button("New question", variant="secondary", size="sm", scale=0, min_width=140)
                gr.Examples(examples=[[e] for e in EXAMPLES], inputs=[box], label="Try a question", elem_id="brapi-examples")
                with gr.Accordion("Full report · sources & methods", open=False, elem_id="brapi-report"):
                    report = gr.Textbox(label="Full report", lines=12, max_lines=30, interactive=False, buttons=["copy"])
            with gr.Tab("FAQ & How to use", id="help", elem_id="brapi-help"):
                gr.Markdown("## A little guidance, when you need it.\n\n"
                            "Get started with a focused question, then review the evidence behind the answer.",
                            elem_id="brapi-help-intro")
                with gr.Row(elem_id="brapi-help-cards"):
                    with gr.Column(scale=3, min_width=280):
                        gr.Markdown(QUICK_START, elem_classes=["brapi-help-card"])
                    with gr.Column(scale=2, min_width=240):
                        gr.Markdown(QUESTION_GUIDE, elem_classes=["brapi-help-card"])
                for question, answer in faq_items(source_markdown(), data_sharing_policy=hosted_notice()):
                    with gr.Accordion(question, open=False, elem_classes=["brapi-faq-item"]):
                        gr.Markdown(answer)
        state = gr.State({}, time_to_live=REPLY_TIMEOUT_S + 60, delete_callback=_close_job)
        controls = [box, new, approve, cancel, accept, reject]              # the order locked() and after_step() use

        def on_submit(message, history, current):
            yield from respond(message, history or [], current or {}, make_config=make_config, make_models=make_models, preflight=preflight)

        def pressing(word: str):
            def on_press(history, current):                                # a button is the same as typing its word ...
                if not (current or {}).get("job"):                         # ... unless nothing waits any more (e.g. after 15 minutes)
                    yield (list(history or []) + [{"role": "assistant", "content": "Nothing is waiting for that reply now; ask a new question."}],
                           current or {}, "")
                    return
                yield from respond(word, history or [], current or {}, make_config=make_config, make_models=make_models, preflight=preflight)
            return on_press

        # One queue group for state-changing callbacks; hidden API endpoints add no authentication.
        event_options = {"concurrency_id": CHAT_CONCURRENCY_ID, "concurrency_limit": 1, "api_visibility": "private"}
        (box.submit(locked, None, controls, **event_options)
            .then(on_submit, inputs=[box, chat, state], outputs=[chat, state, box], **event_options)
            .then(full_report, inputs=[state], outputs=[report], **event_options)
            .then(after_step, inputs=[state], outputs=controls, **event_options))
        for button, word in ((approve, "approve"), (cancel, "cancel"), (accept, "accept"), (reject, "reject")):
            (button.click(locked, None, controls, **event_options)
                .then(pressing(word), inputs=[chat, state], outputs=[chat, state, box], **event_options)
                .then(full_report, inputs=[state], outputs=[report], **event_options)
                .then(after_step, inputs=[state], outputs=controls, **event_options))
        (new.click(start_over, inputs=[state], outputs=[chat, state, box], **event_options)
            .then(full_report, inputs=[state], outputs=[report], **event_options)
            .then(after_step, inputs=[state], outputs=controls, **event_options))
    return demo


def launch_options(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Validate hosting options without launching a server or exposing credentials in messages."""
    values = os.environ if environ is None else environ
    user, password = values.get("APP_USERNAME", ""), values.get("APP_PASSWORD", "")
    has_user, has_password = bool(user.strip()), bool(password.strip())
    if has_user != has_password:
        raise SystemExit("APP_USERNAME and APP_PASSWORD must both be set, or both be absent")
    is_space = bool(values.get("SPACE_ID"))
    open_on_purpose = values.get("APP_ALLOW_OPEN", "").strip().lower() == "true"
    if is_space and not (has_user and has_password) and not open_on_purpose:
        raise SystemExit("refusing to serve a Hugging Face Space without APP_USERNAME and APP_PASSWORD: anyone could spend the model key "
                         "(set APP_ALLOW_OPEN=true only after you verify that the Space is private)")
    host = values.get("GRADIO_SERVER_NAME", "").strip() or ("0.0.0.0" if is_space else "127.0.0.1")
    return {"server_name": host, "auth": (user, password) if has_user and has_password else None}
