"""Presentation only: restrained botanical styling for the Gradio chat.

No settings, network calls, assets, JavaScript, or application state are read here.
The approval workflow and all answer/source content remain in app.py.
"""
from __future__ import annotations

HERO_HTML = """
<header class="brapi-hero">
  <div class="brapi-eyebrow">
    <svg class="brapi-mark" viewBox="0 0 28 28" fill="none" aria-hidden="true" focusable="false">
      <path d="M14 24V12M14 17C7 17 4 13 4 7c7 0 10 4 10 10ZM14 12C14 6 18 3 24 3c0 6-4 9-10 9Z"
            stroke="currentColor" stroke-width="1.65" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    <span>DESCRIPTIVE ANALYSIS</span>
  </div>
  <h1>Breeding data assistant</h1>
  <p class="brapi-intro">Find studies. Explore trait performance. Check the evidence.</p>
</header>
"""

CSS = """
.gradio-container {
  --brapi-page: #f6f7f1;
  --brapi-surface: #ffffff;
  --brapi-soft: #edf2e9;
  --brapi-line: #d8e2d4;
  --brapi-ink: #21382b;
  --brapi-muted: #536557;
  --brapi-green: #236044;
  --brapi-focus: #528866;
  max-width: 1000px !important;
  margin: 0 auto !important;
  padding: 28px 28px 36px !important;
  color: var(--brapi-ink);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif !important;
}
.dark .gradio-container, .gradio-container.dark {
  --brapi-page: #14221a;
  --brapi-surface: #1d3024;
  --brapi-soft: #263c2d;
  --brapi-line: #3b5441;
  --brapi-ink: #e6eee3;
  --brapi-muted: #b9cbb7;
  --brapi-green: #a1cfa9;
  --brapi-focus: #9cca9f;
}
#brapi-hero { margin: 0; padding: 0; }
.brapi-hero { padding: 2px 0 17px; }
.brapi-eyebrow {
  display: flex; align-items: center; gap: 9px;
  color: var(--brapi-green);
  font-size: 11px; font-weight: 700; letter-spacing: .14em;
}
.brapi-mark { width: 26px; height: 26px; flex: 0 0 26px; }
.brapi-hero h1 {
  color: var(--brapi-ink);
  font-family: Georgia, "Times New Roman", serif;
  font-size: clamp(29px, 4.1vw, 42px);
  font-weight: 500; line-height: 1.16; letter-spacing: -.025em;
  margin: 10px 0 9px;
}
.brapi-intro { margin: 0; color: var(--brapi-muted); font-size: 15px; line-height: 1.55; }
#brapi-tabs { margin-top: 0; }
#brapi-tabs [role="tablist"] { border-bottom: 1px solid var(--brapi-line); gap: 6px; padding-bottom: 10px; }
#brapi-tabs [role="tab"] {
  color: var(--brapi-muted); font-size: 14px; font-weight: 600;
  padding: 10px 17px; border-radius: 10px; border: 0;
}
#brapi-tabs [role="tab"][aria-selected="true"] {
  color: var(--brapi-green); background: var(--brapi-soft);
}
#brapi-assistant, #brapi-help { border: 0; padding: 16px 0 0; background: transparent; }
#brapi-help { gap: 12px; }
#brapi-help .prose { max-width: none; color: var(--brapi-ink); font-size: 14px; line-height: 1.75; }
#brapi-help .prose h2 { color: var(--brapi-ink); font-size: 23px; font-weight: 600; margin-bottom: 7px; }
#brapi-help .prose h3 { color: var(--brapi-green); font-size: 16px; margin: 0 0 12px; }
#brapi-help-intro { padding: 5px 0 7px; }
#brapi-help-intro p { color: var(--brapi-muted); margin: 0; }
#brapi-help-cards { gap: 16px; margin-bottom: 5px; }
.brapi-help-card.block {
  height: 100%; padding: 22px !important; border: 1px solid var(--brapi-line) !important;
  border-radius: 16px !important; background: var(--brapi-surface) !important;
}
.brapi-help-card li { margin-bottom: 10px; }
.brapi-faq-item { border: 1px solid var(--brapi-line) !important; border-radius: 12px !important; }
.brapi-faq-item button { color: var(--brapi-ink); }
.brapi-faq-item .prose { padding: 0 6px 5px; }
#brapi-help code { overflow-wrap: anywhere; white-space: pre-wrap; }
#brapi-chat {
  border: 1px solid var(--brapi-line);
  border-radius: 18px;
  background: var(--brapi-surface);
  box-shadow: 0 8px 28px rgba(28, 52, 32, .035);
  margin-top: 5px;
}
#brapi-chat .prose { font-size: 15px; line-height: 1.7; }
#brapi-chat .prose p { margin-top: .45em; margin-bottom: .65em; }
#brapi-chat .prose h1, #brapi-chat .prose h2, #brapi-chat .prose h3 {
  color: var(--brapi-ink); font-size: 17px; line-height: 1.4;
  margin-top: 1em; margin-bottom: .55em;
}
#brapi-chat .prose h1:first-child, #brapi-chat .prose h2:first-child,
#brapi-chat .prose h3:first-child { margin-top: 0; }
#brapi-chat .prose a { color: var(--brapi-green); text-decoration: underline; text-underline-offset: 3px; }
#brapi-chat .prose blockquote {
  border-left: 3px solid var(--brapi-focus);
  background: var(--brapi-soft);
  color: var(--brapi-ink);
  border-radius: 0 9px 9px 0;
  margin: 12px 0; padding: 10px 14px;
}
/* Approval text keeps every line. Wrapping prevents narrow-screen clipping. */
#brapi-chat pre {
  background: var(--brapi-soft) !important;
  color: var(--brapi-ink) !important;
  border: 1px solid var(--brapi-line);
  border-left: 3px solid var(--brapi-focus);
  border-radius: 10px;
  padding: 15px 17px !important;
  white-space: pre-wrap; overflow-wrap: anywhere;
  font-size: 12.5px; line-height: 1.75;
}
#brapi-chat pre code { white-space: inherit; color: inherit; font-size: inherit; }
#brapi-chat table { border-collapse: collapse; font-size: 13px; line-height: 1.55; }
#brapi-chat th, #brapi-chat td { padding: 9px 11px; border-color: var(--brapi-line); }
#brapi-chat th { background: var(--brapi-soft); text-align: left; font-weight: 650; }
#brapi-chat hr { border-color: var(--brapi-line); margin: 18px 0; }
#brapi-composer {
  border: 1px solid var(--brapi-line);
  border-radius: 14px;
  background: var(--brapi-surface);
  box-shadow: 0 3px 12px rgba(28, 52, 32, .025);
}
#brapi-composer:focus-within { border-color: var(--brapi-focus); box-shadow: 0 0 0 3px rgba(82, 136, 102, .12); }
#brapi-composer textarea { font-size: 15px; line-height: 1.55; padding: 14px 16px; }
#brapi-actions { gap: 8px; padding: 0; }
#brapi-actions button { border-radius: 9px; min-height: 39px; font-size: 13px; font-weight: 600; }
#brapi-actions button:disabled { opacity: .5; cursor: wait; }
#brapi-examples { padding: 8px 0 3px; }
#brapi-examples .label, #brapi-report label { color: var(--brapi-muted); }
#brapi-examples table { font-size: 12.5px; }
#brapi-report {
  border: 1px solid var(--brapi-line);
  border-radius: 12px;
  background: var(--brapi-surface);
  margin-top: 4px;
}
#brapi-report textarea {
  font-family: ui-monospace, "Cascadia Code", Consolas, monospace;
  font-size: 12px; line-height: 1.7;
}
.gradio-container button:focus-visible,
.gradio-container a:focus-visible,
.gradio-container summary:focus-visible {
  outline: 3px solid var(--brapi-focus); outline-offset: 3px;
}
@media (max-width: 640px) {
  .gradio-container { padding: 18px 12px 26px !important; }
  .brapi-hero { padding-bottom: 12px; }
  .brapi-hero h1 { font-size: 29px; }
  .brapi-intro { font-size: 14px; }
  #brapi-chat { border-radius: 13px; min-height: 300px !important; }
  #brapi-chat .prose { font-size: 14px; line-height: 1.65; }
  #brapi-chat pre { padding: 12px !important; font-size: 12px; }
  #brapi-actions button { min-height: 44px; }
  #brapi-composer textarea { font-size: 16px; padding: 12px; }
  #brapi-tabs [role="tab"] { padding: 10px 12px; font-size: 13px; }
  .brapi-help-card.block { padding: 17px !important; }
}
"""


def build_theme():
    """Construct the installed Gradio theme using system fonts only."""
    import gradio as gr

    return gr.themes.Soft(
        primary_hue="green", secondary_hue="green", neutral_hue="stone",
        font=[gr.themes.Font(name) for name in ("system-ui", "-apple-system", "Segoe UI", "sans-serif")],
        font_mono=[gr.themes.Font(name) for name in ("ui-monospace", "Cascadia Code", "Consolas", "monospace")],
    ).set(
        body_background_fill="#f6f7f1", body_background_fill_dark="#14221a",
        body_text_color="#21382b", body_text_color_dark="#e6eee3",
        body_text_color_subdued="#536557", body_text_color_subdued_dark="#b9cbb7",
        background_fill_primary="#ffffff", background_fill_primary_dark="#1d3024",
        background_fill_secondary="#edf2e9", background_fill_secondary_dark="#263c2d",
        block_background_fill="#ffffff", block_background_fill_dark="#1d3024",
        block_border_color="#d8e2d4", block_border_color_dark="#3b5441",
        block_shadow="none", block_shadow_dark="none",
        input_background_fill="#ffffff", input_background_fill_dark="#1d3024",
        input_border_color="#d8e2d4", input_border_color_dark="#3b5441",
        input_border_color_focus="#528866", input_border_color_focus_dark="#9cca9f",
        input_placeholder_color="#657463", input_placeholder_color_dark="#b2c2b0",
        button_primary_background_fill="#236044", button_primary_background_fill_dark="#a1cfa9",
        button_primary_background_fill_hover="#194d35", button_primary_background_fill_hover_dark="#b9ddbf",
        button_primary_border_color="#236044", button_primary_border_color_dark="#a1cfa9",
        button_primary_text_color="#ffffff", button_primary_text_color_dark="#14221a",
        button_secondary_background_fill="#ffffff", button_secondary_background_fill_dark="#1d3024",
        button_secondary_background_fill_hover="#edf2e9", button_secondary_background_fill_hover_dark="#263c2d",
        button_secondary_border_color="#d8e2d4", button_secondary_border_color_dark="#3b5441",
        button_secondary_text_color="#334e3c", button_secondary_text_color_dark="#d5e4d2",
        link_text_color="#236044", link_text_color_dark="#a1cfa9",
        code_background_fill="#edf2e9", code_background_fill_dark="#263c2d",
    )
