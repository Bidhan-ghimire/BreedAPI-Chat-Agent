"""Plain-language help for the shared Gradio interface; no settings or side effects."""

QUICK_START = """### Your first question

1. **Ask something focused.** Include a study name or ID and a trait. Add a year or location when it helps.
2. **Review the plan.** Answer any follow-up questions, then choose **Approve** for the data request you want to run.
3. **Check the evidence.** If a data preview appears, review it and choose **Continue to analysis**. Read the answer, then **Accept** or **Reject** it.

Open **Full report** below the conversation to see the sources and methods.
"""

QUESTION_GUIDE = """### Questions to try

- Which studies have a 2026 season?
- How many locations are in the catalog?
- In study *[name or ID]*, what is the mean of *[exact trait name]*?

Start with one study and one trait. You can explore a different question in a new run.
"""


def faq_items(source_details: str, *, data_sharing_policy: str = "") -> list[tuple[str, str]]:
    """Help shared by local and hosted versions; optional workflow steps are explicit."""
    return [
        ("What can I use this assistant for?",
         "Find studies and locations, check available traits, and summarize recorded measurements. "
         "It provides descriptive results such as counts, means and missing-data summaries. "
         "It does not make parental-selection or line-advancement recommendations."),
        ("Where do the data come from?", source_details),
        ("What makes a good question?",
         "Ask about one clear task at a time. Use a study name or ID and the exact trait name when you know them. "
         "Add a season, year or location to narrow a search. If you are unsure what is available, "
         "start by asking which studies or traits are listed."),
        ("What do Approve, Continue and Accept mean?",
         "**Approve** allows the specific database request shown in the chat. The first question may also ask "
         "for permission to prepare study and trait catalogs.\n\n"
         "**Continue to analysis**, when offered, confirms that you have reviewed the retrieved-data preview.\n\n"
         "**Accept** records that you reviewed the final answer and found it useful. It does not prove scientific correctness. "
         "Choose **Cancel** to stop a pending request or **Reject** if the answer does not meet your needs."),
        ("Why does the assistant ask another question?",
         "It may need to identify a study or trait, resolve an ambiguous request, or clarify an analysis. Reply in the chat. "
         "A clarification during retrieval cannot broaden the data access you already approved. "
         "To request different studies or traits, choose **Cancel** and start a **New question**."),
        ("Why might I not get an answer?",
         "The requested data may be missing, incomplete, outside the approved request, or unavailable within the app's limits. "
         "Check the explanation in the chat and try a narrower question. A request waiting for your reply ends after 15 minutes. "
         "**New question** clears the conversation and ends any waiting request."),
        ("Where can I check or keep the evidence?",
         "Open **Full report** below the conversation for the sources, methods and full answer. "
         "A retrieved-data preview, when offered, shows only a small sample of each table. "
         "Review the source and missing-data notes before using a result in your work. Copy the report to keep it. "
         "Switching between the Assistant and FAQ tabs keeps your current conversation."),
        ("Data sharing policy",
         data_sharing_policy or "Questions and relevant database excerpts are processed by the app's configured AI model. "
         "Model processing can begin before you approve a database request. "
         "Run records are saved on the computer or server running the app."),
    ]
