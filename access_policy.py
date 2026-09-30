"""Read the operator's recorded review of the breeding database access policy."""
from __future__ import annotations

import re
from pathlib import Path

from brapi_client import PART2_DIR

__all__ = ["ACCESS_NOTES_PATH", "access_notes_review_date"]

ACCESS_NOTES_PATH = PART2_DIR / "ACCESS_NOTES.md"
_REVIEW_RE = re.compile(r"^Reviewed by Bidhan on:\s*(\d{4}-\d{2}-\d{2})\s*$", re.MULTILINE)


def access_notes_review_date(path: Path = ACCESS_NOTES_PATH) -> str | None:
    """The date on the 'Reviewed by Bidhan on: YYYY-MM-DD' line, or None if not reviewed."""
    if not path.is_file():
        return None
    match = _REVIEW_RE.search(path.read_text(encoding="utf-8"))
    return match.group(1) if match else None
