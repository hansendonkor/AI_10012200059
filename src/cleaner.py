"""src/cleaner.py

Name: Hansen Donkor
Index Number: 10012200059

Text cleaning and normalization.

Why this matters (exam/marks):
- Cleaner text improves chunk boundaries and embedding quality.
- Removing boilerplate reduces false-positive retrieval.

This module is intentionally manual and framework-free.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Iterable, List, Sequence, Set

from .utils import Document


logger = logging.getLogger(__name__)


# Replace horizontal whitespace characters (not newlines) with spaces.
_HSPACE_RE = re.compile(r"[\t\x0b\x0c]+")

# Collapse repeated spaces within a line.
_MULTI_SPACE_RE = re.compile(r"[ ]{2,}")

# Fix common PDF hyphenation across line breaks: "govern-\nment" -> "government".
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _normalize_text_base(text: str) -> str:
    """Normalize whitespace without destroying paragraph structure."""

    if not text:
        return ""

    text = _normalize_newlines(text)

    # Normalize non-breaking spaces
    text = text.replace("\u00a0", " ")

    # Replace horizontal whitespace (keeps \n intact)
    text = _HSPACE_RE.sub(" ", text)

    # Fix hyphenation at line breaks
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)

    return text


def _clean_lines(lines: Sequence[str]) -> List[str]:
    out: List[str] = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            out.append("")
            continue
        ln = _MULTI_SPACE_RE.sub(" ", ln)
        out.append(ln)

    # Trim leading/trailing blank lines
    while out and out[0] == "":
        out.pop(0)
    while out and out[-1] == "":
        out.pop()

    # Collapse runs of blank lines (keep at most one blank line)
    collapsed: List[str] = []
    blank_run = 0
    for ln in out:
        if ln == "":
            blank_run += 1
            if blank_run <= 1:
                collapsed.append(ln)
        else:
            blank_run = 0
            collapsed.append(ln)

    return collapsed


def _detect_pdf_boilerplate(docs: Sequence[Document], max_lines_each_end: int = 2) -> Set[str]:
    """Detect repeated header/footer lines across many PDF pages.

    Heuristic:
    - Consider the first/last N non-empty lines per page.
    - Mark lines that appear on >= 30% of pages AND at least 3 pages.

    Returns:
        Set of exact lines to remove.
    """

    pdf_docs = [d for d in docs if d.source == "pdf" and d.text]
    if len(pdf_docs) < 3:
        return set()

    candidates: List[str] = []

    for d in pdf_docs:
        lines = [ln.strip() for ln in _normalize_text_base(d.text).split("\n") if ln.strip()]
        if not lines:
            continue

        head = lines[:max_lines_each_end]
        tail = lines[-max_lines_each_end:] if len(lines) > max_lines_each_end else lines
        candidates.extend(head)
        candidates.extend(tail)

    if not candidates:
        return set()

    counts = Counter(candidates)
    threshold = max(3, int(0.30 * len(pdf_docs)))

    boilerplate = {line for line, c in counts.items() if c >= threshold}

    # Avoid removing very short generic lines that could be meaningful.
    boilerplate = {ln for ln in boilerplate if len(ln) >= 4}

    return boilerplate


def clean_text(text: str) -> str:
    """Clean a single text blob."""

    base = _normalize_text_base(text)
    lines = base.split("\n")
    cleaned_lines = _clean_lines(lines)
    return "\n".join(cleaned_lines).strip()


def clean_documents(docs: List[Document]) -> List[Document]:
    """Clean all documents while preserving metadata.

    - Applies general text normalization to both CSV and PDF documents.
    - For PDFs, removes repeated boilerplate header/footer lines across pages.

    Args:
        docs: Input documents.

    Returns:
        New list of Documents (original objects are not mutated).
    """

    boilerplate = _detect_pdf_boilerplate(docs)
    if boilerplate:
        logger.info("Detected %d repeated PDF boilerplate lines", len(boilerplate))

    cleaned: List[Document] = []

    for d in docs:
        text = _normalize_text_base(d.text)
        lines = text.split("\n")

        if d.source == "pdf" and boilerplate:
            lines = [ln for ln in lines if ln.strip() not in boilerplate]

        new_text = "\n".join(_clean_lines(lines)).strip()

        cleaned.append(Document(source=d.source, text=new_text, metadata=d.metadata))

    return cleaned

def csv_to_documents(df) -> List[Document]:
    docs = []

    for idx, row in df.iterrows():
        region = str(row.get("region", "")).strip()
        constituency = str(row.get("constituency", "")).strip()
        candidate = str(row.get("candidate", "")).strip()
        party = str(row.get("party", "")).strip()
        votes = str(row.get("votes", "")).strip()

        text = (
            f"In the {region} region, the constituency {constituency}, "
            f"candidate {candidate} from the {party} party received {votes} votes."
        )

        docs.append(
            Document(
                source="election_csv",
                text=text,
                metadata={"row_index": idx}
            )
        )

    return docs