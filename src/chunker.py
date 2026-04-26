"""src/chunker.py

Name: Hansen Donkor
Index Number: 10012200059

Chunking strategies (manual RAG requirement).

Required strategies:
1) Fixed-size chunking with overlap
2) Paragraph/section-based chunking

Design goals:
- Preserve metadata end-to-end (id, source, page, row_index, etc.)
- Produce stable, explainable chunk IDs
- Keep logic framework-free and easy to grade
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence, Tuple

from .utils import Chunk, Document


_HEADING_RE = re.compile(r"^(CHAPTER|PART)\b|^[0-9]{1,2}\.?\s+[A-Z][A-Z\s\-]{3,}$")


def _fixed_chunks(text: str, chunk_size_chars: int, overlap_chars: int) -> List[Tuple[int, int, str]]:
    if chunk_size_chars <= 0:
        raise ValueError("chunk_size_chars must be > 0")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be >= 0")

    chunks: List[Tuple[int, int, str]] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(n, start + chunk_size_chars)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append((start, end, chunk_text))
        if end >= n:
            break
        start = max(0, end - overlap_chars)

    return chunks


def _is_heading(line: str) -> bool:
    if not line:
        return False

    if _HEADING_RE.search(line):
        return True

    # Heuristic: short-ish, mostly uppercase, no period
    if len(line) <= 80 and "." not in line:
        letters = [c for c in line if c.isalpha()]
        if letters:
            upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
            if upper_ratio >= 0.70:
                return True

    return False


def _split_into_sections_and_paragraphs(text: str) -> List[Tuple[str, List[str]]]:
    """Split text into (section_title, paragraphs).

    - Uses simple heading heuristics to start a new section.
    - Within each section, paragraphs are separated by blank lines.
    """

    lines = text.split("\n")
    sections: List[Tuple[str, List[str]]] = []

    current_title = "(no heading)"
    current_lines: List[str] = []

    def flush_section() -> None:
        nonlocal current_title, current_lines
        raw = "\n".join(current_lines).strip()
        if not raw:
            current_lines = []
            return

        # Split into paragraphs by blank lines (cleaner keeps single blank lines)
        paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
        if paragraphs:
            sections.append((current_title, paragraphs))
        current_lines = []

    for ln in lines:
        if _is_heading(ln.strip()):
            flush_section()
            current_title = ln.strip()
        else:
            current_lines.append(ln)

    flush_section()
    return sections


def _pack_paragraphs_into_chunks(
    paragraphs: Sequence[str],
    chunk_size_chars: int,
    overlap_chars: int,
) -> List[Tuple[int, str]]:
    """Pack paragraphs into chunks without breaking paragraph boundaries.

    Returns:
        List of (chunk_index, chunk_text)
    """

    if chunk_size_chars <= 0:
        raise ValueError("chunk_size_chars must be > 0")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be >= 0")

    if not paragraphs:
        return []

    chunks: List[Tuple[int, str]] = []
    buf: List[str] = []
    buf_len = 0
    chunk_idx = 0

    def buf_text(parts: Sequence[str]) -> str:
        return "\n\n".join(parts).strip()

    def compute_overlap_parts(parts: Sequence[str]) -> List[str]:
        if overlap_chars <= 0:
            return []
        out: List[str] = []
        total = 0
        for p in reversed(parts):
            p_len = len(p)
            if not out:
                out.append(p)
                total += p_len
                if total >= overlap_chars:
                    break
            else:
                # +2 for the "\n\n" joiner
                if total + 2 + p_len > overlap_chars and total > 0:
                    break
                out.append(p)
                total += 2 + p_len
        out.reverse()
        return out

    for p in paragraphs:
        p = p.strip()
        if not p:
            continue

        add_len = len(p) + (2 if buf else 0)
        if buf and (buf_len + add_len) > chunk_size_chars:
            # flush current buffer
            chunks.append((chunk_idx, buf_text(buf)))
            chunk_idx += 1

            # start next buffer with overlap paragraphs
            overlap_parts = compute_overlap_parts(buf)
            buf = list(overlap_parts)
            buf_len = len(buf_text(buf)) if buf else 0

        # add paragraph
        if buf:
            buf_len += 2 + len(p)
        else:
            buf_len += len(p)
        buf.append(p)

    if buf:
        chunks.append((chunk_idx, buf_text(buf)))

    return chunks


def chunk_documents(
    docs: List[Document],
    strategy: str,
    chunk_size_chars: int,
    overlap_chars: int,
) -> List[Chunk]:
    """Chunk a list of Documents into retrieval units.

    Args:
        docs: Documents produced by data_loader (+ cleaned by cleaner).
        strategy: "fixed" or "section".
            - fixed: sliding character window with overlap
            - section: paragraph/section-aware chunking (no paragraph splits)
        chunk_size_chars: Soft max size for chunk text.
        overlap_chars: Overlap size in characters (used for fixed; also used as a
            paragraph overlap budget for section strategy).
    """

    strategy = (strategy or "").strip().lower()
    if strategy not in {"fixed", "section"}:
        raise ValueError("strategy must be 'fixed' or 'section'")

    chunks: List[Chunk] = []

    for doc_i, d in enumerate(docs):
        if not d.text or not d.text.strip():
            continue

        doc_id = str(d.metadata.get("id", f"doc_{doc_i}"))

        # CSV rows are already atomic, so chunking just wraps metadata.
        if d.source == "csv":
            chunk_id = f"{doc_id}_c0"
            meta = {
                **d.metadata,
                "doc_id": doc_id,
                "strategy": strategy,
                "chunk_index": 0,
            }
            chunks.append(Chunk(chunk_id=chunk_id, source=d.source, text=d.text.strip(), metadata=meta))
            continue

        # PDF pages are chunked.
        if strategy == "fixed":
            spans = _fixed_chunks(
                d.text,
                chunk_size_chars=chunk_size_chars,
                overlap_chars=overlap_chars,
            )
            for ci, (start, end, chunk_text) in enumerate(spans):
                chunk_id = f"{doc_id}_fixed_{ci}"
                meta = {
                    **d.metadata,
                    "doc_id": doc_id,
                    "strategy": strategy,
                    "chunk_index": ci,
                    "char_start": start,
                    "char_end": end,
                }
                chunks.append(
                    Chunk(chunk_id=chunk_id, source=d.source, text=chunk_text, metadata=meta)
                )
        else:
            sections = _split_into_sections_and_paragraphs(d.text)
            for si, (title, paragraphs) in enumerate(sections):
                packed = _pack_paragraphs_into_chunks(
                    paragraphs,
                    chunk_size_chars=chunk_size_chars,
                    overlap_chars=overlap_chars,
                )
                for ci, chunk_text in packed:
                    chunk_id = f"{doc_id}_sec{si}_c{ci}"
                    meta = {
                        **d.metadata,
                        "doc_id": doc_id,
                        "strategy": strategy,
                        "section_title": title,
                        "section_index": si,
                        "chunk_index": ci,
                    }
                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            source=d.source,
                            text=chunk_text,
                            metadata=meta,
                        )
                    )

    return chunks
