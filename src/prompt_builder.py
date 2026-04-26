"""src/prompt_builder.py

Name: Hansen Donkor
Index Number: 10012200059

Prompt engineering for the manual RAG pipeline.

Requirements satisfied:
- Prompt template that injects retrieved context
- Hallucination control (forces "I don't know" when evidence missing)
- Context window control (limits context size)

This module is intentionally simple and explainable for grading.
"""

from __future__ import annotations

from typing import List

from .utils import Chunk, approx_token_count


def _chunk_citation_label(chunk: Chunk) -> str:
    """Create a short citation label for a chunk."""

    if chunk.source == "pdf":
        page = chunk.metadata.get("page")
        section = chunk.metadata.get("section_title")
        if page and section:
            return f"Budget PDF p.{page} | {section}"
        if page:
            return f"Budget PDF p.{page}"
        return "Budget PDF"

    if chunk.source == "csv":
        row_index = chunk.metadata.get("row_index")
        region = chunk.metadata.get("region")
        constituency = chunk.metadata.get("constituency")
        parts = [p for p in [region, constituency] if p]
        suffix = f" | {' / '.join(parts)}" if parts else ""
        if row_index is not None:
            return f"Election CSV row {row_index}{suffix}"
        return f"Election CSV{suffix}"

    return str(chunk.source)


def _render_context_block(chunk: Chunk) -> str:
    label = _chunk_citation_label(chunk)
    return f"[{label}]\n{chunk.text.strip()}\n"


def select_chunks_to_budget(chunks: List[Chunk], max_context_tokens: int) -> List[Chunk]:
    """Select as many chunks as fit within the context budget.

    Note: This assumes chunks are already ordered by relevance.
    """

    if max_context_tokens <= 0:
        return []

    selected: List[Chunk] = []
    used = 0

    for ch in chunks:
        block = _render_context_block(ch)
        cost = approx_token_count(block)
        if selected and (used + cost) > max_context_tokens:
            break
        if not selected and cost > max_context_tokens:
            # Always include at least one chunk (even if truncated by budget logic).
            selected.append(ch)
            break
        selected.append(ch)
        used += cost

    return selected


def build_prompt(question: str, chunks: List[Chunk], max_context_tokens: int) -> str:
    """Build the final prompt sent to the LLM."""

    question = (question or "").strip()

    system_instructions = (
        "You are an Academic City AI Assistant.\n"
        "You MUST answer ONLY using the provided CONTEXT.\n"
        "If the answer is not supported by the context, reply exactly: "
        "'I don't know based on the provided documents.'\n"
        "Do NOT invent numbers, names, or policy details.\n"
        "When you use evidence, cite it inline using the bracketed labels (e.g., [Budget PDF p.12]).\n"
        "ALWAYS include the source citation in your answer.\n"
        "If possible, briefly explain how the answer was determined from the context.\n"
        "Keep the answer concise, factual, and slightly explanatory.\n"
    )

    # Apply context budget
    budgeted_chunks = select_chunks_to_budget(chunks, max_context_tokens=max_context_tokens)
    context = "\n".join(_render_context_block(ch) for ch in budgeted_chunks).strip()

    prompt = (
        f"SYSTEM\n{system_instructions}\n\n"
        f"CONTEXT\n{context}\n\n"
        f"QUESTION\n{question}\n\n"
        "ANSWER\n"
    )

    return prompt
